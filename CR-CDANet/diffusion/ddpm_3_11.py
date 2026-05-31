import math
import torch
from torch import nn
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from modelnet.mamba_crossattention_unet_3_11 import VMUNet
from torch.nn import functional as F
from diffusion.metrics_utils import *
import os
from transfer_learning.featureloss import FeatureLossCalculator

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    """
    生成带有预热（warm-up）阶段的 beta 值序列。

    参数:
    linear_start (float): 预热阶段起始的 beta 值。
    linear_end (float): 预热阶段结束后的 beta 值，也是整个序列中非预热阶段的 beta 值。
    n_timestep (int): 总的时间步数，即 beta 序列的长度。
    warmup_frac (float): 预热阶段占总时间步数的比例，范围为 [0, 1]。

    返回:
    numpy.ndarray: 形状为 (n_timestep,) 的 beta 值数组。

    说明:
    - 该函数用于生成一个 beta 值序列，其中前 `warmup_time` 个元素是线性增加的，从 `linear_start` 到 `linear_end`，
      而剩余的元素则全部为 `linear_end`。
    - 这种预热机制在扩散模型（Diffusion Model）中非常常见，有助于模型在训练初期更稳定地学习噪声添加过程。
    """

    # 初始化 betas 数组，所有元素初始值为 linear_end
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)

    # 计算预热阶段的时间步数
    warmup_time = int(n_timestep * warmup_frac)

    # 对于前 warmup_time 个元素，生成线性递增的 beta 值
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)

    # 返回最终的 betas 数组
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    根据指定的调度策略生成 beta 值序列。

    参数:
    schedule (str): 调度策略类型，可选值包括 'quad', 'linear', 'warmup10', 'warmup50', 'const', 'jsd', 'cosine'。
    n_timestep (int): 总的时间步数，即 beta 序列的长度。
    linear_start (float): 线性调度起始的 beta 值，默认为 1e-4。
    linear_end (float): 线性调度结束的 beta 值，默认为 2e-2。
    cosine_s (float): 余弦调度中的偏移量，默认为 8e-3。

    返回:
    numpy.ndarray: 形状为 (n_timestep,) 的 beta 值数组。

    说明:
    - 该函数用于根据不同的调度策略生成 beta 值序列，这些 beta 值在扩散模型（Diffusion Model）中用于控制噪声添加过程。
    - 每种调度策略对应不同的 beta 值生成方式，适用于不同的训练需求和场景。

    详细步骤:
    1. 根据 `schedule` 参数选择相应的调度策略。
    2. 对于每种调度策略，使用特定的方法生成 beta 值序列。
    3. 返回最终的 beta 值数组。
    """

    if schedule == 'quad':
        """
        二次线性调度（Quadratic Linear Schedule）
        - 生成从 `linear_start ** 0.5` 到 `linear_end ** 0.5` 的线性递增序列，然后平方。
        - 这种调度方式使得 beta 值的变化更加平滑。
        """
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2

    elif schedule == 'linear':
        """
        线性调度（Linear Schedule）
        - 生成从 `linear_start` 到 `linear_end` 的线性递增序列。
        - 这是最简单的调度方式，beta 值随时间线性增加。
        """
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)

    elif schedule == 'warmup10':
        """
        预热10%调度（Warmup 10% Schedule）
        - 使用 `_warmup_beta` 函数生成带有预热阶段的 beta 值序列，预热阶段占总时间步数的 10%。
        - 在预热阶段，beta 值线性增加到 `linear_end`，之后保持不变。
        """
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)

    elif schedule == 'warmup50':
        """
        预热50%调度（Warmup 50% Schedule）
        - 使用 `_warmup_beta` 函数生成带有预热阶段的 beta 值序列，预热阶段占总时间步数的 50%。
        - 在预热阶段，beta 值线性增加到 `linear_end`，之后保持不变。
        """
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)

    elif schedule == 'const':
        """
        常量调度（Constant Schedule）
        - 所有时间步的 beta 值都为 `linear_end`。
        - 这种调度方式适用于不需要变化的 beta 值的情况。
        """
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)

    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        """
        JSD 调度（Jensen-Shannon Divergence Schedule）
        - 生成从 1/T 到 1 的倒数序列。
        - 这种调度方式使得 beta 值在早期较大，在后期逐渐减小。
        """
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)

    elif schedule == "cosine":
        """
        余弦调度（Cosine Schedule）
        - 使用余弦函数生成 beta 值序列。
        - 通过调整 timesteps 和 alphas 来确保 beta 值在 [0, 1] 范围内，并且具有平滑的变化。
        """
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)

    else:
        raise NotImplementedError(f"Unknown schedule: {schedule}")

    return betas



# gaussian diffusion trainer class 高斯扩散训练器类

def exists(x):
    """
    检查给定的变量是否为 None。

    参数:
    x (Any): 要检查的变量。

    返回:
    bool: 如果变量不是 None，则返回 True；否则返回 False。

    说明:
    - 该函数用于简化对变量是否存在（即是否为 None）的检查。
    - 在许多情况下，特别是配置参数或可选输入时，需要检查某个值是否存在。此函数提供了一种简洁的方法来进行这种检查。
    """
    return x is not None


def default(val, d):
    """
    如果给定的值 `val` 存在（即不为 None），则返回 `val`；否则返回默认值 `d`。

    参数:
    val (Any): 要检查的值。
    d (Any or callable): 默认值。如果 `d` 是一个可调用对象（如函数），则调用它以获取默认值；否则直接使用 `d` 作为默认值。

    返回:
    Any: 如果 `val` 存在，则返回 `val`；否则返回 `d` 或者 `d()` 的结果。

    说明:
    - 该函数用于处理可选参数或配置项，确保在 `val` 不存在时有一个合理的默认值。
    - 如果 `d` 是一个函数或其他可调用对象，则会调用它来生成默认值，这使得默认值可以是动态生成的。
    """
    if exists(val):  # 使用 exists 函数检查 val 是否存在
        return val
    return d() if isfunction(d) else d  # 如果 d 是可调用对象，则调用它；否则直接返回 d



class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            channels=3,
            loss_type='l1',
            conditional=True,
            device=None
    ):
        """
        初始化高斯扩散模型。

        参数:
        model (nn.Module): 用于生成的神经网络模型。
        channels (int): 输入图像的通道数，默认为 3（RGB 图像）。
        loss_type (str): 损失函数类型，可选 'l1' 或 'l2'，默认为 'l1'。
        conditional (bool): 是否使用条件生成，默认为 True。
        device (torch.device): 计算设备（如 GPU 或 CPU），默认为 None。

        说明:
        - 初始化模型、设置损失函数类型、条件生成标志和计算设备。
        - 调用 `set_new_noise_schedule` 方法来初始化噪声调度参数。
        """
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.loss_type = loss_type
        self.conditional = conditional
        self.device = device
        self.set_new_noise_schedule()

    def set_loss(self):
        """
        设置损失函数。

        说明:
        - 根据 `loss_type` 参数选择 L1 或 L2 损失函数，并将其移动到指定设备。
        - 如果 `loss_type` 不是 'l1' 或 'l2'，则抛出 `NotImplementedError` 异常。
        """
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(self.device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(self.device)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def set_new_noise_schedule(self):
        """
        设置新的噪声调度参数。

        说明:
        - 使用 `make_beta_schedule` 函数生成 beta 值序列，并根据这些值计算各种与扩散过程相关的参数。
        - 将这些参数注册为缓冲区（buffer），以便在训练过程中可以访问它们。
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        # 生成 beta 值序列
        # 生成 beta 值序列，用于控制扩散过程中的噪声添加
        betas = make_beta_schedule(
                    schedule="linear",          # 使用线性调度策略
                    n_timestep=2000,            # 总的时间步数为 2000 步
                    linear_start=1e-4,          # 线性调度的起始 beta 值
                    linear_end=2e-2             # 线性调度的结束 beta 值
                )
        # 这里的 betas 是根据起始值 (linear_start)、结束值 (linear_end) 和总的时间步数 (n_timestep) 来一步步线性增加的。
        # 具体来说，make_beta_schedule 函数使用线性调度策略生成一个从 linear_start 到 linear_end 的线性递增序列。

        # 将 betas 转换为 NumPy 数组（如果它是一个 PyTorch 张量）
        # 这一步确保后续操作可以使用 NumPy 函数处理 betas
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas


        # 计算 alphas 和累积 alphas
        # 计算 alphas，即每一步的保留率（1 - beta）
        # 说明: alphas 表示每一步的保留率，beta 表示噪声添加率。alphas = 1 - betas
        alphas = 1. - betas

        # 计算 alphas 的累积乘积 alphas_cumprod
        # 说明: alphas_cumprod[t] 表示从时间步 0 到时间步 t 的所有 alpha 值的累积乘积。
        #      这用于计算前向扩散过程中的累积噪声方差。
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # 计算前一个时间步的累积乘积 alphas_cumprod_prev
        # 说明: alphas_cumprod_prev 是 alphas_cumprod 的前一个时间步版本。
        #      它将 1. 添加到 alphas_cumprod 的前面，并去掉最后一个元素。
        #      用于在反向生成过程中计算后验分布的参数。
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # 验证缓冲区长度
        assert len(alphas_cumprod) == 2000, f"alphas_cumprod length mismatch: {len(alphas_cumprod)}"
        assert len(alphas_cumprod_prev) == 2000, f"alphas_cumprod_prev length mismatch: {len(alphas_cumprod_prev)}"

        # 计算累积乘积的平方根 self.sqrt_alphas_cumprod_prev
        # 说明: 计算 alphas_cumprod 累积乘积并附加 1. 后的平方根。
        #      用于在反向生成过程中计算去噪步骤中的缩放因子。
        #      确保数值稳定性和计算效率。
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))


        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # 注册缓冲区，将各种与扩散过程相关的参数存储为类的持久状态
        self.register_buffer('betas', to_torch(betas))
        # 说明: 将 betas 数组转换为 PyTorch 张量并注册为缓冲区。
        #      betas 表示每个时间步的噪声添加率。

        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        # 说明: 将 alphas_cumprod 数组转换为 PyTorch 张量并注册为缓冲区。
        #      alphas_cumprod 表示从时间步 0 到当前时间步的所有 alpha 值的累积乘积。

        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        # 说明: 将 alphas_cumprod_prev 数组转换为 PyTorch 张量并注册为缓冲区。
        #      alphas_cumprod_prev 表示从时间步 0 到前一个时间步的所有 alpha 值的累积乘积。

        # 扩散过程 q(x_t | x_{t-1}) 和其他相关参数
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # 说明: 计算 alphas_cumprod 的平方根，并将其转换为 PyTorch 张量后注册为缓冲区。
        #      用于在扩散过程中计算缩放因子。

        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        # 说明: 计算 1 - alphas_cumprod 的平方根，并将其转换为 PyTorch 张量后注册为缓冲区。
        #      用于在扩散过程中计算噪声方差。

        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        # 说明: 计算 log(1 - alphas_cumprod)，并将其转换为 PyTorch 张量后注册为缓冲区。
        #      用于在扩散过程中计算对数噪声方差。

        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        # 说明: 计算 1 / alphas_cumprod 的平方根，并将其转换为 PyTorch 张量后注册为缓冲区。
        #      用于在反向生成过程中计算去噪步骤中的缩放因子。

        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        # 说明: 计算 (1 / alphas_cumprod - 1) 的平方根，并将其转换为 PyTorch 张量后注册为缓冲区。
        #      用于在反向生成过程中计算去噪步骤中的缩放因子。

        # 后验分布 q(x_{t-1} | x_t, x_0) 相关参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # 说明: 计算后验分布的方差，并将其转换为 PyTorch 张量后注册为缓冲区。
        #      用于在反向生成过程中计算后验分布的参数。

        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        # 说明: 计算后验分布方差的对数，并进行裁剪以避免数值不稳定（最小值为 1e-20），然后转换为 PyTorch 张量后注册为缓冲区。
        #      用于在反向生成过程中计算后验分布的参数。

        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        # 说明: 计算后验分布均值的第一个系数，并将其转换为 PyTorch 张量后注册为缓冲区。
        #      用于在反向生成过程中计算后验分布的均值。

        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        # 说明: 计算后验分布均值的第二个系数，并将其转换为 PyTorch 张量后注册为缓冲区。
        #      用于在反向生成过程中计算后验分布的均值。


    def predict_start_from_noise(self, x_t, t, noise):
        """
        这个不用！！！
        从噪声预测初始状态 x_0。

        参数:
        x_t (torch.Tensor): 时间步 t 的状态。
        t (int): 当前时间步。
        noise (torch.Tensor): 添加的噪声。

        - self.sqrt_recip_alphas_cumprod[t]：在时间步 t 处，累积乘积 alphas 的倒数平方根。
          这个值用于缩放当前时间步的图像 x_t，以部分恢复原始图像。
        - self.sqrt_recipm1_alphas_cumprod[t]：在时间步 t 处，(1 / alphas_cumprod - 1) 的平方根。
          这个值用于缩放噪声项，以去除添加的噪声。
        - x_t：在时间步 t 处的图像张量。
        - noise：在时间步 t 处添加的噪声张量
        返回:
        torch.Tensor: 预测的初始状态 x_0。
        """
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和对数方差。

        参数:
        x_start (torch.Tensor): 初始状态 x_0。
        x_t (torch.Tensor): 时间步 t 的状态。
        t (int): 当前时间步。

        返回:
        tuple: 后验分布的均值和对数方差。
        """

        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_recon, x_t, t):
        """
        计算预测分布 p(x_{t-1} | x_t) 的均值和对数方差。

        参数:
        x_recon (torch.Tensor): 预测的初始状态 x_0。
        x_t (torch.Tensor): 时间步 t 的状态。
        t (int): 当前时间步。

        返回:
        tuple: 预测分布的均值和对数方差。
        """
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_recon, t, x_t):
        """
        从预测分布 p(x_{t-1} | x_t) 中采样。

        参数:
        x_recon (torch.Tensor): 预测的初始状态 x_0。
        t (int): 当前时间步。
        x_t (torch.Tensor): 时间步 t 的状态。

        返回:
        torch.Tensor: 采样得到的状态 x_{t-1}。
        """
        model_mean, model_log_variance = self.p_mean_variance(x_recon=x_recon, t=t, x_t=x_t)
        noise = torch.randn_like(x_recon) if t > 0 else torch.zeros_like(x_recon)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, RGB, SAR, noise=None):
        """
        执行完整的反向采样循环以生成图像。

        参数:
        RGB (torch.Tensor): 输入的 RGB 图像。
        SAR (torch.Tensor): 输入的 SAR 图像。
        noise (torch.Tensor, optional): 可选的初始噪声，默认为 None。

        返回:
        torch.Tensor: 生成的图像。
        """

        # 获取批量大小、通道数、高度和宽度
        b, c, h, w = RGB.shape

        # 确保模型在正确的设备上（如 GPU 或 CPU）
        self.model = self.model.to(self.device)

        # 初始化噪声图像 img，形状与 SAR 图像相同，内容为随机噪声
        img = torch.randn_like(SAR).to(self.device)

        # 初始化一个空数组 pn，用于存储每个时间步的 PSNR 值（仅在非最后一个时间步计算）
        # pn = np.array([])

        # 反向遍历所有时间步
        for i in tqdm(reversed(range(1900, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if i == 1999:
                # 在第 1999 个时间步（即倒数第二个时间步），直接调用模型进行预测
                x_recon, features = self.model(
                    [RGB, img],
                    timesteps=torch.FloatTensor(np.repeat(self.sqrt_alphas_cumprod_prev[i], b)).view(b, 1).to(
                        self.device)
                )
                # 使用 p_sample 方法从预测分布中采样新的图像 img
                img = self.p_sample(x_recon=x_recon, x_t=img, t=i)
            else:
                # 对于其他时间步，先计算并记录当前时间步的 PSNR 值
                # pn = np.append(pn, psnr(x_recon.cpu().detach().numpy(), SAR.cpu().detach().numpy()))
                # aaa = psnr(x_recon.cpu().detach().numpy(), SAR.cpu().detach().numpy())
                # print("准确率：", aaa)
                # 调用模型进行预测
                x_recon, features= self.model(
                    [RGB, img],
                    timesteps=torch.FloatTensor(np.repeat(self.sqrt_alphas_cumprod_prev[i], b)).view(b, 1).to(
                        self.device)
                )
                # 使用 p_sample 方法从预测分布中采样新的图像 img
                img = self.p_sample(x_recon=x_recon, x_t=img, t=i)

        # 最后一个时间步（t=0），再次调用模型进行最终预测
        x_recon = self.model(
            [RGB, img],
            timesteps=torch.zeros([b, 1]).to(self.device)
        )

        # 返回最终生成的图像
        return x_recon

    @torch.no_grad()
    def super_resolution(self, RGB, SAR, noise=None):
        """

        参数:
        RGB (torch.Tensor): 输入的 RGB 图像。
        SAR (torch.Tensor): 输入的 SAR 图像。
        noise (torch.Tensor, optional): 可选的初始噪声，默认为 None。
        """

        return self.p_sample_loop(RGB, SAR, noise=noise)

    @torch.no_grad()
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        """
        从初始状态 x_0 生成带噪声的状态 x_t。

        参数:
        x_start (torch.Tensor): 初始状态 x_0。
        continuous_sqrt_alpha_cumprod (torch.Tensor): 连续的 sqrt(alpha_cumprod)。
        noise (torch.Tensor, optional): 可选的噪声，默认为 None。

        返回:
        torch.Tensor: 带噪声的状态 x_t。
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise)

    def p_losses(self, RGB, SAR):
        """
        计算损失。

        参数:
        RGB (torch.Tensor): 输入的 RGB 图像。
        SAR (torch.Tensor): 输入的 SAR 图像。

        返回:
        torch.Tensor: 计算得到的损失值。
        """
        x_start = SAR
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps)
        noise = torch.randn(b, c, h, w).to(self.device)

        # 从 x_0 到 x_t
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(np.repeat(self.sqrt_alphas_cumprod_prev[t], b)).to(
            self.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        x_noisy = self.q_sample(x_start=x_start,
                                continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
                                noise=noise)

        # 预测 x_0
        x_recon, intermediate_features = self.model([RGB, x_noisy], continuous_sqrt_alpha_cumprod)  # 只使用最终预测结果

        psnr_batch = psnr(x_recon.cpu().detach().numpy(), x_start.cpu().detach().numpy())
        print(f"当前块psnr值: {psnr_batch:.4f} dB")

        # 计算损失
        loss1 = self.loss_func(x_recon, SAR)
        return loss1, intermediate_features

    def forward(self, RGB, SAR):
        """
        前向传播方法，调用 `p_losses` 计算损失。

        参数:
        RGB (torch.Tensor): 输入的 RGB 图像。
        SAR (torch.Tensor): 输入的 SAR 图像。

        返回:
        torch.Tensor: 计算得到的损失值。
        """
        return self.p_losses(RGB, SAR)


# diffusion/ddpm_2_13.py
class SourceGaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            channels=3,
            loss_type='l1',
            conditional=True,
            device=None
    ):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.loss_type = loss_type
        self.conditional = conditional
        self.device = device
        self.set_new_noise_schedule()
        self.model.eval()  # 设置模型为评估模式，不参与梯度更新

    def set_loss(self):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(self.device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(self.device)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def set_new_noise_schedule(self):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = make_beta_schedule(
                    schedule="linear",
                    n_timestep=2000,
                    linear_start=1e-4,
                    linear_end=2e-2
                )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if noise.device != x_start.device:  # 如果 noise 和 x_start 设备不同
            x_start = x_start.to(noise.device)  # 将 x_start 移动到 noise 的设备
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.to(
                noise.device)  # 将 continuous_sqrt_alpha_cumprod 移动到 noise 的设备
        return (continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise)

    def p_losses(self, RGB, SAR):
        x_start = SAR
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps)
        noise = torch.randn(b, c, h, w).to(self.device)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(np.repeat(self.sqrt_alphas_cumprod_prev[t], b)).to(self.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        x_noisy = self.q_sample(x_start=x_start,
                                continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
                                noise=noise)

        x_recon, source_features = self.model([RGB, x_noisy], continuous_sqrt_alpha_cumprod)
        return source_features

    def forward(self, RGB, SAR):
        return self.p_losses(RGB, SAR)



class DDPM:
    def __init__(self, device, channels=3, out_channels=None, loss_type='l1', conditional=True, load_net=False, name=None):
        """
        初始化 DDPM 类。

        参数:
        device (torch.device): 计算设备，如 'cuda' 或 'cpu'。
        channels (int): 输入图像的通道数，默认为 102。
        out_channels (int, optional): 输出图像的通道数，默认为 None。
        loss_type (str): 损失函数类型，可选 'l1' 或 'l2'，默认为 'l1'。
        conditional (bool): 是否使用条件生成，默认为 True。
        load_net (bool): 是否加载预训练模型，默认为 False。
        name (str, optional): 模型文件的名称，默认为 None。
        """
        super(DDPM, self).__init__()
        # 定义网络并加载预训练模型
        self.device = device
        # print(device)
        model = VMUNet(
            input_channels=3,
            depths=[2, 2, 2],
            depths_decoder=[2, 2, 2],
            drop_path_rate=0.2,
            attn_drop_rate=0.2,
            load_ckpt_path=None
        ).to(device)

        source_model = VMUNet(
            input_channels=3,
            depths=[2, 2, 2],
            depths_decoder=[2, 2, 2],
            drop_path_rate=0.2,
            attn_drop_rate=0.2,
        ).to(device)

        self.netG = self.set_device(GaussianDiffusion(
            model,
            channels=channels,
            loss_type=loss_type,
            conditional=conditional,
            device=device
        ))

        self.source_netG = self.set_device(SourceGaussianDiffusion(
            source_model,
            channels=channels,
            loss_type=loss_type,
            conditional=conditional,
            device=device
        ))

        self.schedule_phase = None
        self.noise = None
        # 设置损失函数并加载恢复状态
        self.set_loss()
        self.set_new_noise_schedule()
        self.netG.train()
        # self.netG.eval()
        # 找到需要优化的参数
        self.source_netG.eval()  # 源域模型不参与训练
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=1e-4)
        if load_net:
            self.load_network(name)
        self.feature_loss_calculator = FeatureLossCalculator()

    def set_device(self, var):
        """
        将变量移动到指定设备。

        参数:
        var (torch.Tensor 或 nn.Module): 要移动的变量。

        返回:
        torch.Tensor 或 nn.Module: 移动到指定设备的变量。
        """
        var = var.to(self.device)
        return var

    def optimize_parameters(self, RGB, SAR, source_RGB, source_SAR):
        """
        优化参数。
        参数:
        RGB (torch.Tensor): 输入的 RGB 图像，形状为 (batch_size, channels, height, width)。
        SAR (torch.Tensor): 输入的 SAR 图像，形状为 (batch_size, channels, height, width)。
        返回:
        torch.Tensor: 计算得到的损失值。
        """
        self.netG.to(self.device)
        self.source_netG.to(self.device)
        self.optG.zero_grad()

        # 计算目标域的损失和中间特征
        target_loss, target_features = self.netG(RGB, SAR)
        # with torch.no_grad():  # 确保源域模型的参数不参与梯度更新
        #     # target_loss, target_features = self.netG(RGB, SAR)
        #     target_loss, target_features = self.netG(RGB, SAR)
        #     print(target_features[0].shape, target_features[1].shape, target_features[2].shape)

        # 计算源域的损失和中间特征
        with torch.no_grad():  # 确保源域模型的参数不参与梯度更新
            source_features = self.source_netG(source_RGB, source_SAR)
            # print(source_features[0].shape, source_features[1].shape, source_features[2].shape)

        # if torch.all(target_features[0] == source_features[0]):
        #     print("same")
        # else:
        #     print("not same")

        # 计算MMD损失
        total_coral_loss, total_mmd_loss, mmd_coral_loss = self.compute_mmd_coral_loss(source_features, target_features)

        # 总损失
        total_loss = target_loss + mmd_coral_loss

        total_loss.backward()
        self.optG.step()
        return total_loss, target_loss, mmd_coral_loss, total_mmd_loss, total_coral_loss

    def compute_mmd_coral_loss(self, source_features, target_features):
        total_coral_loss, total_mmd_loss, total_loss = self.feature_loss_calculator.calculate_total_loss(source_features, target_features)
        return total_coral_loss, total_mmd_loss, total_loss


    def test(self, RGB, SAR):
        """
        测试模型。

        参数:
        RGB (torch.Tensor): 输入的 RGB 图像，形状为 (batch_size, channels, height, width)。
        SAR (torch.Tensor): 输入的 SAR 图像，形状为 (batch_size, channels, height, width)。

        返回:
        torch.Tensor: 生成的超分辨率图像。
        """
        self.netG.eval()
        with torch.no_grad():
            self.SR , features= self.netG.super_resolution(
                RGB, SAR, noise=None)
        self.netG.train()
        return self.SR

    def sample(self, batch_size=1, continous=False):
        """
        采样生成图像。

        参数:
        batch_size (int): 生成图像的批量大小，默认为 1。
        continous (bool): 是否使用连续采样，默认为 False。
        """
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        """
        设置损失函数。
        """
        self.netG.set_loss()
        self.source_netG.set_loss()

    def set_new_noise_schedule(self):
        """
        设置新的噪声调度参数。
        """
        self.netG.set_new_noise_schedule()
        self.source_netG.set_new_noise_schedule()

    def get_current_visuals(self, need_LR=True, sample=False):
        """
        获取当前的可视化结果。

        参数:
        need_LR (bool): 是否需要低分辨率图像，默认为 True。
        sample (bool): 是否需要采样结果，默认为 False。
        """
        pass

    def print_network(self):
        """
        打印网络结构。
        """
        pass

    def save_network(self, epoch=False, name=None, save_dir=None):
        """
        保存网络模型。

        参数:
        epoch (bool): 是否保存特定 epoch 的模型，默认为 False。
        name (str, optional): 模型文件的名称，默认为 None。
        """
        network = self.netG
        network = network.to('cpu')
        state_dict = {'modelnet': network.state_dict()}

        if save_dir is None:
            print('扩散部分模型保存出现问题')
        # 使用传入的 save_dir 参数，如果没有传入则使用默认路径
        if save_dir is None:
            save_dir = '/media/disk_new/WHB/pytorch-stable-diffusion-main/model/ddpm_shuguang_SAR-RGB/'

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        if epoch:
            save_path = os.path.join(save_dir, '{}-{}.pt'.format(name, epoch))
        else:
            save_path = os.path.join(save_dir, '{}-best.pt'.format(name))

        torch.save(state_dict, save_path)

    def load_network(self, target_model_path, source_model_path=None):
        """
        加载网络模型。

        参数:
        target_model_path (str): 目标域模型文件的路径。
        source_model_path (str, optional): 源域模型文件的路径，默认为 None。
        """
        # 加载目标域模型
        state = torch.load(target_model_path)
        self.netG.load_state_dict(state['modelnet'])
        self.netG = self.set_device(self.netG)
        print(f"目标域模型已加载: {target_model_path}")

        # 加载源域模型
        if source_model_path is not None:
            state_source = torch.load(source_model_path)
            self.source_netG.load_state_dict(state_source['modelnet'])
            self.source_netG = self.set_device(self.source_netG)
            print(f"源域模型已加载: {source_model_path}")

