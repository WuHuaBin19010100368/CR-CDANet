import numpy as np
import math
import cv2



def psnr(img1, img2):
    """
    计算两个图像之间的峰值信噪比（PSNR）。

    参数:
    img1 (numpy.ndarray): 第一个图像，形状为 (batch_size, channels, height, width)
    img2 (numpy.ndarray): 第二个图像，形状为 (batch_size, channels, height, width)

    返回:
    float: 平均 PSNR 值

    说明:
    - PSNR 是一种衡量图像质量的常用指标，值越高表示图像质量越好。
    - 对于每个样本和每个通道，计算 MSE 和 PSNR，并返回所有样本和通道的平均 PSNR。
    """
    p = np.array([])  # 存储每个样本和通道的 PSNR 值
    for sample in range(img2.shape[0]):  # 遍历每个样本
        for band in range(img2.shape[1]):  # 遍历每个通道
            mse = np.mean((img1[sample][band] - img2[sample][band]) ** 2)  # 计算均方误差（MSE）
            if mse < 1.0e-10:  # 如果 MSE 接近零，返回最大值 100
                return 100
            p = np.append(p, 20 * math.log10(np.max(img2[sample][band]) / math.sqrt(mse)))  # 计算并存储 PSNR
    return np.mean(p)  # 返回所有样本和通道的平均 PSNR


def SAM(x_true, x_pred):
    """
    计算光谱角制图（Spectral Angle Mapper, SAM）。

    参数:
    x_true (numpy.ndarray): 真实图像，形状为 (batch_size, bands, height, width)
    x_pred (numpy.ndarray): 预测图像，形状为 (batch_size, bands, height, width)

    返回:
    float: 平均 SAM 角度（单位：度）

    说明:
    - SAM 用于评估两幅图像在光谱上的相似性，返回的角度越小表示相似度越高。
    - 计算每对样本的点积和范数，然后计算 arccos 得到角度，并处理 NaN 值。
    """
    dot_sum = np.sum(x_true * x_pred, axis=1)  # 计算点积
    norm_true = np.linalg.norm(x_true, axis=1)  # 计算真实图像的范数
    norm_pred = np.linalg.norm(x_pred, axis=1)  # 计算预测图像的范数
    res = np.arccos(dot_sum / (norm_pred * norm_true))  # 计算 SAM 角度
    is_nan = np.nonzero(np.isnan(res))  # 找出 NaN 值的位置
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0  # 将 NaN 值替换为 0
    return np.mean(res) * 180 / np.pi  # 返回平均 SAM 角度（单位：度）


def rmse(img1, img2):
    """
    计算均方根误差（Root Mean Squared Error, RMSE）。

    参数:
    img1 (numpy.ndarray): 第一个图像，形状为 (batch_size, channels, height, width)
    img2 (numpy.ndarray): 第二个图像，形状为 (batch_size, channels, height, width)

    返回:
    float: RMSE 值

    说明:
    - RMSE 衡量两个图像之间像素值的差异，值越小表示图像越相似。
    """
    mse = np.mean((img1 - img2) ** 2)  # 计算均方误差（MSE）
    rmse = math.sqrt(mse)  # 计算 RMSE
    return rmse


def ERGAS(x_pred, x_turth, d=4):
    """
    计算相对全局误差（Erdas Relative Global Accuracy Score, ERGAS）。

    参数:
    x_pred (numpy.ndarray): 预测图像，形状为 (batch_size, channels, height, width)
    x_turth (numpy.ndarray): 真实图像，形状为 (batch_size, channels, height, width)
    d (int): 缩放因子，默认为 4

    返回:
    float: 平均 ERGAS 值

    说明:
    - ERGAS 用于评估图像融合的质量，值越小表示融合效果越好。
    - 对于每个样本，计算每个通道的 RMSE 和均值，然后计算 ERGAS。
    """
    batches, channels, h, w = x_turth.shape  # 获取图像尺寸
    ergas = np.array([])  # 存储每个样本的 ERGAS 值
    for sample in range(batches):  # 遍历每个样本
        inner_sum = 0
        for channel in range(channels):  # 遍历每个通道
            band_img1 = x_pred[sample, channel, :, :]  # 获取预测图像的单通道数据
            band_img2 = x_turth[sample, channel, :, :]  # 获取真实图像的单通道数据
            rmse_value = rmse(band_img1, band_img2)  # 计算 RMSE
            m = np.mean(band_img2)  # 计算真实图像的均值
            inner_sum += (rmse_value / m) ** 2  # 累加平方项
        mean_sum = inner_sum / channels  # 计算平均值
        ergas = np.append(ergas, 100 * (d ** 2) * np.sqrt(mean_sum))  # 计算并存储 ERGAS
    return np.mean(ergas)  # 返回所有样本的平均 ERGAS


def CC_function1(A, F):
    """
    计算相关系数（Correlation Coefficient, CC），方法一。

    参数:
    A (numpy.ndarray): 第一个图像，形状为 (batch_size, bands, height, width)
    F (numpy.ndarray): 第二个图像，形状为 (batch_size, bands, height, width)

    返回:
    float: 平均 CC 值

    说明:
    - CC 衡量两个图像之间的线性相关性，值越接近 1 表示相关性越高。
    - 对于每个样本和每个通道，计算相关系数，并处理 NaN 值。
    """
    cc = np.array([])  # 存储每个样本和通道的 CC 值
    for i in range(A.shape[0]):  # 遍历每个样本
        for band in range(A.shape[1]):  # 遍历每个通道
            Aj = A[i][band] - np.mean(A[i][band])  # 计算去均值后的 A
            Fj = F[i][band] - np.mean(F[i][band])  # 计算去均值后的 F
            inner = np.sum(Aj * Fj)  # 计算内积
            mod1 = np.sum((A[i][band] - np.mean(A[i][band])) ** 2)  # 计算模长平方
            mod2 = np.sum((F[i][band] - np.mean(F[i][band])) ** 2)  # 计算模长平方
            mod = np.sqrt(mod1 * mod2)  # 计算模长乘积
            cc = np.append(cc, inner / mod)  # 计算并存储 CC
    for i in range(len(cc)):
        if np.isnan(cc[i]):  # 处理 NaN 值
            cc[i] = 1
    return np.mean(cc)  # 返回所有样本和通道的平均 CC


def CC_function2(ref, tar):
    """
    计算相关系数（Correlation Coefficient, CC），方法二。

    参数:
    ref (numpy.ndarray): 参考图像，形状为 (batch_size, bands, height, width)
    tar (numpy.ndarray): 目标图像，形状为 (batch_size, bands, height, width)

    返回:
    float: 平均 CC 值

    说明:
    - 使用 `np.corrcoef` 函数计算相关系数。
    - 对于每个样本和每个通道，计算相关系数，并返回所有样本和通道的平均 CC。
    """
    batch, bands, rows, cols = tar.shape  # 获取图像尺寸
    out = np.zeros((batch, bands))  # 初始化输出数组
    for b in range(batch):  # 遍历每个样本
        for i in range(bands):  # 遍历每个通道
            tar_tmp = tar[b, i, :, :].flatten()  # 获取目标图像的单通道数据并展平
            ref_tmp = ref[b, i, :, :].flatten()  # 获取参考图像的单通道数据并展平
            cc = np.corrcoef(tar_tmp, ref_tmp)  # 计算相关系数矩阵
            out[b, i] = cc[0, 1]  # 获取相关系数
    return np.mean(out)  # 返回所有样本和通道的平均 CC


def ssim(x_pred, x_truth):
    """
    计算结构相似性指数（Structural Similarity Index, SSIM）。

    参数:
    x_pred (numpy.ndarray): 预测图像，形状为 (batch_size, bands, height, width)
    x_truth (numpy.ndarray): 真实图像，形状为 (batch_size, bands, height, width)

    返回:
    float: 平均 SSIM 值

    说明:
    - SSIM 综合考虑亮度、对比度和结构信息来评估图像质量，值越接近 1 表示图像越相似。
    - 使用高斯滤波器计算局部均值和方差，并根据公式计算 SSIM。
    """
    C1 = (0.01 * 255) ** 2  # 第一个常数
    C2 = (0.03 * 255) ** 2  # 第二个常数
    batches, bands, h, w = x_truth.shape  # 获取图像尺寸
    x_pred = x_pred.astype(np.float64)  # 转换为浮点类型
    x_truth = x_truth.astype(np.float64)  # 转换为浮点类型
    kernel = cv2.getGaussianKernel(11, 1.5)  # 获取高斯核
    window = np.outer(kernel, kernel.transpose())  # 构建高斯窗口
    ssim_map = np.array([])  # 存储每个样本和通道的 SSIM 值
    for sample in range(batches):  # 遍历每个样本
        for band in range(bands):  # 遍历每个通道
            img1, img2 = x_pred[sample, band, :, :], x_truth[sample, band, :, :]  # 获取单通道数据
            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # 计算局部均值
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2  # 计算均值平方
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2  # 计算均值乘积
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq  # 计算方差
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2  # 计算协方差
            ssim_map = np.append(ssim_map, ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                                 ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))  # 计算并存储 SSIM
    return np.mean(ssim_map)  # 返回所有样本和通道的平均 SSIM
