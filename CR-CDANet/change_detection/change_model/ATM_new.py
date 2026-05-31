import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


# an alternative for mamba_ssm (in which causal_conv1d is needed)
# try:
#     from selective_scan import selective_scan_fn as selective_scan_fn_v1
#     from selective_scan import selective_scan_ref as selective_scan_ref_v1
# except:
#     pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


# 只使用了一次，将数据使用4*4的卷积和LN产生patch，然后转化为b*w*h*c
class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=2, in_chans=3, embed_dim=64, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 96
        self.dim_scale = dim_scale  # 4
        #        96             384
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        #                          24
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x

class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Multi-head Self Attention Module compatible with (B, H, W, C) input format.

        Args:
            dim (int): Number of channels (features) per token.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias in qkv linear layers.
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for output projection.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Input: (B, H, W, C)
        Output: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Flatten spatial dimensions and generate QKV
        x_flat = x.view(B, H * W, C)  # (B, N, C)
        qkv = self.qkv(x_flat).reshape(B, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each (B, N, H, D)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
    ):
        super().__init__()
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        # print('丢包率',self.dropout)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)  # 这里的z忘记了一个Linear吧
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

"""
*************************************************************************************************************************************************************************************************************************
*************************************************************************************************************************************************************************************************************************
*************************************************************************************************************************************************************************************************************************
"""


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        """
        初始化 SelfAttention 类的实例。

        参数:
        - n_heads (int): 多头注意力机制中的头数。每个头独立地计算查询（Q）、键（K）和值（V），然后将结果合并。
        - d_embed (int): 输入嵌入维度，即输入张量的最后一维大小。用于确定每个 token 的特征维度，并计算每个头的维度。
        - in_proj_bias (bool, 可选): 是否在输入投影层（in_proj）中使用偏置项，默认为 True。如果设置为 True，则在输入投影层的线性变换中会添加一个偏置向量。
        - out_proj_bias (bool, 可选): 是否在输出投影层（out_proj）中使用偏置项，默认为 True。如果设置为 True，则在输出投影层的线性变换中会添加一个偏置向量。

        成员变量:
        - self.in_proj (nn.Linear): 输入投影层，将输入张量从 d_embed 映射到 3 * d_embed，以便生成 Q、K 和 V 矩阵。
        - self.out_proj (nn.Linear): 输出投影层，将多头注意力的结果从 d_embed 映射回 d_embed。
        - self.n_heads (int): 存储传入的 n_heads 参数值，表示多头注意力机制中的头数。
        - self.d_head (int): 每个头的维度，即 d_embed // n_heads。确保每个头处理的数据量适中。
        """

        super().__init__()
        # 输入投影层，将输入张量从 d_embed 映射到 3 * d_embed，以便生成 Q、K 和 V 矩阵
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # 输出投影层，将多头注意力的结果从 d_embed 映射回 d_embed
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        # 多头注意力机制中的头数
        self.n_heads = n_heads

        # 每个头的维度，即 d_embed // n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        """
        前向传播函数，计算自注意力机制的输出。

        参数:
        - x: 输入张量，形状为 (Batch_Size, Seq_Len, Dim)
        - causal_mask: 是否使用因果掩码，默认为 False。用于防止未来信息泄露（如在 Transformer 解码器中）

        返回:
        - output: 自注意力机制的输出张量，形状为 (Batch_Size, Seq_Len, Dim)
        """

        # 获取输入张量的形状 (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        # 分解输入张量的形状
        batch_size, sequence_length, d_embed = input_shape

        # 计算中间形状 (Batch_Size, Seq_Len, H, Dim / H)，其中 H 是头数
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # 将输入张量通过输入投影层，并将其分成三个部分：Q、K 和 V
        # 形状变化：(Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # 将 Q、K 和 V 的形状调整为 (Batch_Size, Seq_Len, H, Dim / H)，然后转置为 (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # 计算注意力权重矩阵
        # 形状变化：(Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        # 如果启用因果掩码，则对上三角矩阵进行掩码处理，防止未来信息泄露
        if causal_mask:
            # 创建一个与 weight 形状相同的布尔掩码矩阵，上三角部分为 True
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # 将上三角部分填充为 -inf，使得 softmax 后这些位置的概率接近于 0
            weight.masked_fill_(mask, -torch.inf)

        # 对注意力权重矩阵除以 sqrt(d_k)，即每个头的维度的平方根
        # 形状不变：(Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head)

        # 应用 softmax 函数，将权重归一化为概率分布
        # 形状不变：(Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1)

        # 计算加权和，得到最终的输出张量
        # 形状变化：(Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # 将输出张量的形状调整回 (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # 将输出张量的形状调整回 (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # 通过输出投影层，将输出张量映射回原始的嵌入维度
        # 形状不变：(Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        # 返回最终的输出张量
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        # self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        """
        初始化 CrossAttention 模块。

        参数:
        n_heads (int): 注意力头的数量。
        d_embed (int): 嵌入维度（即查询、键和值的维度）。
        d_cross (int): 上下文向量的维度。
        in_proj_bias (bool): 是否在输入投影层中使用偏置，默认为 True。
        out_proj_bias (bool): 是否在输出投影层中使用偏置，默认为 True。
        """
        super().__init__()
        # 定义查询、键和值的线性投影层，用于将输入特征映射到注意力机制所需的维度。
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        # 定义输出投影层，用于将多头注意力的结果映射回原始维度。
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        # 设置注意力头的数量和每个头的维度。
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        """
        前向传播函数，对输入特征图和上下文向量进行处理。

        参数:
        x (torch.Tensor): 查询张量（latent），形状为 (Batch_Size, Seq_Len_Q, Dim_Q)。
        y (torch.Tensor): 上下文张量（context），形状为 (Batch_Size, Seq_Len_KV, Dim_KV)。

        返回:
        torch.Tensor: 处理后的特征图，形状为 (Batch_Size, Seq_Len_Q, Dim_Q)。
        """
        # 获取输入张量的形状信息。
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # 计算中间形状，以便将嵌入维度分成多个头。
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # 对查询、键和值进行线性变换。
        # q: (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # k: (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # v: (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # 将查询、键和值重新排列成多头格式。
        # q: (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2)
        # k: (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)
        # v: (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)

        # 计算注意力权重矩阵。
        # weight: (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)

        # 缩放注意力权重矩阵以稳定梯度。
        # weight: (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)

        # 应用 Softmax 函数计算归一化的注意力权重。
        # weight: (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)

        # 使用注意力权重加权求和得到输出。
        # output: (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v

        # 将多头输出重新排列回原始形状。
        # output: (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()

        # output: (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)

        # 应用输出投影层将多头注意力的结果映射回原始维度。
        # output: (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # 返回最终的输出张量，形状为 (Batch_Size, Seq_Len_Q, Dim_Q)。
        return output


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        """
        初始化 UNET_ResidualBlock 模块。

        参数:
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 输出特征图的通道数。
        n_time (int): 时间嵌入的维度，默认为 1280。
        """
        super().__init__()
        # 定义一个 GroupNorm 层，用于对输入特征图进行归一化处理。
        # 这里使用 32 个组来进行归一化。
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)

        # 定义一个卷积层，用于将输入特征图的通道数从 in_channels 转换为 out_channels。
        # 使用 3x3 卷积核，并且 padding=1，以保持特征图的空间尺寸不变。
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 定义一个线性层，用于将时间嵌入从 n_time 维度转换为 out_channels 维度。
        self.linear_time = nn.Linear(n_time, out_channels)

        # 定义一个 GroupNorm 层，用于对合并后的特征图进行归一化处理。
        # 这里使用 32 个组来进行归一化。
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)

        # 定义一个卷积层，用于进一步处理合并后的特征图。
        # 使用 3x3 卷积核，并且 padding=1，以保持特征图的空间尺寸不变。
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 如果输入和输出通道数相同，则使用恒等变换作为残差连接。
        # 否则，使用一个 1x1 卷积层将输入通道数转换为输出通道数。
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        """
        前向传播函数，对输入特征图和时间嵌入进行处理。

        参数:
        feature (torch.Tensor): 输入特征图，形状为 (Batch_Size, In_Channels, Height, Width)。
        time (torch.Tensor): 时间嵌入，形状为 (1, 1280)。

        返回:
        torch.Tensor: 处理后的特征图，形状为 (Batch_Size, Out_Channels, Height, Width)。
        """
        # 保存输入特征图作为残差连接的输入。
        residue = feature

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        # 使用 GroupNorm 层对输入特征图进行归一化处理。
        feature = self.groupnorm_feature(feature)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        # 使用 SiLU 激活函数对归一化后的特征图进行非线性变换。
        feature = F.silu(feature)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 使用卷积层将特征图的通道数从 in_channels 转换为 out_channels。
        feature = self.conv_feature(feature)

        # (1, 1280) -> (1, 1280)
        # 使用 SiLU 激活函数对时间嵌入进行非线性变换。
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        # 使用线性层将时间嵌入从 n_time 维度转换为 out_channels 维度。
        time = self.linear_time(time)

        # 将时间嵌入扩展到与特征图的空间维度相同。
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 使用 GroupNorm 层对合并后的特征图进行归一化处理。
        merged = self.groupnorm_merged(merged)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 使用 SiLU 激活函数对归一化后的特征图进行非线性变换。
        merged = F.silu(merged)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 使用卷积层进一步处理合并后的特征图。
        merged = self.conv_merged(merged)

        # 将处理后的特征图与残差连接的输入相加。
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        # 遍历序列中的每一层
        for layer in self:
            # 如果当前层是 UNET_AttentionBlock 类型
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)  # 传递 x 和 context 给 UNET_AttentionBlock 层
            # 如果当前层是 UNET_ResidualBlock 类型
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)  # 传递 x 和 time 给 UNET_ResidualBlock 层
            else:
                x = layer(x)  # 对于其他类型的层，仅传递 x
        return x  # 返回最终处理后的特征图

"""
*************************************************************************************************************************************************************************************************************************
*************************************************************************************************************************************************************************************************************************
*************************************************************************************************************************************************************************************************************************
"""

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        """
        初始化 UNET_AttentionBlock 模块。

        参数:
        n_head (int): 注意力头的数量。
        n_embd (int): 每个注意力头的嵌入维度。
        """
        super().__init__()
        channels = n_head * n_embd

        # 定义一个 GroupNorm 层，用于对输入特征图进行归一化处理。
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)

        # 定义一个 1x1 卷积层，用于调整输入特征图的通道数。
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # 定义多个 LayerNorm 层和注意力机制模块。
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.layernorm_3 = nn.LayerNorm(channels)

        # 定义 GeGLU 和线性变换层。
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        # 定义一个 1x1 卷积层，用于调整输出特征图的通道数。
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        """
        前向传播函数，对输入特征图和上下文向量进行处理。

        这里输入的是context (torch.Tensor): 上下文向量，形状为 (Batch_Size, Features, Height, Width)。

        返回:
        torch.Tensor: 处理后的特征图，形状为 (Batch_Size, Features, Height, Width)。
        """

        # 保存输入特征图作为最终残差连接的输入。
        residue_long = x

        # 使用 GroupNorm 层对输入特征图进行归一化处理。
        x = self.groupnorm(x)

        # 使用 1x1 卷积层调整输入特征图的通道数。
        x = self.conv_input(x)

        # 获取输入特征图的形状信息。
        n, c, h, w = x.shape

        # 将特征图从 (Batch_Size, Features, Height, Width) 转换为 (Batch_Size, Features, Height * Width)。
        x = x.view((n, c, h * w))

        # 将特征图从 (Batch_Size, Features, Height * Width) 转换为 (Batch_Size, Height * Width, Features)。
        x = x.transpose(-1, -2)

        # 第一部分：自注意力机制 + 残差连接
        # 保存当前特征图作为短残差连接的输入。
        residue_short = x

        # 使用 LayerNorm 层对特征图进行归一化处理。
        x = self.layernorm_1(x)

        # 应用自注意力机制。
        x = self.attention_1(x)

        # 将自注意力机制的结果与短残差连接相加。
        x += residue_short

        # 第二部分：交叉注意力机制 + 残差连接
        # 保存当前特征图作为短残差连接的输入。
        residue_short = x

        # 使用 LayerNorm 层对特征图进行归一化处理。
        x = self.layernorm_2(x)

        # 使用 GeGLU 激活函数，将特征图分为两部分并进行元素级乘法。
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        # 使用线性变换层将特征图的维度从 4 * channels 转换回 channels。
        x = self.linear_geglu_2(x)

        # 将 FFN 的结果与短残差连接相加。
        x += residue_short

        # 将特征图从 (Batch_Size, Height * Width, Features) 转换为 (Batch_Size, Features, Height * Width)。
        x = x.transpose(-1, -2)

        # 将特征图从 (Batch_Size, Features, Height * Width) 转换为 (Batch_Size, Features, Height, Width)。
        x = x.view((n, c, h, w))

        # 最终残差连接：将初始输入与输出相加。
        return self.conv_output(x) + residue_long


class VSSBlock_context(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            drop_path: float = 0,  # 0.2
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            attn_drop_rate: float = 0,  # 0
            d_state: int = 16,
            n_head: int = 8,  # 注意力头的数量
            n_embd: int = 12,  # 每个注意力头的嵌入维度，假设 n_head * n_embd = 96
            d_context: int = 3,  # 上下文向量的维度
            first_block: bool = False,  # 是否是第一个块
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 96             0.2                   16
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.self_attention = MHSA(dim=hidden_dim, num_heads=8, attn_drop=attn_drop_rate)
        self.drop_path = DropPath(drop_path)

        # 添加 UNET_AttentionBlock
        self.unet_attention_block = UNET_AttentionBlock(n_head=n_head, n_embd=n_embd)
        self.first_block = first_block

    def forward(self, context: torch.Tensor):
        # 先通过 UNET_AttentionBlock，仅在第一个块中执行
        if self.first_block:
            input_1 = context.permute(0, 3, 1, 2)
            # print('应该输入的形状是', input_1.shape)
            x1 = self.unet_attention_block(input_1)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # x1 = input_1
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            x1 = x1.permute(0, 2, 3, 1)
            # print('条件第一层')
        else:
            x1 = context
            # print('条件第二层')

        # 最后通过原始的 VSSBlock
        x2 = x1 + self.drop_path(self.self_attention(self.ln_1(x1)))
        # x2 = context + self.drop_path(self.self_attention(self.ln_1(context)))
        return x2


class VSSBlock_x_t(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            drop_path: float = 0,  # 0.2
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            attn_drop_rate: float = 0,  # 0
            d_state: int = 16,
            n_head: int = 8,  # 注意力头的数量
            n_embd: int = 12,  # 每个注意力头的嵌入维度，假设 n_head * n_embd = 96
            d_context: int = 3,  # 上下文向量的维度
            first_block: bool = False,  # 是否是第一个块
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 96             0.2                   16
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.self_attention = MHSA(dim=hidden_dim, num_heads=8, attn_drop=attn_drop_rate)
        self.drop_path = DropPath(drop_path)

        # 添加 UNET_AttentionBlock
        self.unet_attention_block = UNET_AttentionBlock(n_head=n_head, n_embd=n_embd)
        self.first_block = first_block

    def forward(self, context: torch.Tensor):
        # 先通过 UNET_AttentionBlock，仅在第一个块中执行
        if self.first_block:
            input_1 = context.permute(0, 3, 1, 2)
            # print('应该输入的形状是', input_1.shape)
            x1 = self.unet_attention_block(input_1)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # x1 = input_1
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            x1 = x1.permute(0, 2, 3, 1)
            # print('条件第一层')
        else:
            x1 = context
            # print('条件第二层')

        # 最后通过原始的 VSSBlock
        x2 = x1 + self.drop_path(self.self_attention(self.ln_1(x1)))
        # x2 = context + self.drop_path(self.self_attention(self.ln_1(context)))
        return x2


class VSSBlock_decoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            drop_path: float = 0,  # 0.2
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            attn_drop_rate: float = 0,  # 0
            d_state: int = 16,
            n_head: int = 8,  # 注意力头的数量
            n_embd: int = 12,  # 每个注意力头的嵌入维度，假设 n_head * n_embd = 96
            first_block: bool = False,  # 是否是第一个块
    ):

        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 96             0.2                   16
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.self_attention = MHSA(dim=hidden_dim, num_heads=8, attn_drop=attn_drop_rate)
        self.drop_path = DropPath(drop_path)

        # 添加 UNET_AttentionBlock
        self.unet_attention_block = UNET_AttentionBlock(n_head=n_head, n_embd=n_embd)
        self.first_block = first_block
    def forward(self, input: torch.Tensor):

        if self.first_block:
            input_1 = input.permute(0, 3, 1, 2)
            x1 = self.unet_attention_block(input_1)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # x1 = input_1
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            x1 = x1.permute(0, 2, 3, 1)
            # print('解码器第一层')
        else:
            x1 = input
            # print('解码器第二层')

        x = x1 + self.drop_path(self.self_attention(self.ln_1(x1)))
        # x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer_context(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(  # 以第一个为例
            self,
            dim,  # # 96
            depth,  # 2
            d_state=16,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,  # 每一个模块都有一个drop
            norm_layer=nn.LayerNorm,
            downsample=None,  # PatchMergin2D
            use_checkpoint=False,
    ):

        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock_context(
                hidden_dim=dim,  # 96
                drop_path=drop_path[i],  # 0.2
                norm_layer=norm_layer,  # nn.LN
                attn_drop_rate=attn_drop,  # 0
                d_state=d_state,  # 16
                n_head=8,  # 注意力头的数量
                n_embd=dim // 8,  # 每个注意力头的嵌入维度，假设 n_head * n_embd = dim
                d_context=3,  # 上下文向量的维度
                first_block=(i == 0),  # 只在第一个块中执行 UNET_ResidualBlock 和 UNET_AttentionBlock
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, context):
        for blk in self.blocks:
            context = blk(context)

        if self.downsample is not None:
            context = self.downsample(context)

        return context

class VSSLayer_x_t(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(  # 以第一个为例
            self,
            dim,  # # 96
            depth,  # 2
            d_state=16,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,  # 每一个模块都有一个drop
            norm_layer=nn.LayerNorm,
            downsample=None,  # PatchMergin2D
            use_checkpoint=False,
    ):

        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock_context(
                hidden_dim=dim,  # 96
                drop_path=drop_path[i],  # 0.2
                norm_layer=norm_layer,  # nn.LN
                attn_drop_rate=attn_drop,  # 0
                d_state=d_state,  # 16
                n_head=8,  # 注意力头的数量
                n_embd=dim // 8,  # 每个注意力头的嵌入维度，假设 n_head * n_embd = dim
                d_context=3,  # 上下文向量的维度
                first_block=(i == 0),  # 只在第一个块中执行 UNET_ResidualBlock 和 UNET_AttentionBlock
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, context):
        for blk in self.blocks:
            context = blk(context)

        if self.downsample is not None:
            context = self.downsample(context)

        return context


class VSSLayer_decoder(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock_decoder(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                n_head=8,  # 注意力头的数量
                n_embd=dim // 8,  # 每个注意力头的嵌入维度，假设 n_head * n_embd = dim
                first_block=(i == 0),  # 只在第一个块中执行 UNET_ResidualBlock 和 UNET_AttentionBlock
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 2], depths_decoder=[2, 2, 2],
                 dims=[64, 128, 256], dims_decoder=[256, 128, 64], d_state=16, drop_rate=0.1,
                 attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes  # 输出通道数！！！！
        self.num_layers = len(depths)  # 编码器层数，这里是4
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]  # 嵌入维度，这里是96
        self.num_features = dims[-1]  # 最终特征维度，这里是768
        self.dims = dims  # 各层维度，[96, 192, 384, 768]

        # 4*4卷积+LN-> b*w*h*c
        # 输入: (B, C, H, W) -> 输出: (B, H/4, W/4, embed_dim)
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # Dropout层
        # 输入: (B, H/4, W/4, embed_dim) -> 输出: (B, H/4, W/4, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 生成对应的sum(depths)随机深度衰减数值 dpr是正序，dpr_decoder是倒序（用到了[start:end:-1] 反向步长）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        """ 
        编码器部分
        假设输入图像的原始尺寸为 (B, 3, 256, 256)：
        初始输入:
        输入: (B, 3, 256, 256)
        经过 patch_embed 层:
        输入: (B, 3, 256, 256)
        输出: (B, 64, 64, 96)
        经过 pos_drop 层:
        输入: (B, 64, 64, 96)
        输出: (B, 64, 64, 96)
        编码器部分:
        第一层 VSSLayer (i_layer = 0):
        输入: (B, 64, 64, 96)
        输出: (B, 32, 32, 192)
        第二层 VSSLayer (i_layer = 1):
        输入: (B, 32, 32, 192)
        输出: (B, 16, 16, 384)
        第三层 VSSLayer (i_layer = 2):
        输入: (B, 16, 16, 384)
        输出: (B, 8, 8, 768)
        第四层 VSSLayer (i_layer = 3):
        输入: (B, 8, 8, 768)
        输出: (B, 4, 4, 768)
        """
        # 编码器部分
        self.layers_context = nn.ModuleList()
        self.layers_x_t = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 以第一个为例 num_layers = 4
            # 输入: (B, H/4, W/4, embed_dim) -> 输出: (B, H/(4*2^i), W/(4*2^i), dims[i_layer])
            layer_context = VSSLayer_context(
                dim=dims[i_layer],  # 当前层维度，例如96
                depth=depths[i_layer],  # 当前层深度，例如2
                d_state=d_state,  # 状态维度，例如16
                drop=drop_rate,  # Dropout率，例如0
                attn_drop=attn_drop_rate,  # 注意力Dropout率，例如0
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # 每一个模块传一个概率值
                norm_layer=norm_layer,  # 归一化层，例如nn.LayerNorm
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,  # 下采样层
                use_checkpoint=use_checkpoint,  # 是否使用检查点
            )
            self.layers_context.append(layer_context)

            layer_x_t = VSSLayer_x_t(
                dim=dims[i_layer],  # 当前层维度，例如96
                depth=depths[i_layer],  # 当前层深度，例如2
                d_state=d_state,  # 状态维度，例如16
                drop=drop_rate,  # Dropout率，例如0
                attn_drop=attn_drop_rate,  # 注意力Dropout率，例如0
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # 每一个模块传一个概率值
                norm_layer=norm_layer,  # 归一化层，例如nn.LayerNorm
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,  # 下采样层
                use_checkpoint=use_checkpoint,  # 是否使用检查点
            )
            self.layers_x_t.append(layer_x_t)

        """ 
        解码器部分
        """
        # 计算合并后的特征维度
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 以第一个为例，num_layers=2
            # 输入: (B, H/(4*2^(num_layers-1-i)), W/(4*2^(num_layers-1-i)), dims_decoder[i_layer])
            # 输出: (B, H/(4*2^(num_layers-1-i-1)), W/(4*2^(num_layers-1-i-1)), dims_decoder[i_layer])
            layer = VSSLayer_decoder(
                dim=dims_decoder[i_layer],  # 当前层维度，例如768
                depth=depths_decoder[i_layer],  # 当前层深度，例如2
                d_state=d_state,  # 状态维度，例如16
                drop=drop_rate,  # Dropout率，例如0
                attn_drop=attn_drop_rate,  # 注意力Dropout率，例如0
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,  # 归一化层，例如nn.LayerNorm
                upsample=PatchExpand2D if (i_layer != 0) else None,  # 上采样层
                use_checkpoint=use_checkpoint,  # 是否使用检查点
            )
            self.layers_up.append(layer)

        #  输入 64*64*96 ->linear+LN b*256*256*24                          96                             nn.LN
        # 输入: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[-1]) -> 输出: (B, H, W, dims_decoder[-1]//4)
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        #     维度变换 输出b*1*256*256         24                 1
        # 输入: (B, H, W, dims_decoder[-1]//4) -> 输出: (B, num_classes, H, W)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        self.apply(self._init_weights)

        # 动态创建浅层融合卷积层
        self.shallow_fusion1_32 = nn.Conv2d(64, 32, 1, 1, 0)
        self.shallow_fusion2_32 = nn.Conv2d(64, 32, 1, 1, 0)
        self.shallow_fusion1_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion2_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion1_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion2_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion1_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion2_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion1_512 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.shallow_fusion2_512 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.shallow_fusion1_1024 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.shallow_fusion2_1024 = nn.Conv2d(2048, 1024, 1, 1, 0)

    def _init_weights(self, m: nn.Module):
        """
        初始化模型权重的方法。
        注意：
        - out_proj.weight 在 VSSBlock 中已经初始化，但在 nn.Linear 中会被覆盖，因此 VSSBlock 中的初始化实际上是无用的。
        - 模型参数中没有找到 fc.weight 或 nn.Embedding，因此这些部分的初始化不需要在这里处理。
        - Conv2D 的权重没有在这里初始化，因此需要在其他地方进行初始化。

        具体初始化规则：
        - 对于 nn.Linear 层：
            - 使用截断正态分布初始化权重，标准差为 0.02。
            - 如果存在偏置，则将其初始化为 0。
        - 对于 nn.LayerNorm 层：
            - 将偏置初始化为 0。
            - 将权重初始化为 1.0。
        """
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化权重，标准差为 0.02
            trunc_normal_(m.weight, std=.02)
            # 如果存在偏置，则将其初始化为 0
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 将 LayerNorm 的偏置初始化为 0
            nn.init.constant_(m.bias, 0)
            # 将 LayerNorm 的权重初始化为 1.0
            nn.init.constant_(m.weight, 1.0)


    """ 
    在 forward_features 和 forward_features_up 方法中实现了跳跃连接，
    通过保存中间特征图并在解码器中逐层拼接，从而保留更多的细节信息。
    """

    def forward_features(self, x, context):
        skip_list = []
        # 输入: (B, C, H, W) -> 输出: (B, H/4, W/4, embed_dim)
        x = self.patch_embed(x)
        # 输入: (B, H/4, W/4, embed_dim) -> 输出: (B, H/4, W/4, embed_dim)
        x = self.pos_drop(x)
        # print('context的形状是：', context.shape)
        context = self.patch_embed(context)
        context = self.pos_drop(context)


        for i, (layer_context, layer_x_t) in enumerate(zip(self.layers_context, self.layers_x_t)):
            # 输入: (B, H/(4*2^i), W/(4*2^i), dims[i_layer]) -> 输出: (B, H/(4*2^(i+1)), W/(4*2^(i+1)), dims[i_layer+1])
            # print('i的值是',i)
            if i == 0:
                # print('context的形状是：', context.shape)
                context, x = context.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2)
                combined = self.shallow_fusion1_64(torch.concat([context, x], dim=1))
                combined = self.shallow_fusion2_64(torch.concat([combined, x], dim=1)) + combined
                combined = combined.permute(0, 2, 3, 1)
                context, x = context.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)
                skip_list.append(combined)

            context = layer_context(context)
            x = layer_x_t(x)
            # print('条件和噪声的形状是：', context.shape, x.shape)
            base_filter = x.shape[-1]

            # print('此时的通道数是',base_filter)

            context, x = context.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2)

            # print('条件的形状是', context.shape)

            # if base_filter == 128:
            #     combined = self.shallow_fusion1_128(torch.concat([context, x], dim=1))
            #     combined = self.shallow_fusion2_128(torch.concat([combined, x], dim=1)) + combined
            # elif base_filter == 256:
            #     combined = self.shallow_fusion1_256(torch.concat([context, x], dim=1))
            #     combined = self.shallow_fusion2_256(torch.concat([combined, x], dim=1)) + combined

            # if base_filter == 64:
            #     combined = self.shallow_fusion1_64(torch.concat([context, x], dim=1))
            #     combined = self.shallow_fusion2_64(torch.concat([combined, x], dim=1)) + combined

            if base_filter == 128:
                combined = self.shallow_fusion1_128(torch.concat([context, x], dim=1))
                combined = self.shallow_fusion2_128(torch.concat([combined, x], dim=1)) + combined
            elif base_filter == 256:
                combined = self.shallow_fusion1_256(torch.concat([context, x], dim=1))
                combined = self.shallow_fusion2_256(torch.concat([combined, x], dim=1)) + combined

            # print('合成的形状是', combined.shape)
            combined = combined.permute(0, 2, 3, 1)
            # print('combined的形状是：',combined.shape)
            context, x = context.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)
            # print('条件和噪声的形状是2：', context.shape, x.shape)
            # print('combined的形状是：', combined.shape)
            if i != 2:
                skip_list.append(combined)
                # skip_list.append(context)
            # 返回最终特征和跳跃连接特征列表
        return combined, skip_list
        # return context, skip_list


    def forward_features_up(self, x, skip_list):
        layer_outputs = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # 输入: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[0]) -> 输出: (B, H/(4*2^(num_layers-2)), W/(4*2^(num_layers-2)), dims_decoder[0])
                x = layer_up(x)
            else:
                # 输入: (B, H/(4*2^(num_layers-1-i)), W/(4*2^(num_layers-1-i)), dims_decoder[i_layer]) -> 输出: (B, H/(4*2^(num_layers-1-i-1)), W/(4*2^(num_layers-1-i-1)), dims_decoder[i_layer])
                # if isinstance(skip_list, list):
                #     for i, item in enumerate(skip_list):
                #         print(f"Element {i} shape: {item.shape}")
                # else:
                #     print(f"skip_list is not a list, type: {type(skip_list)}")
                x = layer_up(x + skip_list[-inx])
            layer_outputs.append(x)  # 将当前层的输出添加到列表中
        # 返回最终上采样特征
        return layer_outputs

    """
    最终输出层由 Final_PatchExpand2D 和 final_conv 组成，用于将特征图还原为原始图像大小，并生成最终的预测结果。
    """

    def forward_final(self, x):
        # 输入: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[-1]) -> 输出: (B, H, W, dims_decoder[-1]//4)
        x = self.final_up(x)
        # 输入: (B, H, W, dims_decoder[-1]//4) -> 输出: (B, dims_decoder[-1]//4, H, W)
        x = x.permute(0, 3, 1, 2)
        # 输入: (B, dims_decoder[-1]//4, H, W) -> 输出: (B, num_classes, H, W)
        x = self.final_conv(x)
        # 返回最终预测结果
        return x

    # def forward(self, x, context):
    #     # 输入: (B, C, H, W) -> 输出: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[-1]), 跳跃连接特征列表
    #     x, skip_list = self.forward_features(x, context)
    #     # 输入: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[-1]), 跳跃连接特征列表 -> 输出: (B, H, W, dims_decoder[-1]//4)
    #     x, layer_outputs = self.forward_features_up(x, skip_list)
    #     # 输入: (B, H, W, dims_decoder[-1]//4) -> 输出: (B, num_classes, H, W)
    #     x = self.forward_final(x)
    #     x = x.permute(0, 2, 3, 1)
    #     layer_outputs.append(x)
    #     x = x.permute(0, 3, 1, 2)
    #     # 打印 layer_outputs 中每个元素的形状
    #     for i, output in enumerate(layer_outputs):
    #         print(f"layer_outputs[{i}] shape: {output.shape}")
    #     # 返回最终预测结果
    #     return layer_outputs

    def forward(self, x, context):
        # 输入: (B, C, H, W) -> 输出: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[-1]), 跳跃连接特征列表
        x, skip_list = self.forward_features(x, context)
        # 输入: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[-1]), 跳跃连接特征列表 -> 输出: (B, H, W, dims_decoder[-1]//4)
        layer_outputs = self.forward_features_up(x, skip_list)
        # 返回最终预测结果
        return layer_outputs

class VSSM_fusion(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=2, depths=[2, 2, 2], depths_decoder=[2, 2, 2],
                 dims=[64, 128, 256], dims_decoder=[128, 64, 32], d_state=16, drop_rate=0.1,
                 attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes  # 输出通道数！！！！
        self.num_layers = len(depths)  # 编码器层数，这里是4
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]  # 嵌入维度，这里是96
        self.num_features = dims[-1]  # 最终特征维度，这里是768
        self.dims = dims  # 各层维度，[96, 192, 384, 768]

        # 4*4卷积+LN-> b*w*h*c
        # 输入: (B, C, H, W) -> 输出: (B, H/4, W/4, embed_dim)
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # Dropout层
        # 输入: (B, H/4, W/4, embed_dim) -> 输出: (B, H/4, W/4, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 生成对应的sum(depths)随机深度衰减数值 dpr是正序，dpr_decoder是倒序（用到了[start:end:-1] 反向步长）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]


        """ 
        解码器部分
        """
        # 计算合并后的特征维度
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 以第一个为例，num_layers=2
            # 输入: (B, H/(4*2^(num_layers-1-i)), W/(4*2^(num_layers-1-i)), dims_decoder[i_layer])
            # 输出: (B, H/(4*2^(num_layers-1-i-1)), W/(4*2^(num_layers-1-i-1)), dims_decoder[i_layer])
            layer = VSSLayer_decoder(
                dim=dims_decoder[i_layer],  # 当前层维度，例如768
                depth=depths_decoder[i_layer],  # 当前层深度，例如2
                d_state=d_state,  # 状态维度，例如16
                drop=drop_rate,  # Dropout率，例如0
                attn_drop=attn_drop_rate,  # 注意力Dropout率，例如0
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,  # 归一化层，例如nn.LayerNorm
                upsample=PatchExpand2D,  # 上采样层
                use_checkpoint=use_checkpoint,  # 是否使用检查点
            )
            self.layers_up.append(layer)

        #  输入 64*64*96 ->linear+LN b*256*256*24                          96                             nn.LN
        # 输入: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[-1]) -> 输出: (B, H, W, dims_decoder[-1]//4)
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=2, norm_layer=norm_layer)
        #     维度变换 输出b*1*256*256         24                 1
        # 输入: (B, H, W, dims_decoder[-1]//4) -> 输出: (B, num_classes, H, W)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 2, num_classes, 1)
        self.apply(self._init_weights)

        # 动态创建浅层融合卷积层
        self.shallow_fusion1_32 = nn.Conv2d(64, 32, 1, 1, 0)
        self.shallow_fusion2_32 = nn.Conv2d(64, 32, 1, 1, 0)
        self.shallow_fusion1_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion2_64 = nn.Conv2d(128, 64, 1, 1, 0)
        self.shallow_fusion1_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion2_128 = nn.Conv2d(256, 128, 1, 1, 0)
        self.shallow_fusion1_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion2_256 = nn.Conv2d(512, 256, 1, 1, 0)
        self.shallow_fusion1_512 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.shallow_fusion2_512 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.shallow_fusion1_1024 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.shallow_fusion2_1024 = nn.Conv2d(2048, 1024, 1, 1, 0)

    def _init_weights(self, m: nn.Module):
        """
        初始化模型权重的方法。
        注意：
        - out_proj.weight 在 VSSBlock 中已经初始化，但在 nn.Linear 中会被覆盖，因此 VSSBlock 中的初始化实际上是无用的。
        - 模型参数中没有找到 fc.weight 或 nn.Embedding，因此这些部分的初始化不需要在这里处理。
        - Conv2D 的权重没有在这里初始化，因此需要在其他地方进行初始化。

        具体初始化规则：
        - 对于 nn.Linear 层：
            - 使用截断正态分布初始化权重，标准差为 0.02。
            - 如果存在偏置，则将其初始化为 0。
        - 对于 nn.LayerNorm 层：
            - 将偏置初始化为 0。
            - 将权重初始化为 1.0。
        """
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化权重，标准差为 0.02
            trunc_normal_(m.weight, std=.02)
            # 如果存在偏置，则将其初始化为 0
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 将 LayerNorm 的偏置初始化为 0
            nn.init.constant_(m.bias, 0)
            # 将 LayerNorm 的权重初始化为 1.0
            nn.init.constant_(m.weight, 1.0)


    def forward_features_up(self, layer_outputs_1, layer_outputs_2):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                combined = self.shallow_fusion1_256(torch.concat([layer_outputs_1[0].permute(0, 3, 1, 2), layer_outputs_2[0].permute(0, 3, 1, 2)], dim=1))
                combined = self.shallow_fusion2_256(torch.concat([layer_outputs_1[0].permute(0, 3, 1, 2), layer_outputs_2[0].permute(0, 3, 1, 2)], dim=1)) + combined
                combined = combined.permute(0, 2, 3, 1)
                x = layer_up(combined)
            elif inx == 1:
                combined = self.shallow_fusion1_128(torch.concat([layer_outputs_1[1].permute(0, 3, 1, 2), layer_outputs_2[1].permute(0, 3, 1, 2)], dim=1))
                combined = self.shallow_fusion2_128(torch.concat([layer_outputs_1[1].permute(0, 3, 1, 2), layer_outputs_2[1].permute(0, 3, 1, 2)], dim=1)) + combined
                combined = combined.permute(0, 2, 3, 1)
                x = layer_up(x + combined)
            elif inx == 2:
                combined = self.shallow_fusion1_64(torch.concat([layer_outputs_1[2].permute(0, 3, 1, 2), layer_outputs_2[2].permute(0, 3, 1, 2)], dim=1))
                combined = self.shallow_fusion2_64(torch.concat([layer_outputs_1[2].permute(0, 3, 1, 2), layer_outputs_2[2].permute(0, 3, 1, 2)], dim=1)) + combined
                combined = combined.permute(0, 2, 3, 1)
                x = layer_up(x + combined)
        # 返回最终上采样特征
        return x

    """
    最终输出层由 Final_PatchExpand2D 和 final_conv 组成，用于将特征图还原为原始图像大小，并生成最终的预测结果。
    """

    def forward_final(self, x):
        # 输入: (B, H/(4*2^(num_layers-1)), W/(4*2^(num_layers-1)), dims_decoder[-1]) -> 输出: (B, H, W, dims_decoder[-1]//4)
        x = self.final_up(x)
        # 输入: (B, H, W, dims_decoder[-1]//4) -> 输出: (B, dims_decoder[-1]//4, H, W)
        x = x.permute(0, 3, 1, 2)
        # 输入: (B, dims_decoder[-1]//4, H, W) -> 输出: (B, num_classes, H, W)
        x = self.final_conv(x)
        # 返回最终预测结果
        return x


    def forward(self, layer_outputs_1, layer_outputs_2):

        layer_outputs = self.forward_features_up(layer_outputs_1, layer_outputs_2)

        x = self.forward_final(layer_outputs)

        return x


class ATMamba(nn.Module):
    def __init__(self,
                 input_channels= 3,
                 depths=[2, 2, 2],
                 depths_decoder=[2, 2, 2],
                 fusion_decoder=[128, 64, 32],
                 attn_drop_rate=0.2,
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 num_classes=2,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        # 这里就是输出的通道数
        self.num_classes = num_classes

        self.vmunet_m1 = VSSM(in_chans=input_channels,  # 3
                           num_classes=self.num_classes,  # input_channles
                           depths=depths,  # [2,2,9,2]
                           depths_decoder=depths_decoder,  # [2,9,2,2]
                           drop_path_rate=drop_path_rate,  # 0.2
                           attn_drop_rate=attn_drop_rate,
                           )
        self.vmunet_m2 = VSSM(in_chans=input_channels,  # 3
                           num_classes=self.num_classes,  # input_channles
                           depths=depths,  # [2,2,9,2]
                           depths_decoder=depths_decoder,  # [2,9,2,2]
                           drop_path_rate=drop_path_rate,  # 0.2
                           attn_drop_rate=attn_drop_rate,
                           )

        self.fusion = VSSM_fusion(num_classes=num_classes,
                                  depths_decoder=[2, 2, 2],
                                  dims_decoder=fusion_decoder
                                  )
    # def forward(self, x_xnoisy):
    #     # print(timesteps.shape)
    #     context, x = x_xnoisy
    #
    #     logits = self.vmunet(x, context)
    #
    #     if self.num_classes == 1:
    #         return torch.sigmoid(logits)
    #     else:
    #         return logits

    def forward(self, modal1 , modal2):
        # print(timesteps.shape)
        modal1_T1, modal1_T2 = modal1
        modal2_T1, modal2_T2 = modal2

        layer_outputs_1 = self.vmunet_m1(modal1_T1, modal1_T2)
        layer_outputs_2 = self.vmunet_m2(modal2_T1, modal2_T2)

        logits = self.fusion(layer_outputs_1, layer_outputs_2)

        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits


if __name__ == "__main__":
    # 示例使用
    context = torch.randn(128, 3, 16, 16).to("cuda:1")
    x = torch.randn(128, 3, 16, 16).to("cuda:1")
    context2 = torch.randn(128, 224, 16, 16).to("cuda:1")
    x2 = torch.randn(128, 224, 16, 16).to("cuda:1")
    x_xnoisy = [context, x]
    x_xnoisy2 = [context2, x2]
    net = ATMamba().to("cuda:1")
    print(net(x_xnoisy, x_xnoisy2).shape)
