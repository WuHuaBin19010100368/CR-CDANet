import torch
import torch.nn as nn

class CoralLoss(nn.Module):
    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, source, target):
        d = source.size(1)  # 特征维度

        # 计算源域和目标域特征的均值
        source_mean = source.mean(0)
        target_mean = target.mean(0)

        # 中心化特征
        source_centered = source - source_mean
        target_centered = target - target_mean

        # 计算相关性矩阵
        source_corr = torch.matmul(source_centered.t(), source_centered) / (source.size(0) - 1)
        target_corr = torch.matmul(target_centered.t(), target_centered) / (target.size(0) - 1)

        # 计算Frobenius范数差异
        loss = torch.sum(torch.pow(source_corr - target_corr, 2)) / (4 * d * d)

        return loss
