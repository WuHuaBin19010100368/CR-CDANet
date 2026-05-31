import torch
import torch.nn as nn
import torch.nn.functional as F
from transfer_learning.mmd import MMDLoss
from transfer_learning.coral import CoralLoss

class FeatureLossCalculator:
    def __init__(self):
        self.coral_loss = CoralLoss()
        self.mmd_loss = MMDLoss()

    def calculate_losses(self, features_source, features_target):
        total_coral_loss = 0.0
        total_mmd_loss = 0.0
        total_weight = 0.0

        for source, target in zip(features_source, features_target):
            # 展平特征图
            source_flat = source.view(source.size(0), -1)
            target_flat = target.view(target.size(0), -1)

            # 计算特征图的维度
            feature_dim = source_flat.size(1)

            # 根据特征图的维度设置权重
            if feature_dim == 1024:
                coral_weight = 0.7
                mmd_weight = 0.3
            elif feature_dim == 512:
                coral_weight = 0.5
                mmd_weight = 0.5
            elif feature_dim == 256:
                coral_weight = 0.3
                mmd_weight = 0.7
            # if feature_dim == 1024:
            #     coral_weight = 0.5
            #     mmd_weight = 0.5
            # elif feature_dim == 512:
            #     coral_weight = 0.5
            #     mmd_weight = 0.5
            # elif feature_dim == 256:
            #     coral_weight = 0.5
            #     mmd_weight = 0.5
            # if feature_dim == 1024:
            #     coral_weight = 1.0
            #     mmd_weight = 0.0
            #     # print('1')
            # elif feature_dim == 512:
            #     coral_weight = 1.0
            #     mmd_weight = 0.0
            #     # print('2')
            # elif feature_dim == 256:
            #     coral_weight = 1.0
            #     mmd_weight = 0.0
                # print('3')
            # if feature_dim == 2048:
            #     coral_weight = 0.7
            #     mmd_weight = 0.3
            # elif feature_dim == 1024:
            #     coral_weight = 0.5
            #     mmd_weight = 0.5                     
            # elif feature_dim == 512:
            #     coral_weight = 0.3
            #     mmd_weight = 0.7
            else:
                raise ValueError("Invalid feature dimension")
            
            # print('测试比例：',f'coral_weight: {coral_weight}, mmd_weight: {mmd_weight}')

            # 计算 CORAL 损失
            coral_loss_value = self.coral_loss(source_flat, target_flat)
            total_coral_loss += coral_loss_value.item() * coral_weight

            # 计算 MMD 损失
            mmd_loss_value = self.mmd_loss(source=source_flat, target=target_flat)
            total_mmd_loss += mmd_loss_value.item() * mmd_weight

            # 累加权重
            total_weight += mmd_weight + coral_weight

        # 归一化损失
        total_coral_loss /= total_weight
        total_mmd_loss /= total_weight
        # print(total_weight)
        return total_coral_loss, total_mmd_loss

    def calculate_total_loss(self, features_source, features_target):
        total_coral_loss, total_mmd_loss = self.calculate_losses(features_source, features_target)
        total_loss = total_coral_loss + total_mmd_loss
        return total_coral_loss, total_mmd_loss, total_loss

if __name__ == "__main__":
    # 假设你已经从之前的代码中提取了三层特征图
    # 这里我们用随机数据来模拟这些特征图
    features_source = [
        torch.rand(256, 4, 4, 64),  # 第一层特征图
        torch.rand(256, 2, 2, 128), # 第二层特征图
        torch.rand(256, 1, 1, 256)  # 第三层特征图
    ]

    features_target = [
        torch.rand(256, 4, 4, 64),  # 第一层特征图
        torch.rand(256, 2, 2, 128), # 第二层特征图
        torch.rand(256, 1, 1, 256)  # 第三层特征图
    ]

    # 创建 FeatureLossCalculator 实例
    loss_calculator = FeatureLossCalculator()

    # 计算损失
    total_coral_loss, total_mmd_loss, total_loss = loss_calculator.calculate_total_loss(features_source, features_target)

    # 打印每一层的损失
    print(f"Total CORAL Loss: {total_coral_loss}")
    print(f"Total MMD Loss: {total_mmd_loss}")
    print(f"Total Combined Loss: {total_loss}")
