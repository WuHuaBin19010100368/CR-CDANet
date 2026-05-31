import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    - source: 源域数据（n * len(x))
    - target: 目标域数据（m * len(y))
    - kernel_mul: 高斯核的倍数
    - kernel_num: 取不同高斯核的数量
    - fix_sigma: 不同高斯核的sigma值
    Return:
    - loss: MMD loss
    '''

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        '''
        初始化MMDLoss类
        '''
        super(MMDLoss, self).__init__()  # 调用父类初始化方法
        self.kernel_num = kernel_num  # 设置高斯核的数量
        self.kernel_mul = kernel_mul  # 设置高斯核的倍数
        self.fix_sigma = None  # 设置固定sigma值
        self.kernel_type = kernel_type  # 设置核类型

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        '''
        计算高斯核矩阵
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 获取样本总数
        total = torch.cat([source, target], dim=0)  # 将源域和目标域数据拼接在一起
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))  # 扩展维度，方便后续计算
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))  # 扩展维度，方便后续计算
        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算所有样本之间的L2距离
        if fix_sigma:
            bandwidth = fix_sigma  # 如果设置了固定sigma值，则直接使用
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)  # 计算带宽
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值
        bandwidth /= kernel_mul ** (kernel_num // 2)  # 调整带宽
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]  # 生成多个带宽值
        # 高斯核的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]  # 计算高斯核值
        return sum(kernel_val)  # 返回高斯核矩阵

    def linear_mmd2(self, f_of_X, f_of_Y):
        '''
        计算线性MMD距离
        '''
        loss = 0.0  # 初始化损失
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)  # 计算两个分布的均值差
        loss = delta.dot(delta.T)  # 计算MMD距离
        return loss  # 返回损失

    def forward(self, source, target):
        '''
        前向传播，计算MMD损失
        '''
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)  # 如果核类型为线性，调用线性MMD计算
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])  # 获取批量大小
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            # 上面这步计算高斯核矩阵
            XX = torch.mean(kernels[:batch_size, :batch_size])  # 计算源域数据的核矩阵均值
            YY = torch.mean(kernels[batch_size:, batch_size:])  # 计算目标域数据的核矩阵均值
            XY = torch.mean(kernels[:batch_size, batch_size:])  # 计算源域和目标域数据的交叉核矩阵均值
            YX = torch.mean(kernels[batch_size:, :batch_size])  # 计算目标域和源域数据的交叉核矩阵均值
            loss = torch.mean(XX + YY - XY - YX)  # 计算MMD损失
            return loss  # 返回损失

