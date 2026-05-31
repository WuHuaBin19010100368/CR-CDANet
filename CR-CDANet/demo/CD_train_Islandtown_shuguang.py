
import torch.nn as nn

import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.data_loader import CD_DOMAIN_Dataset, CD_DOMAIN
from change_detection.utils_func.metrics import Evaluator
from change_detection.change_model.ATM_ceshi_qianmian import ATMamba
from transfer_learning.featureloss_2048 import FeatureLossCalculator
import change_detection.utils_func.lovasz_loss as L

RESULT_ROOT = os.environ.get("CRCDANET_RESULT_ROOT", "./z_result")

class classifier_target(nn.Module):
    def __init__(self):
        super(classifier_target, self).__init__()
        self.flatten = nn.Flatten(-2)
        self.fc1 = nn.Linear(16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 4)
        self.fc8 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        x = x.squeeze(-1)
        return x

class classifier_source(nn.Module):
    def __init__(self):
        super(classifier_source, self).__init__()
        self.flatten = nn.Flatten(-2)
        self.fc1 = nn.Linear(16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 4)
        self.fc8 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        x = x.squeeze(-1)
        return x

class Trainer(object):
    def __init__(self, args, loop):
        """
        初始化函数，设置模型训练和评估的参数和配置。

        参数:
        - args: 命令行参数或其他配置参数，包含模型训练和评估的各种设置。
        """
        self.args = args

        # 根据args.device_id设置设备
        self.device = torch.device(f'cuda:{args.device_id}' if args.cuda and torch.cuda.is_available() else 'cpu')

        source_data = CD_DOMAIN_Dataset(dataset_name='Islandtown_domain', mode='train', patch_size=17, train_val_ratio=0.01)
        self.source_data_loader = DataLoader(source_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        train_data = CD_DOMAIN(dataset_name='shuguang_domain', mode='train', patch_size=17, train_val_ratio=0.01, loop=loop)
        self.train_data_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        # 初始化评估器，用于评估模型性能
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = ATMamba(
            input_channels=3,
            depths=[2, 2, 2],
            depths_decoder=[2, 2, 2],
            fusion_decoder=[128, 64, 32],
            attn_drop_rate=0.2,
            drop_path_rate=0.2,
            load_ckpt_path=None,
            num_classes=2,
        )

        self.deep_model_source = ATMamba(
            input_channels=3,
            depths=[2, 2, 2],
            depths_decoder=[2, 2, 2],
            fusion_decoder=[128, 64, 32],
            attn_drop_rate=0.2,
            drop_path_rate=0.2,
            load_ckpt_path=None,
            num_classes=2,
        )

        self.deep_model = self.deep_model.to(self.device)
        self.deep_model_source = self.deep_model_source.to(self.device)

        self.classifier_target = classifier_target().to(self.device)
        self.classifier_source = classifier_source().to(self.device)

        # 设置模型保存路径
        self.model_save_path = f'{RESULT_ROOT}/Islandtown_shuguang/model/CD_model/CD_{loop}_0.01'

        # 创建模型保存路径的目录，如果不存在的话
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # 设置学习率和最大迭代次数
        # 设置学习率
        self.lr = args.learning_rate

        # 定义模型加载路径的映射规则
        if loop == 0:
            base_path = f"{RESULT_ROOT}/source/Islandtown/model/CD_model"
            model_name = "100_model.pth"
            classifier_name = "100_classifier_A1.pth"
        else:
            base_path = f"{RESULT_ROOT}/Islandtown_shuguang/model/CD_model/CD_{loop - 1}_0.01"
            model_name = "100_model_target.pth"
            classifier_name = "100_classifier_target.pth"
            model_source_name = "100_model_source.pth"
            classifier_source_name = "100_classifier_source.pth"

        # 加载 deep_model 和 classifier_target
        self.deep_model.load_state_dict(torch.load(
            os.path.join(base_path, model_name),
            map_location=self.device
        ))
        self.classifier_target.load_state_dict(torch.load(
            os.path.join(base_path, classifier_name),
            map_location=self.device
        ))

        # 加载 deep_model_source 和 classifier_source
        if loop == 0:
            # loop=0 时源和目标共享同一组预训练模型
            self.deep_model_source.load_state_dict(torch.load(
                os.path.join(base_path, model_name),
                map_location=self.device
            ))
            self.classifier_source.load_state_dict(torch.load(
                os.path.join(base_path, classifier_name),
                map_location=self.device
            ))
        else:
            self.deep_model_source.load_state_dict(torch.load(
                os.path.join(base_path, model_source_name),
                map_location=self.device
            ))
            self.classifier_source.load_state_dict(torch.load(
                os.path.join(base_path, classifier_source_name),
                map_location=self.device
            ))

        # 初始化优化器
        self.optim_1 = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        self.optim_2 = optim.AdamW(self.classifier_target.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        self.optim_3 = optim.AdamW(self.deep_model_source.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        self.optim_4 = optim.AdamW(self.classifier_source.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)


        self.scheduler_1 = optim.lr_scheduler.StepLR(self.optim_1, step_size=50, gamma=0.1)
        self.scheduler_2 = optim.lr_scheduler.StepLR(self.optim_2, step_size=50, gamma=0.1)
        self.scheduler_3 = optim.lr_scheduler.StepLR(self.optim_3, step_size=50, gamma=0.1)
        self.scheduler_4 = optim.lr_scheduler.StepLR(self.optim_4, step_size=50, gamma=0.1)

        self.best_average_loss_target = float('inf')
        self.best_average_loss_source = float('inf')
        self.best_round_target = 0
        self.best_round_source = 0
        self.feature_loss_calculator = FeatureLossCalculator()

    def training(self, epoch):
        source_iter = iter(self.source_data_loader)  # 创建 source_loader 的迭代器

        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)       # 获取训练数据加载器中的批次数量。
        train_enumerator = enumerate(self.train_data_loader)
        # 将 self.train_data_loader 中的每个元素（即每个批次的数据）与其对应的索引（批次号）结合起来。
        total_loss_target = 0
        total_loss_source = 0
        total_mmd_coral_loss_m1_t1 = 0
        total_total_mmd_loss_m1_t1 = 0
        total_total_coral_loss_m1_t1 = 0
        total_mmd_coral_loss_m1_t2 = 0
        total_total_mmd_loss_m1_t2 = 0
        total_total_coral_loss_m1_t2 = 0
        total_mmd_coral_loss_m2_t1 = 0
        total_total_mmd_loss_m2_t1 = 0
        total_total_coral_loss_m2_t1 = 0
        total_mmd_coral_loss_m2_t2 = 0
        total_total_mmd_loss_m2_t2 = 0
        total_total_coral_loss_m2_t2 = 0
        total_mmd_coral_loss = 0

        # 动态调整损失权重
        if epoch < 30:
            mmd_weight = 0.75  # 逐渐减少mmd_coral_loss的权重
            main_weight = 0.25  # 逐渐增加变化检测损失的权重
        elif 30 <= epoch < 50:
            mmd_weight = 0.5  # 逐渐减少mmd_coral_loss的权重
            main_weight = 0.5  # 逐渐增加变化检测损失的权重
        elif 50 <= epoch < 1000:
            mmd_weight = 0.25  # 逐渐减少mmd_coral_loss的权重
            main_weight = 0.75  # 逐渐增加变化检测损失的权重

        for _ in tqdm(range(elem_num)):             # 使用tqdm库显示进度条，遍历训练数据加载器中的所有批次。
            itera, data = train_enumerator.__next__()               # 获取当前批次的数据，包括变化前图像、变化后图像、标签等。
            # pre_change_imgs, post_change_imgs, labels, pos  = data
            pre_rgb_imgs, post_rgb_imgs, pre_sar_imgs, post_sar_imgs, labels, pos = data
            pre_rgb_imgs = pre_rgb_imgs.to(self.device).float()
            post_rgb_imgs = post_rgb_imgs.to(self.device).float()
            pre_sar_imgs = pre_sar_imgs.to(self.device).float()
            post_sar_imgs = post_sar_imgs.to(self.device).float()
            labels_target = labels.to(self.device).long()

            try:
                pre_rgb_imgs_source, post_rgb_imgs_source, pre_sar_imgs_source, post_sar_imgs_source, labels_source, pos_source = next(source_iter)
            except StopIteration:
                # 如果 source_loader 迭代完毕，重新创建迭代器并获取下一个批次的数据
                source_iter = iter(self.source_data_loader)  # 重新迭代 source_loader
                pre_rgb_imgs_source, post_rgb_imgs_source, pre_sar_imgs_source, post_sar_imgs_source, labels_source, pos_source = next(source_iter)

            pre_rgb_imgs_source = pre_rgb_imgs_source.to(self.device).float()
            post_rgb_imgs_source = post_rgb_imgs_source.to(self.device).float()
            pre_sar_imgs_source = pre_sar_imgs_source.to(self.device).float()
            post_sar_imgs_source = post_sar_imgs_source.to(self.device).float()
            labels_source = labels_source.to(self.device).long()

            self.optim_1.zero_grad() # 清零梯度
            self.optim_2.zero_grad()
            # zheliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
            self.optim_3.zero_grad()
            self.optim_4.zero_grad()

            output_1_target, encoder_outputs_t_m1_t1, encoder_outputs_t_m1_t2, encoder_outputs_t_m2_t1, encoder_outputs_t_m2_t2 \
                = self.deep_model([pre_rgb_imgs, post_rgb_imgs],[pre_sar_imgs, post_sar_imgs])
            if epoch % 2 == 0 :
                output_2_target = self.classifier_target(output_1_target)
            else :
                output_2_target = self.classifier_source(output_1_target)
            encoder_outputs_t_m1_t1 = [x.permute(0, 2, 3, 1) for x in encoder_outputs_t_m1_t1]
            encoder_outputs_t_m1_t2 = [x.permute(0, 2, 3, 1) for x in encoder_outputs_t_m1_t2]
            encoder_outputs_t_m2_t1 = [x.permute(0, 2, 3, 1) for x in encoder_outputs_t_m2_t1]
            encoder_outputs_t_m2_t2 = [x.permute(0, 2, 3, 1) for x in encoder_outputs_t_m2_t2]

            output_1_source, encoder_outputs_s_m1_t1, encoder_outputs_s_m1_t2, encoder_outputs_s_m2_t1, encoder_outputs_s_m2_t2 \
                = self.deep_model_source([pre_rgb_imgs_source, post_rgb_imgs_source],[pre_sar_imgs_source, post_sar_imgs_source])
            if epoch % 2 == 0:
                output_2_source = self.classifier_source(output_1_source)
            else:
                output_2_source = self.classifier_target(output_1_source)
            encoder_outputs_s_m1_t1 = [x.permute(0, 2, 3, 1) for x in encoder_outputs_s_m1_t1]
            encoder_outputs_s_m1_t2 = [x.permute(0, 2, 3, 1) for x in encoder_outputs_s_m1_t2]
            encoder_outputs_s_m2_t1 = [x.permute(0, 2, 3, 1) for x in encoder_outputs_s_m2_t1]
            encoder_outputs_s_m2_t2 = [x.permute(0, 2, 3, 1) for x in encoder_outputs_s_m2_t2]

            # 计算目标域损失
            ce_loss_1_target = F.cross_entropy(output_2_target, labels_target)
            total_loss_target += ce_loss_1_target.item()
            # 计算Lovasz损失
            main_loss_target = ce_loss_1_target
            # main_loss = ce_loss_1

            # 计算源域损失
            ce_loss_1_source = F.cross_entropy(output_2_source, labels_source)
            total_loss_source += ce_loss_1_source.item()
            # 计算Lovasz损失
            main_loss_source = ce_loss_1_source

            # 计算MMD和CORAL损失
            total_coral_loss_m1_t1, total_mmd_loss_m1_t1, mmd_coral_loss_m1_t1 = self.compute_mmd_coral_loss(encoder_outputs_t_m1_t1, encoder_outputs_s_m1_t2)
            total_coral_loss_m1_t2, total_mmd_loss_m1_t2, mmd_coral_loss_m1_t2 = self.compute_mmd_coral_loss(encoder_outputs_t_m1_t2, encoder_outputs_s_m1_t2)
            total_coral_loss_m2_t1, total_mmd_loss_m2_t1, mmd_coral_loss_m2_t1 = self.compute_mmd_coral_loss(encoder_outputs_t_m2_t1, encoder_outputs_s_m2_t2)
            total_coral_loss_m2_t2, total_mmd_loss_m2_t2, mmd_coral_loss_m2_t2 = self.compute_mmd_coral_loss(encoder_outputs_t_m2_t2, encoder_outputs_s_m2_t2)
            mmd_coral_loss = mmd_coral_loss_m1_t1 + mmd_coral_loss_m1_t2 + mmd_coral_loss_m2_t1 + mmd_coral_loss_m2_t2
            # 应用权重
            final_loss = main_weight * (main_loss_target + main_loss_source) + mmd_weight * mmd_coral_loss
            # 最终损失
            final_loss.backward()       # 对最终损失进行反向传播，计算梯度

            mmd_coral_loss_m1_t1 = torch.tensor(mmd_coral_loss_m1_t1)
            total_mmd_coral_loss_m1_t1 += mmd_coral_loss_m1_t1.item()
            total_mmd_loss_m1_t1 = torch.tensor(total_mmd_loss_m1_t1)
            total_total_mmd_loss_m1_t1 += total_mmd_loss_m1_t1.item()
            total_coral_loss_m1_t1 = torch.tensor(total_coral_loss_m1_t1)
            total_total_coral_loss_m1_t1 += total_coral_loss_m1_t1.item()

            mmd_coral_loss_m1_t2 = torch.tensor(mmd_coral_loss_m1_t2)
            total_mmd_coral_loss_m1_t2 += mmd_coral_loss_m1_t2.item()
            total_mmd_loss_m1_t2 = torch.tensor(total_mmd_loss_m1_t2)
            total_total_mmd_loss_m1_t2 += total_mmd_loss_m1_t2.item()
            total_coral_loss_m1_t2 = torch.tensor(total_coral_loss_m1_t2)
            total_total_coral_loss_m1_t2 += total_coral_loss_m1_t2.item()

            mmd_coral_loss_m2_t1 = torch.tensor(mmd_coral_loss_m2_t1)
            total_mmd_coral_loss_m2_t1 += mmd_coral_loss_m2_t1.item()
            total_mmd_loss_m2_t1 = torch.tensor(total_mmd_loss_m2_t1)
            total_total_mmd_loss_m2_t1 += total_mmd_loss_m2_t1.item()
            total_coral_loss_m2_t1 = torch.tensor(total_coral_loss_m2_t1)
            total_total_coral_loss_m2_t1 += total_coral_loss_m2_t1.item()

            mmd_coral_loss_m2_t2 = torch.tensor(mmd_coral_loss_m2_t2)
            total_mmd_coral_loss_m2_t2 += mmd_coral_loss_m2_t2.item()
            total_mmd_loss_m2_t2 = torch.tensor(total_mmd_loss_m2_t2)
            total_total_mmd_loss_m2_t2 += total_mmd_loss_m2_t2.item()
            total_coral_loss_m2_t2 = torch.tensor(total_coral_loss_m2_t2)
            total_total_coral_loss_m2_t2 += total_coral_loss_m2_t2.item()

            mmd_coral_loss = torch.tensor(mmd_coral_loss)
            total_mmd_coral_loss += mmd_coral_loss.item()

            self.optim_1.step()  
            self.optim_2.step()
            self.optim_3.step()
            self.optim_4.step()

        average_loss_target = total_loss_target / elem_num
        average_loss_source = total_loss_source / elem_num
        avg_mmd_coral_loss_m1_t1 = total_mmd_coral_loss_m1_t1 / elem_num
        avg_mmd_coral_loss_m1_t2 = total_mmd_coral_loss_m1_t2 / elem_num
        avg_mmd_coral_loss_m2_t1 = total_mmd_coral_loss_m2_t1 / elem_num
        avg_mmd_coral_loss_m2_t2 = total_mmd_coral_loss_m2_t2 / elem_num
        avg_total_mmd_loss_m1_t1 = total_total_mmd_loss_m1_t1 / elem_num
        avg_total_mmd_loss_m1_t2 = total_total_mmd_loss_m1_t2 / elem_num
        avg_total_mmd_loss_m2_t1 = total_total_mmd_loss_m2_t1 / elem_num
        avg_total_mmd_loss_m2_t2 = total_total_mmd_loss_m2_t2 / elem_num
        avg_total_coral_loss_m1_t1 = total_total_coral_loss_m1_t1 / elem_num
        avg_total_coral_loss_m1_t2 = total_total_coral_loss_m1_t2 / elem_num
        avg_total_coral_loss_m2_t1 = total_total_coral_loss_m2_t1 / elem_num
        avg_total_coral_loss_m2_t2 = total_total_coral_loss_m2_t2 / elem_num
        avg_total_mmd_coral_loss = total_mmd_coral_loss / elem_num

        print(f'epoch: {epoch + 1}, 目标域变化检测损失：{average_loss_target},',
              f' 源域变化检测损失：{average_loss_source},')
        print(f' m1_t1迁移学习部分中的总损失是：{avg_mmd_coral_loss_m1_t1},',
              f' m1_t1迁移学习部分中的MMD损失是：{avg_total_mmd_loss_m1_t1},',
              f' m1_t1迁移学习部分中的CORAL损失是：{avg_total_coral_loss_m1_t1},',)
        print(f' m1_t2迁移学习部分中的总损失是：{avg_mmd_coral_loss_m1_t2},',
              f' m1_t2迁移学习部分中的MMD损失是：{avg_total_mmd_loss_m1_t2},',
              f' m1_t2迁移学习部分中的CORAL损失是：{avg_total_coral_loss_m1_t2},',)
        print(f' m2_t1迁移学习部分中的总损失是：{avg_mmd_coral_loss_m2_t1},',
              f' m2_t1迁移学习部分中的MMD损失是：{avg_total_mmd_loss_m2_t1},',
              f' m2_t1迁移学习部分中的CORAL损失是：{avg_total_coral_loss_m2_t1},',)
        print(f' m2_t2迁移学习部分中的总损失是：{avg_mmd_coral_loss_m2_t2},',
              f' m2_t2迁移学习部分中的MMD损失是：{avg_total_mmd_loss_m2_t2},',
              f' m2_t2迁移学习部分中的CORAL损失是：{avg_total_coral_loss_m2_t2},',)
        print(f' 迁移学习的总损失是：{avg_total_mmd_coral_loss},')

        if average_loss_target < self.best_average_loss_target:
            self.best_average_loss_target = average_loss_target
            self.best_round_target = epoch + 1

        if average_loss_source < self.best_average_loss_source:
            self.best_average_loss_source = average_loss_source
            self.best_round_source = epoch + 1

        if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次模型
            torch.save(self.deep_model.state_dict(),
                       os.path.join(self.model_save_path, f'{epoch + 1}_model_target.pth'))
            torch.save(self.classifier_target.state_dict(),
                       os.path.join(self.model_save_path, f'{epoch + 1}_classifier_target.pth'))

            torch.save(self.deep_model_source.state_dict(),
                       os.path.join(self.model_save_path, f'{epoch + 1}_model_source.pth'))
            torch.save(self.classifier_source.state_dict(),
                       os.path.join(self.model_save_path, f'{epoch + 1}_classifier_source.pth'))

        print(f'变化检测阶段：目标域最高的轮次是 {self.best_round_target}, '
              f'目标域的损失是 {self.best_average_loss_target},'
              f'源域最高的轮次是 {self.best_round_source},'
              f'源域的损失是 {self.best_average_loss_source},')

        # 更新学习率调度器
        self.scheduler_1.step()
        self.scheduler_2.step()
        self.scheduler_3.step()
        self.scheduler_4.step()

    def compute_mmd_coral_loss(self, source_features, target_features):
        total_coral_loss, total_mmd_loss, total_loss = self.feature_loss_calculator.calculate_total_loss(source_features, target_features)
        return total_coral_loss, total_mmd_loss, total_loss

def main(loop):
    """
    主函数，用于初始化参数解析器并开始训练过程。
    """
    # 初始化参数解析器并设置描述
    parser = argparse.ArgumentParser(description="Training")

    # 添加是否使用CUDA参数
    parser.add_argument('--cuda', type=bool, default=True)

    # 添加CUDA设备ID参数
    parser.add_argument('--device_id', type=int, default=0)

    # 添加学习率参数
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    # 添加权重衰减参数
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # 解析参数
    args = parser.parse_args()

    # 创建训练器实例
    trainer = Trainer(args, loop)

    # 开始训练过程
    for i in range(100):
        trainer.training(i)

if __name__ == "__main__":
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    import argparse
    parser = argparse.ArgumentParser(description="Training CD Model")
    parser.add_argument('--loop', type=int, default=1, help='Loop index')
    args = parser.parse_args()
    main(args.loop)
