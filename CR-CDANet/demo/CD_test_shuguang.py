import os
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader.data_loader import CD_DOMAIN_test
from change_detection.change_model.ATM_new import ATMamba
from change_detection.utils_func.metrics2 import calculate_metrics

RESULT_ROOT = os.environ.get("CRCDANET_RESULT_ROOT", "./z_result")

def predict2img(predict, img_gt, pos, save_folder):
    """
    将预测结果转换为图像格式并保存。

    参数:
    predict: 预测结果数组。
    img_gt: 地面真实图像。
    pos: 预测位置数组。
    save_folder: 保存结果的文件夹路径。
    i: 当前处理的索引，用于保存文件名。
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # 创建保存结果的文件夹
        print("Folder created successfully!")
    else:
        print("Folder already exists.")

    predict_img = np.zeros_like(img_gt)  # 初始化预测图像数组

    for i in range(len(predict)):
        x = pos[i][0]  # 获取预测位置的x坐标
        y = pos[i][1]  # 获取预测位置的y坐标
        v = predict[i]  # 获取预测值

        predict_img[x][y] = v  # 将预测值赋给预测图像对应位置

    savemat(os.path.join(save_folder, f'predict_mat.mat'), {'pred': predict_img})  # 保存预测结果为MAT文件
    print("图像的mat文件保存成功!")

class classifier_A1(nn.Module):
    def __init__(self):
        super(classifier_A1, self).__init__()
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

class Tester(object):
    def __init__(self, args, loop):
        """
        初始化函数，设置模型测试的参数和配置。

        参数:
        - args: 命令行参数或其他配置参数，包含模型测试的各种设置。
        """
        self.args = args
        self.loop = loop
        # 根据args.device_id设置设备
        self.device = torch.device(f'cuda:{args.device_id}' if args.cuda and torch.cuda.is_available() else 'cpu')

        # 初始化深度学习模型
        self.deep_model = ATMamba(
            input_channels=3,
            depths=[2, 2, 2],
            depths_decoder=[2, 2, 2],
            drop_path_rate=0.2,
            attn_drop_rate=0.2,
            load_ckpt_path=None
        )

        # 将模型转移到指定设备上
        self.deep_model = self.deep_model.to(self.device)
        # 初始化分类器
        self.classifier_A1 = classifier_A1().to(self.device)

    def main(self):
        """
        测试模型性能。
        """
        print('开始测试！！！！！！！！！！')  # 打印测试开始信息

        # 使用self.device来设置设备
        device = self.device

        model_path = f"{RESULT_ROOT}/Islandtown_shuguang/model/CD_model/CD_{self.loop}_0.01/100_model_target.pth"
        classifier_path = f"{RESULT_ROOT}/Islandtown_shuguang/model/CD_model/CD_{self.loop}_0.01/100_classifier_target.pth"

        self.deep_model.load_state_dict(torch.load(model_path, map_location=device))
        self.classifier_A1.load_state_dict(torch.load(classifier_path, map_location=device))

        self.deep_model.eval()
        self.classifier_A1.eval()

        db = CD_DOMAIN_test(dataset_name='shuguang_domain', mode='test', patch_size=17, train_val_ratio=0.01,
                            loop=self.loop)
        test_data = DataLoader(db, batch_size=256, num_workers=0, drop_last=False)

        img_gt = db.img_gt

        preds_all = []  # 存储所有预测结果
        labels_all = []  # 存储所有标签
        pos_all = []

        batch_total_time = 0  # 初始化批处理总时间
        start_total_time = time.time()  # 记录测试开始时间
        torch.cuda.synchronize()  # 同步CUDA设备

        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (img_m1_t1, img_m1_t2, img_m2_t1, img_m2_t2, label, pos) in enumerate(test_data):  # 遍历数据加载器
                batch_start_time = time.time()  # 记录当前批处理开始时间

                img_m1_t1 = img_m1_t1.to(device=device, dtype=torch.float32)  # 将图像1移动到指定设备
                img_m1_t2 = img_m1_t2.to(device=device, dtype=torch.float32)  # 将图像2移动到指定设备
                img_m2_t1 = img_m2_t1.to(device=device, dtype=torch.float32)  # 将图像1移动到指定设备
                img_m2_t2 = img_m2_t2.to(device=device, dtype=torch.float32)  # 将图像2移动到指定设备
                label = label.to(device=device)  # 将标签移动到指定设备

                output_1 = self.deep_model([img_m1_t1, img_m1_t2], [img_m2_t1, img_m2_t2])  # 进行前向传播，获取预测结果
                output_2 = self.classifier_A1(output_1)

                batch_total_time += time.time() - batch_start_time  # 更新批处理总时间
                print('\r', batch_idx, end='')  # 打印当前批处理索引

                preds_all.append(output_2.detach().cpu().numpy())  # 将预测结果添加到列表中
                labels_all.append(label.detach().cpu().numpy())  # 将标签添加到列表中
                pos_all.append(pos.detach().cpu().numpy())  # 将位置添加到列表中

        infer_total_time = time.time() - start_total_time  # 计算总运行时间

        print('测试完成耗时：', infer_total_time)  # 打印测试完成信息

        preds_all = np.concatenate(preds_all, axis=0)  # 将所有预测结果拼接成一个数组
        labels_all = np.concatenate(labels_all, axis=0)  # 将所有标签拼接成一个数组
        pos_all = np.concatenate(pos_all, axis=0)  # 将所有位置拼接成一个数组

        preds_all = np.argmax(preds_all, 1)  # 获取预测结果的最大值索引
        preds_all = preds_all.astype(float)  # 将预测结果转换为浮点数

        oa, kappa, f1, pr, re = calculate_metrics(preds_all, labels_all)  # 计算评估指标
        print(' OA:{}\n Kappa:{}\n F1:{}\n Pr:{}\n Re:{}\n'.format(oa, kappa, f1, pr, re))  # 打印评估指标

        save_results_folder = f'{RESULT_ROOT}/Islandtown_shuguang/result/CD_result/CD_{self.loop}_0.01'  # 定义保存结果的文件夹路径
        if not os.path.exists(save_results_folder):
            os.makedirs(save_results_folder)  # 创建保存结果的文件夹
            print("Folder created successfully!")
        else:
            print("Folder already exists.")

        predict2img(preds_all, img_gt, pos_all, save_results_folder)  # 调用函数将预测结果转换为图像并保存

def run_test(loop):
    # if __name__ == "__main__":
    # 假设你有一个命令行参数解析器

    # 初始化参数解析器并设置描述
    parser = argparse.ArgumentParser(description="Testing")

    # 取块的时候 2 * padding + 1
    parser.add_argument('--padding', type=int, default=8)

    # 添加是否使用CUDA参数
    parser.add_argument('--cuda', type=bool, default=True)

    # 添加CUDA设备ID参数
    parser.add_argument('--device_id', type=int, default=0)

    # 解析参数
    args = parser.parse_args()

    tester = Tester(args, loop)
    tester.main()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing CD Model")
    parser.add_argument('--loop', type=int, default=0, help='Loop index')
    args = parser.parse_args()
    run_test(args.loop)