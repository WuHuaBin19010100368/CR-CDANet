import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.get_datasets import get_dataset, get_dataset_loop, get_cd_dataset_loop, get_dataset_CD_test


BOUNDARY_REGION_MASK = np.array([
    [1, 1, 1, 1, 3, 3, 3, 3, 4],
    [2, 1, 1, 1, 3, 3, 3, 4, 4],
    [2, 2, 1, 1, 3, 3, 4, 4, 4],
    [2, 2, 2, 1, 3, 4, 4, 4, 4],
    [2, 2, 2, 2, 0, 5, 5, 5, 5],
    [6, 6, 6, 6, 7, 8, 5, 5, 5],
    [6, 6, 6, 7, 7, 8, 8, 5, 5],
    [6, 6, 7, 7, 7, 8, 8, 8, 5],
    [6, 7, 7, 7, 7, 8, 8, 8, 8],
], dtype=np.int32)


def _label_ratio(gt, i, j, radius, label):
    h, w = gt.shape
    patch = gt[max(0, i - radius):min(h, i + radius + 1),
               max(0, j - radius):min(w, j + radius + 1)]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch == label))


def is_interior_high_confidence_point(gt, i, j, label, tau1=0.8, tau2=0.6):
    """High-confidence interior point: first-ring and second-ring consistency."""
    h, w = gt.shape
    if i < 0 or i >= h or j < 0 or j >= w or gt[i, j] != label:
        return False
    return _label_ratio(gt, i, j, 1, label) >= tau1 and _label_ratio(gt, i, j, 2, label) >= tau2


def is_boundary_high_confidence_point(gt, i, j, label, tau=0.8):
    """High-confidence boundary point based on two consecutive directional regions."""
    h, w = gt.shape
    if i < 0 or i >= h or j < 0 or j >= w or gt[i, j] != label:
        return False

    for region_id in range(1, 9):
        next_region_id = region_id + 1 if region_id < 8 else 1
        matched, total = 0, 0
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                x, y = i + dx, j + dy
                if x < 0 or x >= h or y < 0 or y >= w:
                    continue
                if BOUNDARY_REGION_MASK[dx + 4, dy + 4] in (region_id, next_region_id):
                    total += 1
                    matched += int(gt[x, y] == label)
        if total > 0 and matched / total >= tau:
            return True
    return False


def is_high_confidence_point(gt, i, j, label):
    """Pseudo-label selection used in CR-CDANet: interior or boundary confidence."""
    return (
        is_interior_high_confidence_point(gt, i, j, label)
        or is_boundary_high_confidence_point(gt, i, j, label)
    )



class DF_DOMAIN(Dataset):
    def __init__(self, dataset_name='Shuguang', crop_size=16, num_samples=10000, mode='train', train_val_split=0.999,
                 positive_ratio=0.0, loop = 0):
        assert mode in ['train', 'val', 'test'], "模式必须为 'train'、'val' 或 'test'"

        np.random.seed(42)
        torch.manual_seed(42)

        self.crop_size = crop_size
        self.num_samples = num_samples
        self.mode = mode
        self.train_val_split = train_val_split
        self.positive_ratio = positive_ratio
        self.loop = loop

        self.t1_img, self.t2_img, self.gt_img = get_dataset_loop(dataset_name, loop)

        # 获取原始图像尺寸
        self.original_h, self.original_w = self.gt_img.shape

        # 对图像进行填充
        self.t1_img = self.pad_image(self.t1_img)
        self.t2_img = self.pad_image(self.t2_img)
        self.gt_img = self.pad_image(self.gt_img, is_label=True)

        # 获取填充后的图像尺寸
        self.h, self.w = self.gt_img.shape

        # 计算 T1 和 T2 的均值和标准差
        self.t1_mean, self.t1_std = self.compute_mean_std(self.t1_img)
        self.t2_mean, self.t2_std = self.compute_mean_std(self.t2_img)

        # 生成测试集的切片位置（用于按顺序切分）

        # 生成训练/验证集的有效裁剪位置
        if self.mode in ['train', 'val']:
            # 获取所有有效的位置（裁剪窗口内至少有一个GT ==1）
            self.positive_positions = self.get_positive_positions()
            self.negative_positions = self.get_negative_positions()

            if len(self.positive_positions) == 0:
                raise ValueError("没有找到任何包含GT==1的裁剪窗口，请检查数据集或调整裁剪大小。")

            # 计算需要的正样本和负样本数量
            num_positive = int(self.num_samples * self.positive_ratio)
            num_negative = self.num_samples - num_positive

            print('设置的正负样本数量分别为：', num_positive, num_negative)

            # 如果负样本不足，允许重复采样
            if len(self.negative_positions) >= num_negative:
                selected_negative = np.random.choice(len(self.negative_positions), num_negative, replace=False)
                self.selected_negative_positions = [self.negative_positions[i] for i in selected_negative]
            else:
                selected_negative = np.random.choice(len(self.negative_positions), num_negative, replace=True)
                self.selected_negative_positions = [self.negative_positions[i] for i in selected_negative]

            # 如果正样本不足，允许重复采样
            if len(self.positive_positions) >= num_positive:
                selected_positive = np.random.choice(len(self.positive_positions), num_positive, replace=False)
                self.selected_positive_positions = [self.positive_positions[i] for i in selected_positive]
            else:
                selected_positive = np.random.choice(len(self.positive_positions), num_positive, replace=True)
                self.selected_positive_positions = [self.positive_positions[i] for i in selected_positive]

            self.valid_positions = self.selected_positive_positions + self.selected_negative_positions

            # 打乱位置列表
            np.random.shuffle(self.valid_positions)

            if self.mode == 'train':
                self.positions = self.valid_positions[:int(self.num_samples * self.train_val_split)]
                print('训练集样本的数目', len(self.positions))
            else:
                self.positions = self.valid_positions[int(self.num_samples * self.train_val_split): self.num_samples]
                print('验证集样本的数目', len(self.positions))
        elif self.mode == 'test':
            self.test_positions = self.generate_test_positions()

    def pad_image(self, img, is_label=False):
        """
        对图像进行填充，使其高度和宽度能够被裁剪大小整除。
        """
        h, w = img.shape[:2]
        pad_h = (self.crop_size - h % self.crop_size) % self.crop_size
        pad_w = (self.crop_size - w % self.crop_size) % self.crop_size

        if pad_h == 0 and pad_w == 0:
            return img  # 无需填充

        if is_label:
            # 标签图像，单通道
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        else:
            # T1 和 T2 图像，三通道
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        return img_padded

    def __len__(self):
        if self.mode in ['train', 'val']:
            if self.mode == 'train':
                return int(self.num_samples * self.train_val_split)
            else:
                return int(self.num_samples * (1 - self.train_val_split))
        else:
            return len(self.test_positions)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            # 随机裁剪
            i, j = self.positions[idx]
        else:
            # 顺序裁剪
            i, j = self.test_positions[idx]

        # 裁剪子图像
        t1_crop = self.t1_img[i:i + self.crop_size, j:j + self.crop_size]
        t2_crop = self.t2_img[i:i + self.crop_size, j:j + self.crop_size]
        gt_crop = self.gt_img[i:i + self.crop_size, j:j + self.crop_size]

        # 转换为浮点数并归一化（范围 [0, 1]）
        t1_crop = t1_crop.astype(np.float32) / 255.0
        t2_crop = t2_crop.astype(np.float32) / 255.0
        # max_t1_crop = np.max(t1_crop)
        # print(f"t1_crop 的最大值是: {max_t1_crop}")

        # 转换为 Tensor，并调整维度顺序为 [C, H, W]
        t1_crop = torch.from_numpy(t1_crop.transpose((2, 0, 1)))
        t2_crop = torch.from_numpy(t2_crop.transpose((2, 0, 1)))
        gt_crop = torch.from_numpy(gt_crop).long()  # 标签为整数

        return t1_crop, t2_crop, gt_crop

    def get_positive_positions(self):
        """
        获取所有满足条件的位置，即裁剪窗口内至少有一个GT ==1的区域。

        返回：
        - positions (list of tuples): [(i1, j1), (i2, j2), ...]
        """
        positions = []
        # 遍历所有可能的裁剪窗口位置，使用步幅
        for i in range(0, self.h - self.crop_size + 1, self.crop_size):
            for j in range(0, self.w - self.crop_size + 1, self.crop_size):
                gt_crop = self.gt_img[i:i + self.crop_size, j:j + self.crop_size]
                if np.any(gt_crop == 1):
                    positions.append((i, j))
        return positions

    def get_negative_positions(self):
        """
        获取所有不满足条件的位置，即裁剪窗口内所有GT ==0的区域。

        返回：
        - positions (list of tuples): [(i1, j1), (i2, j2), ...]
        """
        positions = []
        # 遍历所有可能的裁剪窗口位置，使用步幅
        for i in range(0, self.h - self.crop_size + 1, self.crop_size):
            for j in range(0, self.w - self.crop_size + 1, self.crop_size):
                gt_crop = self.gt_img[i:i + self.crop_size, j:j + self.crop_size]
                # if not np.any(gt_crop == 1):
                #     if kuai_is_high_confidence_point(self.gt_img, i, j, 0, radius=self.crop_size // 2 + 1):
                #         positions.append((i, j))
                # 计算总像素数和值为1的像素数
                total_pixels = gt_crop.size
                positive_pixels = np.sum(gt_crop == 1)
                if positive_pixels / total_pixels <= 0.3:
                    positions.append((i, j))

        print('满足条件的扩散块共有：', len(positions))
        # # 根据 loop 的值随机选择一定数量的负样本位置
        # if self.loop == 0:
        #     num_to_select = 200
        # elif self.loop == 1:
        #     num_to_select = 2000
        # elif self.loop == 2:
        #     num_to_select = 2000
        # elif self.loop == 3:
        #     num_to_select = 2000
        # else:
        #     num_to_select = len(positions)  # 默认选择所有负样本位置
        #
        # if len(positions) >= num_to_select:
        #     selected_positions = np.random.choice(len(positions), num_to_select, replace=False)
        # else:
        #     selected_positions = np.random.choice(len(positions), num_to_select, replace=True)
        #
        # print('扩散部分训练选择了：', len(selected_positions))
        # return [positions[i] for i in selected_positions]
        return positions

    def compute_mean_std(self, img):
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        std[std == 0] = 1e-8
        return mean, std

    def generate_test_positions(self):
        positions = []
        for i in range(0, self.h, self.crop_size):
            for j in range(0, self.w, self.crop_size):
                positions.append((i, j))
        return positions

    def reconstruct_image(self, patches):
        """
        从子图像列表重建原始图像，并去除填充部分。

        参数：
        - patches: 子图像列表，按照 generate_test_positions 的顺序

        返回：
        - reconstructed_img: 重建后的原始图像
        """
        # 初始化重建图像
        channels = patches[0].shape[0]  # 通道数
        reconstructed_img = torch.zeros((channels, self.h, self.w))

        idx = 0
        for i in range(0, self.h, self.crop_size):
            for j in range(0, self.w, self.crop_size):
                reconstructed_img[:, i:i + self.crop_size, j:j + self.crop_size] = patches[idx]
                idx += 1

        # 去除填充部分
        reconstructed_img = reconstructed_img[:, :self.original_h, :self.original_w]

        return reconstructed_img

class CD_DOMAIN(Dataset):
    def __init__(self, dataset_name='China', mode='train', patch_size=17, train_val_ratio=0.01, loop=0):
        super().__init__()

        self.dataset_name = dataset_name
        self.mode = mode
        self.padding = int(patch_size / 2)
        self.train_val_ratio = train_val_ratio
        self.loop = loop

        if self.loop == 0:
            self.train_val_ratio = 0.01
        elif self.loop == 1:
            self.train_val_ratio = 0.05
        elif self.loop == 2:
            self.train_val_ratio = 0.10
        elif self.loop == 3:
            self.train_val_ratio = 0.15
        elif self.loop == 4:
            self.train_val_ratio = 0.20
        # elif self.loop == 5:
        #     self.train_val_ratio = 0.25

        # if self.loop == 0:
        #     self.train_val_ratio = 0.01
        # elif self.loop == 1:
        #     self.train_val_ratio = 0.025
        # elif self.loop == 2:
        #     self.train_val_ratio = 0.05
        # elif self.loop == 3:
        #     self.train_val_ratio = 0.075
        # elif self.loop == 4:
        #     self.train_val_ratio = 0.10
        # elif self.loop == 5:
        #     self.train_val_ratio = 0.125
        # elif self.loop == 6:
        #     self.train_val_ratio = 0.15
        # elif self.loop == 7:
        #     self.train_val_ratio = 0.175
        # elif self.loop == 8:
        #     self.train_val_ratio = 0.20

        # 全标数据集 0为unchanged 1为changed 其中对于River数据集，255为changed，已经将其gt除以255
        if (self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River' or
                self.dataset_name == 'Italy' or self.dataset_name == 'California' or self.dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
            self.img_m1_t1, self.img_m1_t2, self.img_m2_t1, self.img_m2_t2, self.img_gt, self.img_gt0 = get_cd_dataset_loop(self.dataset_name, loop)
            # 在这里进行了标准化
            self.img_m1_t1, self.img_m1_t2 = self.data_preprocess(self.img_m1_t1), self.data_preprocess(self.img_m1_t2)
            self.img_m2_t1, self.img_m2_t2 = self.data_preprocess(self.img_m2_t1), self.data_preprocess(self.img_m2_t2)
            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_m1_t1_padding = cv2.copyMakeBorder(self.img_m1_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m1_t2_padding = cv2.copyMakeBorder(self.img_m1_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m2_t1_padding = cv2.copyMakeBorder(self.img_m2_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m2_t2_padding = cv2.copyMakeBorder(self.img_m2_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.random_points = self.get_random_points(self.img_gt, self.dataset_name, self.img_gt0)

        # 未全标数据集 1为changed 2为unchanged 所以得进行调整，将gt中的1变为2,2变为1
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
            self.img_m1_t1, self.img_m1_t2, self.img_m2_t1, self.img_m2_t2, self.img_gt, self.img_gt0 = get_cd_dataset_loop(self.dataset_name, loop)
            # 在这里进行了标准化
            self.img_m1_t1, self.img_m1_t2 = self.data_preprocess(self.img_m1_t1), self.data_preprocess(self.img_m1_t2)
            self.img_m2_t1, self.img_m2_t2 = self.data_preprocess(self.img_m2_t1), self.data_preprocess(self.img_m2_t2)

            img_gt_tmp = np.zeros_like(self.img_gt)
            img_gt_tmp[self.img_gt == 1.] = 2.
            self.img_gt[self.img_gt == 2.] = 1.
            self.img_gt[img_gt_tmp == 2.] = 2.

            img_gt_tmp0 = np.zeros_like(self.img_gt0)
            img_gt_tmp0[self.img_gt0 == 1.] = 2.
            self.img_gt0[self.img_gt0 == 2.] = 1.
            self.img_gt0[img_gt_tmp0 == 2.] = 2.

            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_m1_t1_padding = cv2.copyMakeBorder(self.img_m1_t1, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m1_t2_padding = cv2.copyMakeBorder(self.img_m1_t2, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m2_t1_padding = cv2.copyMakeBorder(self.img_m2_t1, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m2_t2_padding = cv2.copyMakeBorder(self.img_m2_t2, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.random_points = self.get_random_points(self.img_gt, self.dataset_name, self.img_gt0)


    def get_random_points(self, gt, dataset_name, gt0):
        random.seed(42)

        all_num = self.h * self.w

        whole_point0 = gt0.reshape(1, all_num)
        changed_indices0 = []
        unchanged_indices0 = []
        if (dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River' or dataset_name == 'Italy'
                or dataset_name == 'California' or dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
            changed_indices0 = np.where(whole_point0[0] == 1.)[0].tolist()
            unchanged_indices0 = np.where(whole_point0[0] == 0.)[0].tolist()
        elif dataset_name == 'BayArea' or dataset_name == 'Barbara' or self.dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
            changed_indices0 = np.where(whole_point0[0] == 2.)[0].tolist()
            unchanged_indices0 = np.where(whole_point0[0] == 1.)[0].tolist()

        print(f"这张图gt的变化点有: {len(changed_indices0)}")
        print(f"这张图gt的未变化点有: {len(unchanged_indices0)}")

        whole_point = gt.reshape(1, all_num)
        random_points = []
        changed_indices = []
        unchanged_indices = []
        if (dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River' or dataset_name == 'Italy'
                or dataset_name == 'California' or dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
            changed_indices = np.where(whole_point[0] == 1.)[0].tolist()
            unchanged_indices = np.where(whole_point[0] == 0.)[0].tolist()
        elif dataset_name == 'BayArea' or dataset_name == 'Barbara' or self.dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
            changed_indices = np.where(whole_point[0] == 2.)[0].tolist()
            unchanged_indices = np.where(whole_point[0] == 1.)[0].tolist()

        print(f"根据上个结果测出来的变化点有: {len(changed_indices)}")
        print(f"根据上个结果测出来的未变化点有: {len(unchanged_indices)}")

        if (dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River' or dataset_name == 'Italy'
                or dataset_name == 'California' or dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
            # 过滤高置信度的变化点
            high_confidence_changed_indices = []
            for idx in changed_indices:
                i, j = divmod(idx, self.w)
                if is_high_confidence_point(gt, i, j, 1):
                    high_confidence_changed_indices.append(idx)

            # 过滤高置信度的未变化点
            high_confidence_unchanged_indices = []
            for idx in unchanged_indices:
                i, j = divmod(idx, self.w)
                if is_high_confidence_point(gt, i, j, 0):
                    high_confidence_unchanged_indices.append(idx)

        elif dataset_name == 'BayArea' or dataset_name == 'Barbara' or self.dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
            # 过滤高置信度的变化点
            high_confidence_changed_indices = []
            for idx in changed_indices:
                i, j = divmod(idx, self.w)
                if is_high_confidence_point(gt, i, j, 2):
                    high_confidence_changed_indices.append(idx)

            # 过滤高置信度的未变化点
            high_confidence_unchanged_indices = []
            for idx in unchanged_indices:
                i, j = divmod(idx, self.w)
                if is_high_confidence_point(gt, i, j, 1):
                    high_confidence_unchanged_indices.append(idx)

        print(f"高置信度的变化点有: {len(high_confidence_changed_indices)}")
        print(f"高置信度的未变化点有: {len(high_confidence_unchanged_indices)}")

        if self.loop == 0:
            # num_changed_train = int(len(changed_indices) * self.train_val_ratio)
            num_changed_train = 200
            num_unchanged_train = int(len(unchanged_indices) * self.train_val_ratio)

            changed_points_train = random.sample(changed_indices, num_changed_train)
            unchanged_points_train = random.sample(unchanged_indices, num_unchanged_train)

        else :
            # num_changed_train0 = int(len(changed_indices0) * 0.01)
            num_changed_train0 = 200
            num_unchanged_train0 = int(len(unchanged_indices0) * 0.01)

            changed_points_train0 = random.sample(changed_indices0, num_changed_train0)
            unchanged_points_train0 = random.sample(unchanged_indices0, num_unchanged_train0)

            num_changed_train = int(len(changed_indices0) * (self.train_val_ratio - 0.01))
            num_unchanged_train = int(len(unchanged_indices0) * (self.train_val_ratio - 0.01))

            # 剔除已经选择的点
            remaining_changed_indices = list(set(high_confidence_changed_indices) - set(changed_points_train0))
            remaining_unchanged_indices = list(set(high_confidence_unchanged_indices) - set(unchanged_points_train0))
            print(f"变化检测图剔除原始gt后的变化点有: {len(remaining_changed_indices)}")
            print(f"变化检测图剔除原始gt后的未变化点有: {len(remaining_unchanged_indices)}")

            # 对于变化点
            if len(high_confidence_changed_indices) >= num_changed_train:
                changed_points_train = random.sample(remaining_changed_indices, num_changed_train)  # 无放回抽样
            else:
                changed_points_train = random.choices(remaining_changed_indices, k=num_changed_train)  # 有放回抽样

            # 对于未变化点
            if len(high_confidence_unchanged_indices) >= num_unchanged_train:
                unchanged_points_train = random.sample(remaining_unchanged_indices, num_unchanged_train)
            else:
                unchanged_points_train = random.choices(remaining_unchanged_indices, k=num_unchanged_train)

            # 最后将 changed_points_train0 和 unchanged_points_train0 加入
            changed_points_train.extend(changed_points_train0)
            unchanged_points_train.extend(unchanged_points_train0)

        print(f"用于训练的变化点有: {len(changed_points_train)}")
        print(f"用于训练的未变化点有: {len(unchanged_points_train)}")

        changed_points_val = random.sample(list(set(changed_indices) - set(changed_points_train)),
                                           int(len(changed_indices) * 0.0014))
        unchanged_points_val = random.sample(list(set(unchanged_indices) - set(unchanged_points_train)),
                                             int(len(changed_indices) * 0.0014))

        if self.mode == 'train':
            random_points = changed_points_train + unchanged_points_train
        elif self.mode == 'val':
            random_points = changed_points_val + unchanged_points_val
        elif self.mode == 'test':
            if (dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River' or
                    dataset_name == 'Italy' or dataset_name == 'California' or dataset_name == 'Shuguang'
                    or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                    or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
                random_points = list(range(all_num))
            elif dataset_name == 'BayArea' or dataset_name == 'Barbara' or self.dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
                random_points = np.nonzero(whole_point[0])[0].tolist()
        else:
            raise ValueError(f"Invalid mode {self.mode}. Expected one of: train, val, test.")

        return random_points

    def data_preprocess(self, img):
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))

        img_normalized = (img - mean) / std

        return img_normalized

    def __len__(self):
        return len(self.random_points)

    def __getitem__(self, index):
        original_i, original_j = divmod(self.random_points[index], self.w)  # 第几行，第几列
        new_i = original_i + self.padding
        new_j = original_j + self.padding

        img_patch_m1_t1 = self.img_m1_t1_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m1_t2 = self.img_m1_t2_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m2_t1 = self.img_m2_t1_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m2_t2 = self.img_m2_t2_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)

        img_patch_m1_t1 = torch.from_numpy(img_patch_m1_t1)
        img_patch_m1_t2 = torch.from_numpy(img_patch_m1_t2)
        img_patch_m2_t1 = torch.from_numpy(img_patch_m2_t1)
        img_patch_m2_t2 = torch.from_numpy(img_patch_m2_t2)

        if (self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River' or
                self.dataset_name == 'Italy' or self.dataset_name == 'California' or self.dataset_name == 'Shuguang'
                    or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
            label = self.img_gt[original_i, original_j]
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
            label = self.img_gt[original_i, original_j] - 1.
        label = torch.tensor(label, dtype=torch.float32)

        return img_patch_m1_t1, img_patch_m1_t2, img_patch_m2_t1, img_patch_m2_t2, label, torch.tensor((original_i, original_j), dtype=torch.long)

class CD_DOMAIN_test(Dataset):
    def __init__(self, dataset_name='China', mode='train', patch_size=17, train_val_ratio=0.01, loop = None):
        super().__init__()

        self.dataset_name = dataset_name
        self.mode = mode
        self.padding = int(patch_size / 2)
        self.train_val_ratio = train_val_ratio

        # 全标数据集 0为unchanged 1为changed 其中对于River数据集，255为changed，已经将其gt除以255
        if (self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River' or
                self.dataset_name == 'Italy' or self.dataset_name == 'California' or self.dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
            self.img_m1_t1, self.img_m1_t2, self.img_m2_t1, self.img_m2_t2, self.img_gt = get_dataset_CD_test(self.dataset_name, loop)
            # 在这里进行了标准化
            self.img_m1_t1, self.img_m1_t2 = self.data_preprocess(self.img_m1_t1), self.data_preprocess(self.img_m1_t2)
            self.img_m2_t1, self.img_m2_t2 = self.data_preprocess(self.img_m2_t1), self.data_preprocess(self.img_m2_t2)
            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_m1_t1_padding = cv2.copyMakeBorder(self.img_m1_t1, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m1_t2_padding = cv2.copyMakeBorder(self.img_m1_t2, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m2_t1_padding = cv2.copyMakeBorder(self.img_m2_t1, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m2_t2_padding = cv2.copyMakeBorder(self.img_m2_t2, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.random_points = self.get_random_points(self.img_gt, self.dataset_name)

        # 未全标数据集 1为changed 2为unchanged 所以得进行调整，将gt中的1变为2,2变为1
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
            self.img_m1_t1, self.img_m1_t2, self.img_m2_t1, self.img_m2_t2, self.img_gt = get_dataset_CD_test(self.dataset_name, loop)
            self.img_m1_t1, self.img_m1_t2 = self.data_preprocess(self.img_m1_t1), self.data_preprocess(self.img_m1_t2)
            self.img_m2_t1, self.img_m2_t2 = self.data_preprocess(self.img_m2_t1), self.data_preprocess(self.img_m2_t2)
            img_gt_tmp = np.zeros_like(self.img_gt)
            img_gt_tmp[self.img_gt == 1.] = 2.
            self.img_gt[self.img_gt == 2.] = 1.
            self.img_gt[img_gt_tmp == 2.] = 2.

            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_m1_t1_padding = cv2.copyMakeBorder(self.img_m1_t1, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m1_t2_padding = cv2.copyMakeBorder(self.img_m1_t2, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m2_t1_padding = cv2.copyMakeBorder(self.img_m2_t1, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.img_m2_t2_padding = cv2.copyMakeBorder(self.img_m2_t2, self.padding, self.padding, self.padding,
                                                        self.padding,
                                                        cv2.BORDER_REPLICATE)
            self.random_points = self.get_random_points(self.img_gt, self.dataset_name)

        else:
            raise ValueError(f"Invalid dataset name {dataset_name}. Expected one of: China, Hermiston, River, "
                             f"BayArea, Barbara.")

    def get_random_points(self, gt, dataset_name):
        random.seed(42)

        all_num = self.h * self.w
        whole_point = gt.reshape(1, all_num)
        random_points = []

        changed_indices = []
        unchanged_indices = []
        if (dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River' or dataset_name == 'Italy'
                or dataset_name == 'California' or dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
            changed_indices = np.where(whole_point[0] == 1.)[0].tolist()
            unchanged_indices = np.where(whole_point[0] == 0.)[0].tolist()
        elif dataset_name == 'BayArea' or dataset_name == 'Barbara' or dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
            changed_indices = np.where(whole_point[0] == 2.)[0].tolist()
            unchanged_indices = np.where(whole_point[0] == 1.)[0].tolist()

        print(f"这张图的变化点有: {len(changed_indices)}")
        print(f"这张图的未变化点有: {len(unchanged_indices)}")

        num_changed_train = int(len(changed_indices) * self.train_val_ratio)
        num_unchanged_train = int(len(unchanged_indices) * self.train_val_ratio)

        changed_points_train = random.sample(changed_indices, num_changed_train)
        unchanged_points_train = random.sample(unchanged_indices, num_unchanged_train)
        print(f"用于训练的变化点有: {len(changed_points_train)}")
        print(f"用于训练的未变化点有: {len(unchanged_points_train)}")

        changed_points_val = random.sample(list(set(changed_indices) - set(changed_points_train)),
                                           int(len(changed_indices) * 0.0014))
        unchanged_points_val = random.sample(list(set(unchanged_indices) - set(unchanged_points_train)),
                                             int(len(changed_indices) * 0.0014))

        if self.mode == 'train':
            random_points = changed_points_train + unchanged_points_train
        elif self.mode == 'val':
            random_points = changed_points_val + unchanged_points_val
        elif self.mode == 'test':
            if (dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River' or
                    dataset_name == 'Italy' or dataset_name == 'California' or dataset_name == 'Shuguang'
                    or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                    or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
                random_points = list(range(all_num))
            elif dataset_name == 'BayArea' or dataset_name == 'Barbara' or dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
                random_points = np.nonzero(whole_point[0])[0].tolist()
        else:
            raise ValueError(f"Invalid mode {self.mode}. Expected one of: train, val, test.")

        return random_points

    def data_preprocess(self, img):
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))

        img_normalized = (img - mean) / std

        return img_normalized

    def __len__(self):
        return len(self.random_points)

    def __getitem__(self, index):
        original_i, original_j = divmod(self.random_points[index], self.w)  # 第几行，第几列
        new_i = original_i + self.padding
        new_j = original_j + self.padding

        img_patch_m1_t1 = self.img_m1_t1_padding[new_i - self.padding:new_i + self.padding + 1,
                          new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m1_t2 = self.img_m1_t2_padding[new_i - self.padding:new_i + self.padding + 1,
                          new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m2_t1 = self.img_m2_t1_padding[new_i - self.padding:new_i + self.padding + 1,
                          new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m2_t2 = self.img_m2_t2_padding[new_i - self.padding:new_i + self.padding + 1,
                          new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)

        img_patch_m1_t1 = torch.from_numpy(img_patch_m1_t1)
        img_patch_m1_t2 = torch.from_numpy(img_patch_m1_t2)
        img_patch_m2_t1 = torch.from_numpy(img_patch_m2_t1)
        img_patch_m2_t2 = torch.from_numpy(img_patch_m2_t2)

        if (self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River' or
                self.dataset_name == 'Italy' or self.dataset_name == 'California' or self.dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'shuguang_domain' or self.dataset_name == 'Islandtown_domain'):
            label = self.img_gt[original_i, original_j]
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'barbara_domain' or self.dataset_name == 'liyukou_domain':
            label = self.img_gt[original_i, original_j] - 1.
        label = torch.tensor(label, dtype=torch.float32)

        return img_patch_m1_t1, img_patch_m1_t2, img_patch_m2_t1, img_patch_m2_t2, label, torch.tensor(
            (original_i, original_j), dtype=torch.long)

class CD_DOMAIN_Dataset(Dataset):
    def __init__(self, dataset_name='China', mode='train', patch_size=17, train_val_ratio=0.01):
        super().__init__()

        self.dataset_name = dataset_name
        self.mode = mode
        self.padding = int(patch_size / 2)
        self.train_val_ratio = train_val_ratio

        # 全标数据集 0为unchanged 1为changed 其中对于River数据集，255为changed，已经将其gt除以255
        if (self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River' or
                self.dataset_name == 'Italy' or self.dataset_name == 'California' or self.dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'Shuguang_domain' or self.dataset_name == 'Islandtown_domain' or self.dataset_name == 'Islandtown_source'):
            self.img_m1_t1, self.img_m1_t2, self.img_m2_t1, self.img_m2_t2, self.img_gt = get_dataset(self.dataset_name)
            # 在这里进行了标准化
            self.img_m1_t1, self.img_m1_t2 = self.data_preprocess(self.img_m1_t1), self.data_preprocess(self.img_m1_t2)
            self.img_m2_t1, self.img_m2_t2 = self.data_preprocess(self.img_m2_t1), self.data_preprocess(self.img_m2_t2)
            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_m1_t1_padding = cv2.copyMakeBorder(self.img_m1_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m1_t2_padding = cv2.copyMakeBorder(self.img_m1_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m2_t1_padding = cv2.copyMakeBorder(self.img_m2_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m2_t2_padding = cv2.copyMakeBorder(self.img_m2_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.random_points = self.get_random_points(self.img_gt, self.dataset_name)

        # 未全标数据集 1为changed 2为unchanged 所以得进行调整，将gt中的1变为2,2变为1
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'liyukou_source' \
                or self.dataset_name == 'barbara_source':
            self.img_m1_t1, self.img_m1_t2, self.img_m2_t1, self.img_m2_t2, self.img_gt = get_dataset(self.dataset_name)
            self.img_m1_t1, self.img_m1_t2 = self.data_preprocess(self.img_m1_t1), self.data_preprocess(self.img_m1_t2)
            self.img_m2_t1, self.img_m2_t2 = self.data_preprocess(self.img_m2_t1), self.data_preprocess(self.img_m2_t2)
            img_gt_tmp = np.zeros_like(self.img_gt)
            img_gt_tmp[self.img_gt == 1.] = 2.
            self.img_gt[self.img_gt == 2.] = 1.
            self.img_gt[img_gt_tmp == 2.] = 2.

            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_m1_t1_padding = cv2.copyMakeBorder(self.img_m1_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m1_t2_padding = cv2.copyMakeBorder(self.img_m1_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m2_t1_padding = cv2.copyMakeBorder(self.img_m2_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.img_m2_t2_padding = cv2.copyMakeBorder(self.img_m2_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REPLICATE)
            self.random_points = self.get_random_points(self.img_gt, self.dataset_name)

        else:
            raise ValueError(f"Invalid dataset name {dataset_name}. Expected one of: China, Hermiston, River, "
                             f"BayArea, Barbara.")

    def get_random_points(self, gt, dataset_name):
        random.seed(42)

        all_num = self.h * self.w
        whole_point = gt.reshape(1, all_num)
        random_points = []

        changed_indices = []
        unchanged_indices = []
        if (dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River' or dataset_name == 'Italy'
                or dataset_name == 'California' or dataset_name == 'Shuguang'
                or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'Shuguang_domain' or self.dataset_name == 'Islandtown_domain' or self.dataset_name == 'Islandtown_source'):
            changed_indices = np.where(whole_point[0] == 1.)[0].tolist()
            unchanged_indices = np.where(whole_point[0] == 0.)[0].tolist()
        elif dataset_name == 'BayArea' or dataset_name == 'Barbara' or dataset_name == 'liyukou_source' or self.dataset_name == 'barbara_source':
            changed_indices = np.where(whole_point[0] == 2.)[0].tolist()
            unchanged_indices = np.where(whole_point[0] == 1.)[0].tolist()

        print(f"这张图的变化点有: {len(changed_indices)}")
        print(f"这张图的未变化点有: {len(unchanged_indices)}")

        num_changed_train = int(len(changed_indices) * self.train_val_ratio)
        num_unchanged_train = int(len(unchanged_indices) * self.train_val_ratio)

        changed_points_train = random.sample(changed_indices, num_changed_train)
        # changed_points_train = random.sample(changed_indices, num_unchanged_train)
        unchanged_points_train = random.sample(unchanged_indices, num_unchanged_train)
        print(f"用于训练的变化点有: {len(changed_points_train)}")
        print(f"用于训练的未变化点有: {len(unchanged_points_train)}")

        changed_points_val = random.sample(list(set(changed_indices) - set(changed_points_train)),
                                           int(len(changed_indices) * 0.0014))
        unchanged_points_val = random.sample(list(set(unchanged_indices) - set(unchanged_points_train)),
                                             int(len(changed_indices) * 0.0014))

        if self.mode == 'train':
            random_points = changed_points_train + unchanged_points_train
        elif self.mode == 'val':
            random_points = changed_points_val + unchanged_points_val
        elif self.mode == 'test':
            if (dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River' or
                    dataset_name == 'Italy' or dataset_name == 'California' or dataset_name == 'Shuguang'
                    or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                    or self.dataset_name == 'Shuguang_domain' or self.dataset_name == 'Islandtown_domain' or self.dataset_name == 'Islandtown_source'):
                random_points = list(range(all_num))
            elif dataset_name == 'BayArea' or dataset_name == 'Barbara' or dataset_name == 'liyukou_source' or self.dataset_name == 'barbara_source':
                random_points = np.nonzero(whole_point[0])[0].tolist()
        else:
            raise ValueError(f"Invalid mode {self.mode}. Expected one of: train, val, test.")

        return random_points

    def data_preprocess(self, img):
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))

        img_normalized = (img - mean) / std

        return img_normalized

    def __len__(self):
        return len(self.random_points)

    def __getitem__(self, index):
        original_i, original_j = divmod(self.random_points[index], self.w)  # 第几行，第几列
        new_i = original_i + self.padding
        new_j = original_j + self.padding

        img_patch_m1_t1 = self.img_m1_t1_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m1_t2 = self.img_m1_t2_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m2_t1 = self.img_m2_t1_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_m2_t2 = self.img_m2_t2_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)

        img_patch_m1_t1 = torch.from_numpy(img_patch_m1_t1)
        img_patch_m1_t2 = torch.from_numpy(img_patch_m1_t2)
        img_patch_m2_t1 = torch.from_numpy(img_patch_m2_t1)
        img_patch_m2_t2 = torch.from_numpy(img_patch_m2_t2)

        if (self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River' or
                self.dataset_name == 'Italy' or self.dataset_name == 'California' or self.dataset_name == 'Shuguang'
                    or self.dataset_name == 'Shuguang_RGB' or self.dataset_name == 'Shuguang_SAR'
                or self.dataset_name == 'Shuguang_domain' or self.dataset_name == 'Islandtown_domain' or self.dataset_name == 'Islandtown_source'):
            label = self.img_gt[original_i, original_j]
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara' or self.dataset_name == 'liyukou_source' or self.dataset_name == 'barbara_source':
            label = self.img_gt[original_i, original_j] - 1.
        label = torch.tensor(label, dtype=torch.float32)

        return img_patch_m1_t1, img_patch_m1_t2, img_patch_m2_t1, img_patch_m2_t2, label, torch.tensor((original_i, original_j), dtype=torch.long)

class DDPMDataset_diffusion_1(Dataset):
    def __init__(self, dataset_name='Shuguang', crop_size=16, num_samples=20000, mode='train', train_val_split=0.9999,
                 positive_ratio=0.0):
        assert mode in ['train', 'val', 'test'], "模式必须为 'train'、'val' 或 'test'"

        np.random.seed(42)
        torch.manual_seed(42)

        self.crop_size = crop_size
        self.num_samples = num_samples
        self.mode = mode
        self.train_val_split = train_val_split
        self.positive_ratio = positive_ratio
        self.dataset_name = dataset_name

        # 加载图像
        self.t1_img, self.t2_img, self.gt_img = get_dataset(dataset_name)

        # 获取原始图像尺寸
        self.original_h, self.original_w = self.gt_img.shape

        # 对图像进行填充
        self.t1_img = self.pad_image(self.t1_img)
        self.t2_img = self.pad_image(self.t2_img)
        self.gt_img = self.pad_image(self.gt_img, is_label=True)

        # 获取填充后的图像尺寸
        self.h, self.w = self.gt_img.shape

        # 计算 T1 和 T2 的均值和标准差
        self.t1_mean, self.t1_std = self.compute_mean_std(self.t1_img)
        self.t2_mean, self.t2_std = self.compute_mean_std(self.t2_img)

        # 生成测试集的切片位置（用于按顺序切分）

        # 生成训练/验证集的有效裁剪位置
        if self.mode in ['train', 'val']:
            # 获取所有有效的位置（裁剪窗口内至少有一个GT ==1）
            self.positive_positions = self.get_positive_positions()
            self.negative_positions = self.get_negative_positions()

            if len(self.positive_positions) == 0:
                raise ValueError("没有找到任何包含GT==1的裁剪窗口，请检查数据集或调整裁剪大小。")

            print('可以选择的样本数为', len(self.negative_positions))
            # 计算需要的正样本和负样本数量
            num_positive = int(self.num_samples * self.positive_ratio)
            num_negative = self.num_samples - num_positive

            print('设置的正负样本数量分别为：', num_positive, num_negative)

            # 如果负样本不足，允许重复采样
            if len(self.negative_positions) >= num_negative:
                selected_negative = np.random.choice(len(self.negative_positions), num_negative, replace=False)
                self.selected_negative_positions = [self.negative_positions[i] for i in selected_negative]
            else:
                selected_negative = np.random.choice(len(self.negative_positions), num_negative, replace=True)
                self.selected_negative_positions = [self.negative_positions[i] for i in selected_negative]

            # 如果正样本不足，允许重复采样
            if len(self.positive_positions) >= num_positive:
                selected_positive = np.random.choice(len(self.positive_positions), num_positive, replace=False)
                self.selected_positive_positions = [self.positive_positions[i] for i in selected_positive]
            else:
                selected_positive = np.random.choice(len(self.positive_positions), num_positive, replace=True)
                self.selected_positive_positions = [self.positive_positions[i] for i in selected_positive]

            self.valid_positions = self.selected_positive_positions + self.selected_negative_positions

            # 打乱位置列表
            np.random.shuffle(self.valid_positions)

            if self.mode == 'train':
                self.positions = self.valid_positions[:int(self.num_samples * self.train_val_split)]
                print('训练集样本的数目', len(self.positions))
            else:
                self.positions = self.valid_positions[int(self.num_samples * self.train_val_split): self.num_samples]
                print('验证集样本的数目', len(self.positions))
        elif self.mode == 'test':
            self.test_positions = self.generate_test_positions()

    def pad_image(self, img, is_label=False):
        """
        对图像进行填充，使其高度和宽度能够被裁剪大小整除。
        """
        h, w = img.shape[:2]
        pad_h = (self.crop_size - h % self.crop_size) % self.crop_size
        pad_w = (self.crop_size - w % self.crop_size) % self.crop_size

        if pad_h == 0 and pad_w == 0:
            return img  # 无需填充

        if is_label:
            # 标签图像，单通道
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        else:
            # T1 和 T2 图像，三通道
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        return img_padded

    def __len__(self):
        if self.mode in ['train', 'val']:
            if self.mode == 'train':
                return int(self.num_samples * self.train_val_split)
            else:
                return int(self.num_samples * (1 - self.train_val_split))
        else:
            return len(self.test_positions)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            # 随机裁剪
            i, j = self.positions[idx]
        else:
            # 顺序裁剪
            i, j = self.test_positions[idx]

        # 裁剪子图像
        t1_crop = self.t1_img[i:i + self.crop_size, j:j + self.crop_size]
        t2_crop = self.t2_img[i:i + self.crop_size, j:j + self.crop_size]
        gt_crop = self.gt_img[i:i + self.crop_size, j:j + self.crop_size]

        # 转换为浮点数并归一化（范围 [0, 1]）
        if self.dataset_name == 'liyukou' or 'barbara':
            t1_crop = t1_crop.astype(np.float32)
            t2_crop = t2_crop.astype(np.float32)
        else:
            t1_crop = t1_crop.astype(np.float32) / 255.0
            t2_crop = t2_crop.astype(np.float32) / 255.0
        # max_t1_crop = np.max(t1_crop)
        # print(f"t1_crop 的最大值是: {max_t1_crop}")

        # 转换为 Tensor，并调整维度顺序为 [C, H, W]
        t1_crop = torch.from_numpy(t1_crop.transpose((2, 0, 1)))
        t2_crop = torch.from_numpy(t2_crop.transpose((2, 0, 1)))
        gt_crop = torch.from_numpy(gt_crop).long()  # 标签为整数

        return t1_crop, t2_crop, gt_crop

    def get_positive_positions(self):
        """
        获取所有满足条件的位置，即裁剪窗口内至少有一个GT ==1的区域。

        返回：
        - positions (list of tuples): [(i1, j1), (i2, j2), ...]
        """
        positions = []
        # 遍历所有可能的裁剪窗口位置，使用步幅
        for i in range(0, self.h - self.crop_size + 1, self.crop_size):
            for j in range(0, self.w - self.crop_size + 1, self.crop_size):
                gt_crop = self.gt_img[i:i + self.crop_size, j:j + self.crop_size]
                if np.any(gt_crop == 1):
                    positions.append((i, j))
        return positions

    def get_negative_positions(self):
        """
        获取所有不满足条件的位置，即裁剪窗口内所有GT ==0的区域。

        返回：
        - positions (list of tuples): [(i1, j1), (i2, j2), ...]
        """
        positions = []
        # 遍历所有可能的裁剪窗口位置，使用步幅
        for i in range(0, self.h - self.crop_size + 1, self.crop_size):
            for j in range(0, self.w - self.crop_size + 1, self.crop_size):
                gt_crop = self.gt_img[i:i + self.crop_size, j:j + self.crop_size]
                if not np.any(gt_crop == 1):
                    positions.append((i, j))
        return positions

    def compute_mean_std(self, img):
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        std[std == 0] = 1e-8
        return mean, std

    def generate_test_positions(self):
        positions = []
        for i in range(0, self.h, self.crop_size):
            for j in range(0, self.w, self.crop_size):
                positions.append((i, j))
        return positions

    def reconstruct_image(self, patches):
        """
        从子图像列表重建原始图像，并去除填充部分。

        参数：
        - patches: 子图像列表，按照 generate_test_positions 的顺序

        返回：
        - reconstructed_img: 重建后的原始图像
        """
        # 初始化重建图像
        channels = patches[0].shape[0]  # 通道数
        reconstructed_img = torch.zeros((channels, self.h, self.w))

        idx = 0
        for i in range(0, self.h, self.crop_size):
            for j in range(0, self.w, self.crop_size):
                reconstructed_img[:, i:i + self.crop_size, j:j + self.crop_size] = patches[idx]
                idx += 1

        # 去除填充部分
        reconstructed_img = reconstructed_img[:, :self.original_h, :self.original_w]

        return reconstructed_img