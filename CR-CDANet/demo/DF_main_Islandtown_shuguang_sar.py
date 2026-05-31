import os
import random
import numpy as np
import scipy.io
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.data_loader import DDPMDataset_diffusion_1, DF_DOMAIN
from diffusion.ddpm_3_11 import DDPM
from diffusion.metrics_utils import psnr

RESULT_ROOT = os.environ.get("CRCDANET_RESULT_ROOT", "./z_result")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

source_data = DDPMDataset_diffusion_1(dataset_name='Islandtown', mode='train')
source_loader = DataLoader(source_data, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

def main(loop):

    train_data = DF_DOMAIN(dataset_name='shuguang_domain', mode='train', loop=loop)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    # val_data = HSICD_Dataset(dataset_name=self.dataset_name, mode='val', patch_size=self.patch_size)
    val_data = DF_DOMAIN(dataset_name='shuguang_domain', mode='val', loop=loop)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 加载测试数据
    dataset_name = 'shuguang_domain'
    test_data = DF_DOMAIN(dataset_name=dataset_name, mode='test', loop=loop)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # 设置随机种子
    set_seed(42)  # 选择一个固定的种子值

    # 设置训练参数
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    channels = 3
    out_channels = 3
    loss_type = 'l2'
    conditional = True
    load_net = False
    name = 'ddpm_model'
    resume_training = True  # 设置为 True 时加载预训练模型

    best_psnr = 0.0
    save_interval = 10  # 每多少个 epoch 保存一次模型

    # 初始化模型
    ddpm = DDPM(
        device=device,
        channels=channels,
        out_channels=out_channels,
        loss_type=loss_type,
        conditional=conditional,
        load_net=load_net,
        name=name
    )

    # 如果 resume_training 为 True，则加载之前的训练模型
    # 如果 resume_training 为 True，则加载之前的训练模型
    if resume_training:
        if loop == 0:
            target_model_path = f'{RESULT_ROOT}/source/Islandtown/model/DF_model/rgb_sar/ddpm_model-best.pt'
        elif loop == 1:
            target_model_path = f'{RESULT_ROOT}/Islandtown_shuguang/model/DF_model/rgb_sar_0/ddpm_model-best.pt'
        elif loop == 2:
            target_model_path = f'{RESULT_ROOT}/Islandtown_shuguang/model/DF_model/rgb_sar_1/ddpm_model-best.pt'
        elif loop == 3:
            target_model_path = f'{RESULT_ROOT}/Islandtown_shuguang/model/DF_model/rgb_sar_2/ddpm_model-best.pt'

        source_model_path = f'{RESULT_ROOT}/source/Islandtown/model/DF_model/rgb_sar/ddpm_model-best.pt'
        ddpm.load_network(target_model_path, source_model_path)
        print(f"源域加载原始模型不参与训练，目标域加载预训练好的源域模型参与训练")

    # 初始化 MultiStepLR 调度器
    scheduler = lr_scheduler.MultiStepLR(ddpm.optG, milestones=[300], gamma=0.1)

    # 训练循环
    if loop == 0:
        num_epochs = 100
    elif loop == 1:
        num_epochs = 50
    elif loop == 2:
        num_epochs = 50
    elif loop == 3:
        num_epochs = 50

    print("扩散部分 开始训练！！！！！！！！！！！！！！！！！！！！")

    source_iter = iter(source_loader)  # 创建 source_loader 的迭代器

    for epoch in range(num_epochs):
        ddpm.netG.train()
        total_loss = 0.0
        target_loss = 0.0
        mmd_coral_loss = 0.0
        total_mmd_loss = 0.0
        total_coral_loss = 0.0

        for batch_idx, (img_t1, img_t2, label) in enumerate(tqdm(train_loader)):
            img_t1 = img_t1.to(device)
            img_t2 = img_t2.to(device)

            # 尝试从 source_loader 中获取下一个批次的数据
            try:
                source_img_t1, source_img_t2, source_label = next(source_iter)
            except StopIteration:
                # 如果 source_loader 迭代完毕，重新创建迭代器并获取下一个批次的数据
                source_iter = iter(source_loader)  # 重新迭代 source_loader
                source_img_t1, source_img_t2, source_label = next(source_iter)
            source_img_t2 = source_img_t2.to(device)
            source_img_t1 = source_img_t1.to(device)

            loss, target_loss, mmd_coral_loss, total_mmd_loss, total_coral_loss = ddpm.optimize_parameters(img_t2, img_t1, source_img_t2, source_img_t1)
            # loss = ddpm.optimize_parameters(条件!!!!!, 要转的噪声!!!!!)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            total_loss += loss.item()
            target_loss += target_loss.item()
            mmd_coral_loss = torch.tensor(mmd_coral_loss)
            mmd_coral_loss += mmd_coral_loss.item()
            total_mmd_loss = torch.tensor(total_mmd_loss)
            total_mmd_loss += total_mmd_loss.item()
            total_coral_loss = torch.tensor(total_coral_loss)
            total_coral_loss += total_coral_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_target_loss = target_loss / len(train_loader)
        avg_mmd_coral_loss = mmd_coral_loss / len(train_loader)
        avg_total_mmd_loss = total_mmd_loss / len(train_loader)
        avg_total_coral_loss = total_coral_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, 总损失: {avg_loss:.4f}",
              f"扩散损失: {avg_target_loss}",
              f"总的迁移学习损失: {avg_mmd_coral_loss}",
              f"mmd部分损失: {avg_total_mmd_loss}",
              f"coral部分损失: {avg_total_coral_loss}")

        # 更新学习率
        scheduler.step()
        print(f"当前学习率: {scheduler.get_last_lr()[0]}")

        # 每 save_interval 个 epoch 保存一次模型
        if (epoch + 1) % save_interval == 0:
            save_dir = f'{RESULT_ROOT}/Islandtown_shuguang/model/DF_model/rgb_sar_{loop}/'
            ddpm.save_network(epoch=epoch, name='ddpm_model', save_dir=save_dir)

        # 验证 ***************************************************************************************************
        # 每十个验证一次
        if (epoch + 1) % 10 == 0:
            print("验证部分 开始验证！！！！！！！！！！！！！！！！！！！！")
            ddpm.netG.eval()
            val_psnr = 0.0
            with torch.no_grad():
                for batch_idx, (img_t1, img_t2, label) in enumerate(val_loader):
                    img_t1 = img_t1.to(device)
                    img_t2 = img_t2.to(device)
                    generated_image = ddpm.test(img_t2, img_t1)

                    psnr_batch = psnr(generated_image.cpu().detach().numpy(), img_t1.cpu().detach().numpy())
                    print(f"当前块psnr值: {psnr_batch:.4f} dB")

                    val_psnr += psnr(generated_image.cpu().detach().numpy(), img_t1.cpu().detach().numpy())

            avg_val_psnr = val_psnr / len(val_loader)
            print(f"验证集平均PSNR值: {avg_val_psnr:.4f}")

            # 保存最佳模型
            if avg_val_psnr > best_psnr:
                best_psnr = avg_val_psnr
                save_dir = f'{RESULT_ROOT}/Islandtown_shuguang/model/DF_model/rgb_sar_{loop}/'
                ddpm.save_network(epoch=False, name='ddpm_model', save_dir=save_dir)
                print(f"目前验证集最高的PSNR平均值为{best_psnr:.4f}, 相应模型已保存.")

    print("扩散部分 开始测试！！！！！！！！！！！！！！！！！！！！")

    # 存储重构的小块
    reconstructed_patches = []
    # 存储原始图像的小块
    original_patches = []
    # 测试
    # 改这里！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    load_dir = f'{RESULT_ROOT}/Islandtown_shuguang/model/DF_model/rgb_sar_{loop}/ddpm_model-best.pt'
    ddpm.netG.eval()
    test_psnr = 0.0
    with torch.no_grad():
        for batch_idx, (img_t1, img_t2, label) in enumerate(test_loader):
            img_t1 = img_t1.to(device)
            img_t2 = img_t2.to(device)
            generated_image = ddpm.test(img_t2, img_t1)
            # test_psnr += psnr(generated_image.cpu().detach().numpy(), img_t1.cpu().detach().numpy())
            # 将重构的图像转换为 PyTorch 张量并存储
            # 计算当前批次的 PSNR
            # max_value = torch.max(img_t1)
            # print(f"Batch {batch_idx + 1} - img_t1 的最大值是: {max_value.item()}")

            psnr_batch = psnr(generated_image.cpu().detach().numpy(), img_t1.cpu().detach().numpy())
            print(f"当前块psnr值: {psnr_batch:.4f} dB")

            reconstructed_patches.extend(generated_image.detach().cpu())

            # 将原始图像转换为 PyTorch 张量并存储
            original_patches.extend(img_t1.detach().cpu())

    # 将重构的小块拼接成原始图像
    reconstructed_img = test_data.reconstruct_image(reconstructed_patches)
    # 将原始图像的小块拼接成原始图像
    original_img = test_data.reconstruct_image(original_patches)

    # 将重构的图像转换为 numpy 数组并进行后处理
    reconstructed_img_np = reconstructed_img.numpy().transpose((1, 2, 0))  # [C, H, W] -> [H, W, C]
    # 将原始图像转换为 numpy 数组并进行后处理
    original_img_np = original_img.numpy().transpose((1, 2, 0))  # [C, H, W] -> [H, W, C]

    # 如果是单通道图像，去掉通道维度
    if reconstructed_img_np.shape[2] == 1:
        reconstructed_img_np = reconstructed_img_np.squeeze(2)

    # 改这里！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # 保存为 .mat 文件
    output_dir = f'{RESULT_ROOT}/Islandtown_shuguang/result/DF_result/rgb_sar_{loop}'
    output_path = os.path.join(output_dir, f"{dataset_name}_SAR_ddpm_{loop}.mat")
    output_path_original = os.path.join(output_dir, f"{dataset_name}_SAR_ddpm_original_{loop}.mat")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # scipy.io.savemat(output_path, {'T1': reconstructed_img_np})
    # scipy.io.savemat(output_path_original, {'T1_original': original_img_np})
    # 改这里！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    scipy.io.savemat(output_path, {'T2': reconstructed_img_np})
    scipy.io.savemat(output_path_original, {'T2_original': original_img_np})

    avg_test_psnr = test_psnr / len(test_loader)
    print(f"Test PSNR: {avg_test_psnr:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training SAR Model")
    parser.add_argument('--loop', type=int, default=0, help='Loop index')
    args = parser.parse_args()
    main(args.loop)