import argparse

from DF_main_Islandtown_shuguang_rgb import main as run_rgb
from DF_main_Islandtown_shuguang_sar import main as run_sar
from CD_train_Islandtown_shuguang import main as run_train
from CD_test_shuguang import run_test as run_test_shuguang

def main():
    # 初始化参数解析器并设置描述
    parser = argparse.ArgumentParser(description="Main script for running the entire pipeline")

    # 添加 start_loop 参数
    parser.add_argument('--start_loop', type=int, default=5, help='The loop index to start from (default: 0)')

    # 解析参数
    args = parser.parse_args()
    num_loops = 6
    start_loop = args.start_loop

    for loop in range(start_loop, num_loops):
        print(f"\n开始第 {loop} 轮循环...\n")

        # 运行 DF_main_Islandtown_shuguang_rgb.py
        print("运行 DF_main_Islandtown_shuguang_rgb.py")
        run_rgb(loop)
        
        # 运行 DF_main_Islandtown_shuguang_sar.py
        print("运行 DF_main_Islandtown_shuguang_sar.py")
        run_sar(loop)

        # 运行 CD_train_shuguang-Islandtown.py，并传递 train_val_ratio 参数
        print("运行 CD_train_Islandtown_shuguang.py")
        run_train(loop)

        # 运行 CD_test_shuguang.py
        print("运行 CD_test_shuguang.py")
        run_test_shuguang(loop)

if __name__ == "__main__":
    main()