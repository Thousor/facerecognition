import os
import argparse
from wearmask import process_dataset
import shutil


def prepare_masked_dataset(input_dir='data', output_dir='mask_dataset'):
    """
    准备带口罩的人脸数据集，直接将结果写入输出目录。
    Args:
        input_dir: 原始人脸数据集目录，默认为'data'
        output_dir: 最终数据集输出目录，默认为'mask_dataset'
    """
    try:
        print("开始准备口罩数据集...")

        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            raise ValueError(f"输入目录 {input_dir} 不存在！")

        # 清理并重建输出目录
        if os.path.exists(output_dir):
            print(f"正在清理旧的口罩数据集: {output_dir}")
            shutil.rmtree(output_dir)
        
        print(f"正在创建空的输出目录: {output_dir}")
        os.makedirs(output_dir)

        # 直接处理数据集，将结果存入输出目录
        print(f"正在从 {input_dir} 生成口罩图片到 {output_dir}...")
        process_dataset(input_dir, output_dir)

        print(f"口罩数据集处理完成！结果保存在: {output_dir}")
        return True

    except Exception as e:
        print(f"处理数据集时发生错误: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备带口罩的人脸识别数据集')
    parser.add_argument('--input_dir', type=str, default='data', help='输入数据集目录')
    parser.add_argument('--output_dir', type=str, default='mask_dataset', help='输出数据集目录')
    args = parser.parse_args()

    prepare_masked_dataset(args.input_dir, args.output_dir)
