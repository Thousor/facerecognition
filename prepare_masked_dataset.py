import os
import argparse
from wearmask import process_dataset
import shutil


def prepare_masked_dataset(input_dir='data', output_dir='mask_dataset', temp_dir='temp_masked'):
    """
    准备带口罩的人脸数据集
    Args:
        input_dir: 原始人脸数据集目录，默认为'data'
        output_dir: 最终数据集输出目录，默认为'mask_dataset'
        temp_dir: 临时目录，用于存储中间结果
    """
    try:
        print("开始处理数据集...")

        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            raise ValueError(f"输入目录 {input_dir} 不存在！")

        # 清理临时目录（如果存在）
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # 清理输出目录（如果存在）
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # 创建临时目录和输出目录
        os.makedirs(temp_dir)
        os.makedirs(output_dir)

        # 第一步：为每个人脸添加口罩
        print("正在为人脸添加口罩...")
        process_dataset(input_dir, temp_dir)

        # 第二步：整理数据集结构
        print("正在整理数据集结构...")

        # 遍历临时目录中的所有文件
        for root, dirs, files in os.walk(temp_dir):
            for dir_name in dirs:
                # 获取相对路径
                rel_path = os.path.relpath(os.path.join(root, dir_name), temp_dir)

                # 创建对应的输出目录
                out_dir = os.path.join(output_dir, rel_path)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                # 复制所有图片到输出目录
                src_dir = os.path.join(root, dir_name)
                for img in os.listdir(src_dir):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src_path = os.path.join(src_dir, img)
                        dst_path = os.path.join(out_dir, img)
                        shutil.copy2(src_path, dst_path)

        print("清理临时文件...")
        shutil.rmtree(temp_dir)

        print(f"数据集处理完成！结果保存在: {output_dir}")
        return True

    except Exception as e:
        print(f"处理数据集时发生错误: {str(e)}")
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备带口罩的人脸识别数据集')
    parser.add_argument('--input_dir', type=str, default='data', help='输入数据集目录')
    parser.add_argument('--output_dir', type=str, default='mask_dataset', help='输出数据集目录')
    args = parser.parse_args()

    prepare_masked_dataset(args.input_dir, args.output_dir)