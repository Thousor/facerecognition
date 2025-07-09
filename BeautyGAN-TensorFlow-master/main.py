import tensorflow as tf
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='妆容迁移应用')
    parser.add_argument('--no_makeup', type=str, help='无妆容图像路径')
    parser.add_argument('--makeup_dir', type=str, help='妆容图像目录或单个文件路径')
    parser.add_argument('--output', type=str, help='输出结果图像路径')
    # 以下参数在当前调用方式下不是必需的，但保留以兼容原始脚本
    parser.add_argument('--model_path', type=str, default=os.path.join('model'), help='模型保存目录')
    parser.add_argument('--img_size', type=int, default=256, help='图像尺寸')
    return parser.parse_args()

def preprocess(img):
    """图像预处理：归一化到[-1, 1]范围"""
    return (img.astype(np.float32) / 255. - 0.5) * 2

def deprocess(img):
    """图像后处理：从[-1, 1]范围恢复到[0, 1]"""
    img = (img + 1) / 2
    return np.clip(img, 0, 1)

def load_image(path, size):
    """加载并调整图像大小"""
    try:
        # 使用 imageio.v2.imread 避免 DeprecationWarning
        import imageio.v2 as imageio
        img = imageio.imread(path)
        # 检查图像是否为空
        if img is None:
            print("警告: 无法读取图像 {}".format(path))
            return None
        # 转换灰度图为RGB
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        # 调整大小并确保是3通道
        img = cv2.resize(img, (size, size))
        if img.shape[2] == 4:
            img = img[..., :3] # 去除alpha通道
        return img
    except Exception as e:
        print("无法加载图像 {}: {}".format(path, e))
        return None

def main():
    args = parse_args()

    # 加载无妆容图像
    no_makeup_img = load_image(args.no_makeup, args.img_size)
    if no_makeup_img is None:
        print("无法继续：未成功加载无妆容图像")
        return

    X_img = np.expand_dims(preprocess(no_makeup_img), 0)

    # 检查妆容路径是文件还是目录
    if os.path.isdir(args.makeup_dir):
        makeup_images = glob.glob(os.path.join(args.makeup_dir, '*.*'))
    elif os.path.isfile(args.makeup_dir):
        makeup_images = [args.makeup_dir]
    else:
        makeup_images = []

    if not makeup_images:
        print("在 {} 中未找到妆容图像".format(args.makeup_dir))
        return

    print("找到 {} 个妆容图像".format(len(makeup_images)))

    # 加载模型
    with tf.Session() as sess:
        try:
            # 从 .meta 文件加载图结构并恢复权重
            saver = tf.train.import_meta_graph(os.path.join(args.model_path, 'model.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(args.model_path))
            graph = tf.get_default_graph()
            print("成功加载TensorFlow模型")
        except Exception as e:
            print("加载TensorFlow模型失败: {}".format(e))
            return

        # 从图中获取输入和输出张量
        # 这些名称需要与模型保存时的名称完全一致
        try:
            X = graph.get_tensor_by_name("X:0") # 无妆输入
            Y = graph.get_tensor_by_name("Y:0") # 带妆输入
            Xs = graph.get_tensor_by_name("generator/xs:0") # 输出
        except KeyError as e:
            print("无法从图中找到必要的张量: {}".format(e))
            print("请检查模型结构，确保输入输出张量的名称正确。")
            return

        # 只处理找到的第一个妆容图像
        makeup_path = makeup_images[0]
        makeup_img = load_image(makeup_path, args.img_size)
        if makeup_img is None:
            print("无法处理妆容图像 {}".format(makeup_path))
            return

        Y_img = np.expand_dims(preprocess(makeup_img), 0)

        # 应用妆容迁移
        try:
            Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
            result_img = deprocess(Xs_[0])
        except Exception as e:
            print("模型推理失败: {}".format(e))
            return

    # 保存结果图像
    try:
        # 将 [0, 1] 范围的浮点图像转换为 [0, 255] 的uint8图像
        final_image = (result_img * 255).astype(np.uint8)
        # imageio 保存时会自动处理RGB顺序
        imsave(args.output, final_image)
        print("结果已保存至: {}".format(args.output))
    except Exception as e:
        print("保存图像失败: {}".format(e))

if __name__ == "__main__":
    main()
