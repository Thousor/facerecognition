import tensorflow as tf
import numpy as np
from PIL import Image
import os
import argparse # 导入 argparse

def preprocess(img, img_size=256):
    """图像预处理：调整大小并归一化到[-1, 1]范围"""
    # 使用PIL确保图像格式和大小正确
    img_pil = Image.fromarray(img).resize((img_size, img_size), Image.BICUBIC)
    img_np = np.array(img_pil, dtype=np.float32)
    return (img_np / 127.5) - 1.0

def deprocess(img):
    """图像后处理：从[-1, 1]范围恢复到[0, 255]的uint8"""
    img = (img + 1.0) * 127.5
    return np.clip(img, 0, 255).astype(np.uint8)

def load_tf1_checkpoint_model(model_path):
    """
    使用 TF1.x 兼容模式加载 Checkpoint 模型。
    返回一个包含 session 和输入/输出张量的字典。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("模型路径不存在: {}".format(model_path))

    print("正在以TF1兼容模式加载Checkpoint模型: {}".format(model_path))
    
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session()
        # 从 .meta 文件加载图结构
        meta_file = os.path.join(model_path, 'model.meta')
        if not os.path.exists(meta_file):
            raise FileNotFoundError("模型元数据文件 (.meta) 不存在于: {}".format(meta_file))
        
        saver = tf.compat.v1.train.import_meta_graph(meta_file)
        # 从checkpoint恢复权重
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint(model_path))
        
        # 从图中获取输入和输出张量
        try:
            X = graph.get_tensor_by_name("X:0")
            Y = graph.get_tensor_by_name("Y:0")
            Xs = graph.get_tensor_by_name("generator/xs:0")
        except KeyError as e:
            raise ValueError("无法从图中找到必要的张量: {}. 请确认模型结构是否正确。".format(e))

    print("TF1 Checkpoint模型加载成功。")
    
    return {
        "session": sess,
        "X": X,
        "Y": Y,
        "Xs": Xs
    }

def run_tf_inference(model_dict, no_makeup_img_path, makeup_img_path):
    """
    使用加载的TF1模型Session和张量执行妆容迁移。
    """
    # 1. 获取session和张量
    sess = model_dict["session"]
    X = model_dict["X"]
    Y = model_dict["Y"]
    Xs = model_dict["Xs"]

    # 加载图像为numpy数组
    no_makeup_img = Image.open(no_makeup_img_path).convert('RGB')
    makeup_img = Image.open(makeup_img_path).convert('RGB')
    no_makeup_np = np.array(no_makeup_img)
    makeup_np = np.array(makeup_img)

    # 2. 预处理图像
    X_img = np.expand_dims(preprocess(no_makeup_np), 0)
    Y_img = np.expand_dims(preprocess(makeup_np), 0)

    # 3. 模型推理
    feed_dict = {X: X_img, Y: Y_img}
    result_tensor = sess.run(Xs, feed_dict=feed_dict)

    # 4. 后处理图像
    result_img_np = deprocess(result_tensor[0])
    
    return result_img_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用BeautyGAN-TensorFlow进行妆容迁移')
    parser.add_argument('--no_makeup', type=str, required=True, help='无妆图像的路径。')
    parser.add_argument('--makeup_style', type=str, required=True, help='妆容风格图像的路径。')
    parser.add_argument('--model_path', type=str, required=True, help='BeautyGAN模型检查点目录的路径。')
    parser.add_argument('--output', type=str, required=True, help='保存输出图像的路径。')

    args = parser.parse_args()

    # 加载模型
    model_dict = load_tf1_checkpoint_model(args.model_path)

    # 运行推理
    result_image_np = run_tf_inference(model_dict, args.no_makeup, args.makeup_style)

    # 保存结果
    result_image_pil = Image.fromarray(result_image_np)
    result_image_pil.save(args.output)

    print(f"妆容迁移完成，结果保存到: {args.output}")