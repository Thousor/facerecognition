import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # 禁用TF2.x行为，以兼容TF1.x模型
import numpy as np
from PIL import Image
import os

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
    
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        # 从 .meta 文件加载图结构
        meta_file = os.path.join(model_path, 'model.meta')
        if not os.path.exists(meta_file):
            raise FileNotFoundError("模型元数据文件 (.meta) 不存在于: {}".format(meta_file))
        
        saver = tf.train.import_meta_graph(meta_file)
        # 从checkpoint恢复权重
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        
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

def run_tf_inference(model_dict, no_makeup_img_np, makeup_img_np):
    """
    使用加载的TF1模型Session和张量执行妆容迁移。
    """
    # 1. 获取session和张量
    sess = model_dict["session"]
    X = model_dict["X"]
    Y = model_dict["Y"]
    Xs = model_dict["Xs"]

    # 2. 预处理图像
    X_img = np.expand_dims(preprocess(no_makeup_img_np), 0)
    Y_img = np.expand_dims(preprocess(makeup_img_np), 0)

    # 3. 模型推理
    feed_dict = {X: X_img, Y: Y_img}
    result_tensor = sess.run(Xs, feed_dict=feed_dict)

    # 4. 后处理图像
    result_img_np = deprocess(result_tensor[0])
    
    return result_img_np