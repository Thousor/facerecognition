import os
import sys
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import logging

# 将父目录添加到sys.path以导入makeup_transfer_tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from makeup_transfer_tf import load_tf1_checkpoint_model, run_tf_inference

# 为此服务器配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# BeautyGAN模型的全局变量
beauty_gan_model_dict = None
BEAUTY_GAN_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))

def initialize_beauty_gan_model():
    """在启动时加载BeautyGAN模型。"""
    global beauty_gan_model_dict
    try:
        logger.info("尝试加载BeautyGAN TF模型...")
        beauty_gan_model_dict = load_tf1_checkpoint_model(BEAUTY_GAN_MODEL_PATH)
        logger.info("BeautyGAN TF模型加载成功。")
    except Exception as e:
        logger.error(f"BeautyGAN模型加载失败: {e}")
        beauty_gan_model_dict = None # 如果加载失败，确保其为None

@app.route('/health', methods=['GET'])
def health_check():
    if beauty_gan_model_dict:
        return jsonify({'status': 'healthy', 'model_loaded': True}), 200
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False, 'message': 'BeautyGAN模型未加载'}), 503

@app.route('/transfer_makeup', methods=['POST'])
def transfer_makeup():
    if beauty_gan_model_dict is None:
        logger.error("妆容迁移模型未加载。")
        return jsonify({'status': 'error', 'message': '妆容迁移模型未加载。'}), 500

    if 'no_makeup_image' not in request.files:
        return jsonify({'status': 'error', 'message': '未提供无妆图像文件。'}), 400
    if 'makeup_style_image' not in request.files:
        return jsonify({'status': 'error', 'message': '未提供妆容风格图像文件。'}), 400

    no_makeup_file = request.files['no_makeup_image']
    makeup_style_file = request.files['makeup_style_image']

    if no_makeup_file.filename == '' or makeup_style_file.filename == '':
        return jsonify({'status': 'error', 'message': '一个或两个文件为空。'}), 400

    # 为上传的文件创建临时目录
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp_uploads')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        no_makeup_path = os.path.join(temp_dir, secure_filename(no_makeup_file.filename))
        makeup_style_path = os.path.join(temp_dir, secure_filename(makeup_style_file.filename))
        
        no_makeup_file.save(no_makeup_path)
        makeup_style_file.save(makeup_style_path)

        logger.info(f"接收文件: {no_makeup_path}, {makeup_style_path}")

        # 使用加载的模型运行推理
        result_np = run_tf_inference(beauty_gan_model_dict, no_makeup_path, makeup_style_path)
        result_img = Image.fromarray(result_np)

        # 将结果保存到BytesIO对象以发送回去
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        logger.info("妆容迁移成功。")
        return img_byte_arr, 200, {'Content-Type': 'image/png'}

    except Exception as e:
        logger.error(f"妆容迁移过程中出错: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    initialize_beauty_gan_model()
    # 在不同的端口运行以避免与主应用程序冲突
    app.run(host='127.0.0.1', port=5001, debug=False)