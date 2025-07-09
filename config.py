import os
import json

# Configuration file path
CONFIG_FILE = 'config.json'

# Default configuration values
DEFAULT_CONFIG = {
    "threshold": 0.5,
    "num_images_to_collect": 50,
    "image_size": 128,
    "batch_size": 32,
    "data_dir": "dataset/",  # 修改为正确的数据目录
    "model_file_path": "face.keras",
    "masked_model_file_path": "masked_face.keras",
    "recognition_log_file": "recognition_log.txt",
    "collection_fps": 1,
    "DATA_PATH": "dataset/",  # 添加数据路径
    "RECOGNITION_LOG_FILE": "recognition_log.txt",  # 添加日志文件路径
    "FACE_CASCADE_FILE": "config/haarcascade_frontalface_alt.xml",  # 添加人脸检测器路径
    "MASKED_DATA_DIR": "mask_dataset/",  # 添加口罩数据目录
    "BEAUTY_UPLOAD_FOLDER": "beauty/uploads/",  # 添加美颜上传目录
    "BEAUTY_OUTPUT_FOLDER": "static/beauty_processed/",  # 添加美颜输出目录
    "SESSION_TIMEOUT": 1800,  # 会话超时时间（秒）
    "MIN_FACE_SIZE": 64,  # 最小人脸尺寸
    "MAX_FACE_SIZE": 256,  # 最大人脸尺寸
    "FACE_DETECTION_SCALE": 1.3,  # 人脸检测缩放因子
    "FACE_DETECTION_NEIGHBORS": 5,  # 人脸检测最小邻居数
    "CONFIDENCE_THRESHOLD": 0.6  # 人脸识别置信度阈值
}

class Config:
    def __init__(self):
        self._config = {}
        self.load_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults to ensure all keys are present
                self._config = {**DEFAULT_CONFIG, **loaded_config}
        else:
            self._config = DEFAULT_CONFIG.copy()
            self.save_config() # Save default config if file doesn't exist

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self._config, f, indent=4)

    def get(self, key, default=None):
        return self._config.get(key, default)

    def __getattr__(self, name):
        # Allow accessing config values as attributes (e.g., config.THRESHOLD)
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Allow setting config values as attributes (e.g., config.THRESHOLD = 0.6)
        if name == '_config':
            super().__setattr__(name, value)
        else:
            self._config[name] = value
            self.save_config() # Save changes immediately

# Instantiate the Config class to make it a singleton-like object
config = Config()
