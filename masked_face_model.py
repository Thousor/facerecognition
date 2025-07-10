import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import (
    Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input
)
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import CosineDecay
import json

CONFIG_FILE = 'config.json'


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


current_config = load_config()

# 配置参数
IMAGE_SIZE = current_config.get('image_size', 128)
BATCH_SIZE = current_config.get('batch_size', 32)
MASKED_MODEL_PATH = "masked_face.keras"


def get_datasets(data_dir, image_size, batch_size):
    """创建训练和验证数据集"""
    # Count total images to determine validation split accurately
    total_images = 0
    for root, _, files in os.walk(data_dir):
        total_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

    train_size = int(total_images * 0.8)
    val_size = total_images - train_size

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb'
    )

    class_names = train_ds.class_names

    # 配置数据集性能
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, train_size, val_size


class StopTrainingCallback(Callback):
    def __init__(self, flag):
        super(StopTrainingCallback, self).__init__()
        self.flag = flag

    def on_epoch_end(self, epoch, logs=None):
        if self.flag and self.flag.is_set():
            self.model.stop_training = True

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(2048, activation='sigmoid') # Adjusted to match x.shape[-1]

    def call(self, x):
        attention = self.global_avg_pool(x)
        attention = self.dense1(attention)
        attention = self.dense2(attention)
        attention = tf.reshape(attention, [-1, 1, 1, x.shape[-1]])
        return tf.multiply(x, attention)

class MaskedFaceModel:
    """口罩人脸识别模型"""

    def __init__(self, num_classes, image_size=IMAGE_SIZE):
        self.model = None
        self.num_classes = num_classes
        self.image_size = image_size
        self.data_augmentation = Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ])

    def build_model(self):
        """构建模型架构"""
        # 使用预训练的ResNet50作为基础模型
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_size, self.image_size, 3)
        )

        # 冻结部分层进行迁移学习
        for layer in base_model.layers[:100]:
            layer.trainable = False

        # 构建完整模型
        inputs = Input(shape=(self.image_size, self.image_size, 3))
        x = self.data_augmentation(inputs)
        x = preprocess_input(x)

        x = base_model(x, training=False)
        x = AttentionLayer(name="attention_layer")(x)

        # 全局平均池化
        x = GlobalAveragePooling2D()(x)

        # 添加全连接层
        x = Dense(1024, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # 输出层
        outputs = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs, outputs)
        self.model.summary()

    def train_model(self, train_ds, val_ds, stop_flag=None, progress_queue=None, train_size=None, val_size=None):
        """训练模型"""
        # 配置优化器和学习率
        total_steps = (train_size // BATCH_SIZE) * 300
        lr_schedule = CosineDecay(initial_learning_rate=0.0001, decay_steps=total_steps)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 配置回调函数
        callbacks = []

        # 早停
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        # 停止训练回调
        if stop_flag is not None:
            callbacks.append(StopTrainingCallback(stop_flag))

        # 训练模型
        print(f"train_size: {train_size}, BATCH_SIZE: {BATCH_SIZE}")
        steps_per_epoch = int(np.ceil(train_size // BATCH_SIZE)) if train_size is not None else None
        validation_steps = int(np.ceil(val_size // BATCH_SIZE)) if val_size is not None else None

        history = self.model.fit(
            train_ds,
            epochs=300,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        return history

    def evaluate(self, dataset, num_samples=None):
        """
        评估模型
        Args:
            dataset: TensorFlow Dataset对象
            num_samples: 数据集中的样本数量
        Returns:
            评估结果（loss和accuracy）
        """
        steps = int(np.ceil(num_samples / BATCH_SIZE)) if num_samples is not None else None
        return self.model.evaluate(dataset, steps=steps)

    def save(self, file_path=MASKED_MODEL_PATH):
        """保存模型"""
        print('正在保存模型...')
        self.model.save(file_path)
        print('模型保存完成！')

    def load(self, file_path=MASKED_MODEL_PATH):
        """加载模型"""
        print('正在加载模型...')
        self.model = load_model(file_path, custom_objects={'AttentionLayer': AttentionLayer})
        print('模型加载完成！')

    def predict(self, img):
        """预测单张图片"""
        # 确保图像是RGB格式
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 预处理图像
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # 使用Keras模型进行预测
        result = self.model.predict(img)

        max_index = np.argmax(result)
        return max_index, result[0][max_index]


def train_and_save_model(data_dir, stop_flag=None, progress_queue=None):
    """训练并保存模型的主函数"""
    try:
        print('正在加载数据集...')
        train_ds, val_ds, class_names, train_size, val_size = get_datasets(data_dir, IMAGE_SIZE, BATCH_SIZE)
        num_classes = len(class_names)
        print(f'检测到 {num_classes} 个类别')

        print('开始构建模型...')
        model = MaskedFaceModel(num_classes)
        model.build_model()

        print('开始训练模型...')
        model.train_model(train_ds, val_ds, stop_flag, progress_queue, train_size, val_size)

        print('正在评估模型...')
        loss, accuracy = model.evaluate(val_ds, val_size)
        print(f'验证集评估结果 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

        model.save()
        print('模型训练完成！')
        return True

    except Exception as e:
        print(f'训练过程中发生错误：{str(e)}')
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='训练口罩人脸识别模型')
    parser.add_argument('--data_dir', type=str, required=True, help='带口罩的人脸数据集目录')
    args = parser.parse_args()

    train_and_save_model(args.data_dir)