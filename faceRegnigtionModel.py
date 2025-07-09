#!/usr/bin/env python
# encoding:utf-8
from __future__ import division

'''
功能： 构建人脸识别模型
'''

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.layers import (
    Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Rescaling,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast, GlobalAveragePooling2D, Input, RandomBrightness
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

# --- Constants from config ---
IMAGE_SIZE = current_config.get('image_size', 128)
BATCH_SIZE = current_config.get('batch_size', 32)
DATA_DIR = current_config.get('data_dir', 'dataset/')
MODEL_FILE_PATH = current_config.get('model_file_path', "face.keras")


def get_datasets(data_dir, image_size, batch_size):
    """
    Creates training and validation datasets from image directories.
    """
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

    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.queue.put({
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy')
        })


class StopTrainingCallback(Callback):
    def __init__(self, flag):
        super(StopTrainingCallback, self).__init__()
        self.flag = flag

    def on_epoch_end(self, epoch, logs=None):
        if self.flag and self.flag.is_set():
            self.model.stop_training = True


class FaceRecognitionModel:
    '''
    人脸识别模型
    '''

    def __init__(self, num_classes, image_size=IMAGE_SIZE):
        self.model = None
        self.num_classes = num_classes
        self.image_size = image_size
        self.data_augmentation = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.2),
            RandomZoom(0.2),
            RandomContrast(0.2),
            RandomBrightness(0.2),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ])

    def build_model(self):
        # 使用ResNet50预训练模型
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_size, self.image_size, 3)
        )

        # 解冻更多层进行微调
        for layer in base_model.layers[:143]:
            layer.trainable = False

        # 构建模型
        inputs = Input(shape=(self.image_size, self.image_size, 3))
        x = self.data_augmentation(inputs)
        
        # Apply ResNet50's specific preprocessing
        x = preprocess_input(x)

        # 预训练特征提取
        x = base_model(x, training=False)

        # 全局平均池化替代Flatten
        x = GlobalAveragePooling2D()(x)

        # 添加分类层
        x = Dense(2048, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.6)(x)

        x = Dense(1024, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.6)(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs, outputs)
        self.model.summary()

    def train_model(self, train_ds, val_ds, stop_flag=None, progress_queue=None):
        # 使用CosineDecay学习率调度器
        total_steps = len(train_ds) * 300  # epochs
        lr_schedule = CosineDecay(initial_learning_rate=0.0001, decay_steps=total_steps)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 改进的回调函数
        callbacks = []
        
        # 添加早停回调
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        # 添加停止训练回调（如果提供了stop_flag）
        if stop_flag is not None:
            callbacks.append(StopTrainingCallback(stop_flag))

        # 添加进度回调（如果提供了progress_queue）
        if progress_queue is not None:
            callbacks.append(TrainingProgressCallback(progress_queue))

        # 训练模型
        history = self.model.fit(
            train_ds,
            epochs=300,
            validation_data=val_ds,
            callbacks=callbacks
        )

        return history

    def evaluate(self, dataset):
        """
        评估模型性能
        Args:
            dataset: TensorFlow Dataset对象
        Returns:
            评估结果（loss和accuracy）
        """
        return self.model.evaluate(dataset)

    def save(self, file_path=MODEL_FILE_PATH):
        """
        保存模型
        Args:
            file_path: 模型保存路径
        """
        print('正在保存模型...')
        self.model.save(file_path)
        print('模型保存完成！')

    def load(self, file_path=MODEL_FILE_PATH):
        """
        加载模型
        Args:
            file_path: 模型文件路径
        """
        print('正在加载模型...')
        self.model = load_model(file_path)
        print('模型加载完成！')

    def predict(self, img):
        """
        预测单张图片
        Args:
            img: 输入图片（numpy数组）
        Returns:
            (预测的类别索引, 预测的概率)
        """
        # 确保图像是RGB格式
        if len(img.shape) == 2:  # 如果是灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.expand_dims(img, axis=0)  # 添加批次维度
        
        # 应用预处理
        img = preprocess_input(img)

        result = self.model.predict(img)
        max_index = np.argmax(result)
        return max_index, result[0][max_index]


def train_and_save_model(stop_flag=None, progress_queue=None):
    """
    训练并保存模型
    Args:
        stop_flag: 停止训练的标志
        progress_queue: 进度队列，用于报告训练进度
    """
    try:
        # 获取数据集
        print('正在加载数据集...')
        train_ds, val_ds, class_names = get_datasets(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
        num_classes = len(class_names)
        print(f'检测到 {num_classes} 个类别')

        # 创建并训练模型
        print('开始构建模型...')
        face_model = FaceRecognitionModel(num_classes)
        face_model.build_model()
        
        print('开始训练模型...')
        face_model.train_model(train_ds, val_ds, stop_flag, progress_queue)
        
        # 评估模型
        print('正在评估模型...')
        loss, accuracy = face_model.evaluate(val_ds)
        print(f'验证集评估结果 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # 保存模型
        face_model.save()
        
        print('模型训练完成！')
        return True
        
    except Exception as e:
        print(f'训练过程中发生错误：{str(e)}')
        return False


if __name__ == '__main__':
    train_and_save_model()
