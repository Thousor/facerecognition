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
from keras.applications.resnet50 import preprocess_input # Import preprocess_input
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

class FaceRecognitionModel(object):  # 重命名自定义类
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
        for layer in base_model.layers[:143]: # Unfreeze more layers
            layer.trainable = False

        # 构建模型
        inputs = Input(shape=(self.image_size, self.image_size, 3))
        x = self.data_augmentation(inputs)
        
        # Apply ResNet50's specific preprocessing
        x = preprocess_input(x) # Changed: Use preprocess_input

        # 预训练特征提取
        x = base_model(x, training=False)

        # 全局平均池化替代Flatten
        x = GlobalAveragePooling2D()(x)

        # 添加分类层
        x = Dense(2048, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.6)(x) # Increase dropout

        x = Dense(1024, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.6)(x) # Increase dropout

        outputs = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs, outputs)
        self.model.summary()

    def train_model(self, train_ds, val_ds, stop_flag, progress_queue=None):
        # 使用CosineDecay学习率调度器
        total_steps = len(train_ds) * 300 # epochs
        lr_schedule = CosineDecay(initial_learning_rate=0.0001, decay_steps=total_steps)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 改进的回调函数
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=25, # Increase patience
            verbose=1,
            restore_best_weights=True
        )

        stop_training_callback = StopTrainingCallback(stop_flag)
        callbacks = [early_stopping, stop_training_callback]
        if progress_queue:
            callbacks.append(TrainingProgressCallback(progress_queue))

        # 训练模型
        self.model.fit(
            train_ds,
            epochs=300,  # 增加最大训练轮次
            validation_data=val_ds,
            callbacks=callbacks
        )

    def evaluate(self, dataset):
        x_test, y_test = dataset.get_test_data()
        return self.model.evaluate(x_test, y_test)


class StopTrainingCallback(Callback):
    def __init__(self, flag):
        super(StopTrainingCallback, self).__init__()
        self.flag = flag

    def on_epoch_end(self, epoch, logs=None):
        if self.flag.is_set():
            self.model.stop_training = True

    def save(self, file_path=MODEL_FILE_PATH):
        print('Model Saved Finished!!!')
        self.model.save(file_path)

    def load(self, file_path=MODEL_FILE_PATH):
        print('Model Loaded Successful!!!')
        self.model = load_model(file_path)

    def predict(self, img):
        # 确保图像是RGB格式
        if len(img.shape) == 2:  # 如果是灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.expand_dims(img, axis=0)  # 添加批次维度
        
        # Apply ResNet50's specific preprocessing for prediction as well
        img = preprocess_input(img) # Changed: Apply preprocess_input here too

        result = self.model.predict(img)
        max_index = np.argmax(result)
        return max_index, result[0][max_index]

def train_and_save_model(progress_queue=None):
    # 获取数据集
    train_ds, val_ds, class_names = get_datasets(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    num_classes = len(class_names)

    # 创建并训练模型
    face_model = FaceRecognitionModel(num_classes)
    face_model.build_model()
    face_model.train_model(train_ds, val_ds, progress_queue)
    face_model.evaluate_model(val_ds)
    face_model.save()

if __name__ == '__main__':
    train_and_save_model()
