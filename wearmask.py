import os
import sys
import argparse
import numpy as np
import cv2
import math
import dlib
from PIL import Image, ImageFile
import face_recognition

__version__ = '0.3.0'

# 修改默认路径
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, 'blue-mask.png')


def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)


def face_alignment(faces):
    """人脸对齐处理"""
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # 关键点：左眼、右眼、鼻子、左嘴角、右嘴角
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y

        # 计算两眼的中心坐标
        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,
                      (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)

        # 计算角度
        angle = math.atan2(dy, dx) * 180. / math.pi

        # 计算仿射矩阵
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

        # 进行仿射变换，即旋转
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned


class FaceMasker:
    """为人脸添加口罩的类"""
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path=DEFAULT_IMAGE_PATH, show=False, model='hog', save_path=''):
        self.face_path = face_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None

    def mask(self):
        """添加口罩的主要处理函数"""
        # 加载图片并检测人脸
        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)

        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # 检查是否包含所需的面部特征
            if all(feature in face_landmark for feature in self.KEY_FACIAL_FEATURES):
                found_face = True
                self._mask_face(face_landmark)

        if found_face:
            # 对齐处理
            src_faces = []
            with_mask_face = np.asarray(self._face_img)

            for rect in face_locations:
                (x, y, w, h) = rect_to_bbox(rect)
                detect_face = with_mask_face[y:y + h, x:x + w]
                src_faces.append(detect_face)

            # 人脸对齐并保存
            faces_aligned = face_alignment(src_faces)

            for face_idx, face in enumerate(faces_aligned):
                face = cv2.cvtColor(face, cv2.COLOR_RGBA2BGR)
                face_resized = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)

                # 如果没有指定保存路径，则使用默认命名方式
                if not self.save_path:
                    path_splits = os.path.splitext(self.face_path)
                    save_path = f"{path_splits[0]}_masked_{face_idx}{path_splits[1]}"
                else:
                    save_path = self.save_path

                cv2.imwrite(save_path, face_resized)
                print(f'已保存带口罩的人脸图片到：{save_path}')
        else:
            print(f'未检测到人脸：{self.face_path}')

    def _mask_face(self, face_landmark: dict):
        """在检测到的人脸上添加口罩"""
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # 分割口罩图片并调整大小
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.5  # 增加宽度比例
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v) * 1.2)  # 增加高度

        # 处理左半部分
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(
            chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # 处理右半部分
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(
            chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # 合并口罩
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # 旋转口罩
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1],
                           chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # 计算口罩位置
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # 添加口罩
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        """计算点到直线的距离"""
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


def process_dataset(input_dir, output_dir):
    """处理整个数据集的函数"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录
    for root, dirs, files in os.walk(input_dir):
        # 获取当前处理的子目录相对于input_dir的路径
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)

        # 如果目标子目录已经存在，则跳过整个子目录的处理
        if os.path.exists(output_subdir) and len(os.listdir(output_subdir)) > 0:
            print(f"跳过目录 {root}：目标口罩目录 {output_subdir} 已存在且不为空。")
            continue

        # 创建输出子目录
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 构建输入和输出路径
                input_path = os.path.join(root, name)

                # 构建输出文件路径
                output_path = os.path.join(output_subdir, f"masked_{name}")

                try:
                    # 处理图片
                    masker = FaceMasker(input_path, save_path=output_path)
                    masker.mask()
                except Exception as e:
                    print(f"处理图片 {input_path} 时出错: {str(e)}")


if __name__ == '__main__':
    # a dictionary for mapping class names to integers
    class_names = []
    # a list of all file paths
    file_paths = []
    # a list of all file paths
    masked_file_paths = []

    # 遍历data文件夹
    for i, person in enumerate(os.listdir("data")):
        # if person is not a directory, skip it
        if not os.path.isdir(os.path.join("data", person)):
            continue
        
        masked_face_dir = os.path.join("mask_dataset", person)
        if os.path.exists(masked_face_dir):
            print(f"Directory for {person} already exists. Skipping.")
            continue
        else:
            os.makedirs(masked_face_dir)
            
        # add the class name to the list
        class_names.append(person)
        # iterate over the images in the person's directory
        for file in os.listdir(os.path.join("data", person)):
            # if the file is not a jpg or png file, skip it
            if not file.endswith(".jpg") and not file.endswith(".png"):
                continue
            # add the file path to the list
            file_paths.append(os.path.join("data", person, file))
            # wear mask
            masker = FaceMasker(os.path.join("data", person, file), save_path=os.path.join(masked_face_dir, f"masked_{file}"))
            masker.mask()