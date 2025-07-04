#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import sys
import json

CONFIG_FILE = 'config.json'


def load_config():
    """加载配置文件，返回配置字典"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # print(f"   配置文件 {CONFIG_FILE} 格式错误，将使用默认配置") # Commented out
            return {}
    # print(f"   未找到配置文件 {CONFIG_FILE}，将使用默认配置") # Commented out
    return {}


current_config = load_config()


def collect_faces(name, num_images=None):
    """
    通过摄像头采集指定数量的人脸图片
    :param name: 采集对象的姓名（用于创建保存文件夹）
    :param num_images: 采集数量（None则从配置读取，默认50）
    """
    # 确定采集数量
    if num_images is None:
        num_images = current_config.get('num_images_to_collect', 50)
    try:
        num_images = int(num_images)
        if num_images <= 0:
            # print("  采集数量必须为正整数") # Commented out
            return
    except ValueError:
        # print("  采集数量必须为整数") # Commented out
        return

    # 创建保存目录
    output_folder = os.path.join('data', name)
    try:
        os.makedirs(output_folder, exist_ok=True)  # 已存在则不报错
        # print(f"  图片将保存至：{output_folder}") # Commented out
    except OSError as e:
        # print(f"  无法创建保存目录 {output_folder}：{str(e)}（可能是权限不足）") # Commented out
        return

    # 检查人脸检测模型
    cascade_path = 'config/haarcascade_frontalface_alt.xml'
    if not os.path.exists(cascade_path):
        # print(f"  人脸检测模型文件不存在：{cascade_path}") # Commented out
        # print("   请确认config目录下是否有haarcascade_frontalface_alt.xml文件") # Commented out
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        # print(f"  无法加载人脸检测模型：{cascade_path}（文件可能损坏）") # Commented out
        return

    # 初始化摄像头
    camera = cv2.VideoCapture(0)  # 默认摄像头（多摄像头可修改索引）
    if not camera.isOpened():
        # print("   无法打开摄像头！可能原因：") # Commented out
        # print("   1. 摄像头被其他程序占用") # Commented out
        # print("   2. 摄像头未正确连接（USB设备请检查接口）") # Commented out
        # print("   3. 无可用摄像头设备") # Commented out
        return

    # 启动提示
    # print("\n📸 开始采集人脸图片（按 'q' 或 'Esc' 键退出）") # Commented out
    # print(f"   需采集 {num_images} 张，仅保存清晰且尺寸足够的人脸（>100x100像素）") # Commented out

    # 采集参数
    count = 0  # 已保存图片数量
    frame_counter = 0  # 帧计数器
    frame_skip = 5  # 每隔5帧处理一次（减少重复）

    while count < num_images:
        # 读取摄像头帧
        success, frame = camera.read()
        if not success:
            # print(" 无法获取摄像头画面（可能摄像头已断开）") # Commented out
            break

        frame_counter += 1
        # 每隔frame_skip帧才处理（提高效率）
        if frame_counter % frame_skip != 0:
            # 实时显示进度（不处理但显示画面）
            progress_text = f"已采集：{count}/{num_images} 张（按q退出）"
            cv2.putText(frame, progress_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("人脸采集（请面对摄像头）", frame)

            # 检查退出按键
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:  # 'q'或Esc
                # print("\n 用户手动退出采集") # Commented out
                break
            continue

        # 检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图（提高检测效率）
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            # 选择最大的人脸（减少多脸干扰）
            (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])

            # 绘制人脸框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "检测到人脸", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 提取人脸区域
            face_roi = frame[y:y + h, x:x + w]

            # 过滤小尺寸人脸（保证质量）
            if w > 100 and h > 100:
                img_name = f"{count + 1}.jpg"
                img_path = os.path.join(output_folder, img_name)
                # 保存图片并检查结果
                if cv2.imwrite(img_path, face_roi):
                    # print(f" 已保存 {count + 1}/{num_images}：{img_name}") # Commented out
                    count += 1
                else:
                    # print(f" 保存失败：{img_path}（可能是权限问题）") # Commented out
                    pass # Suppress error for now

        # 显示实时画面和进度
        progress_text = f"已采集：{count}/{num_images} 张（按q退出）"
        cv2.putText(frame, progress_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("人脸采集（请面对摄像头）", frame)

        # 检查退出按键
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            # print("\n 用户手动退出采集") # Commented out
            break

    # 清理资源
    camera.release()
    cv2.destroyAllWindows()

    # 最终反馈
    if count == num_images:
        # print(f"\n🎉 采集完成！共成功保存 {count} 张图片至 {output_folder}") # Commented out
        pass # Suppress message for now
    else:
        # print(f"\n🔚 采集终止。已保存 {count}/{num_images} 张图片至 {output_folder}") # Commented out
        pass # Suppress message for now


def process_and_save_face(image, user_name, source='cam'):
    """
    从给定图像中检测、处理并保存人脸。
    :param image: 输入的图像 (OpenCV a BGR numpy.ndarray).
    :param user_name: 用户名，用于创建文件夹。
    :param source: 图像来源标识 (e.g., 'cam' or 'upload')，用于命名。
    :return: (bool, str) 元组，表示成功状态和消息。
    """
    output_folder = os.path.join('data', user_name)
    os.makedirs(output_folder, exist_ok=True)

    cascade_path = 'config/haarcascade_frontalface_alt.xml'
    if not os.path.exists(cascade_path):
        print(f"人脸检测模型文件不存在：{cascade_path}")
        return False, f"人脸检测模型文件不存在：{cascade_path}"

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"无法加载人脸检测模型：{cascade_path}")
        return False, f"无法加载人脸检测模型：{cascade_path}"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("在上传的图片中未检测到清晰的人脸。")
        return False, "在上传的图片中未检测到清晰的人脸。"

    # 选择最大的人脸
    (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])

    # 过滤小尺寸人脸
    if w < 100 or h < 100:
        print(f"检测到的人脸尺寸过小 ({w}x{h})，请使用更高分辨率或更近距离的照片。")
        return False, f"检测到的人脸尺寸过小 ({w}x{h})，请使用更高分辨率或更近距离的照片。"

    face_roi = image[y:y + h, x:x + w]

    # 查找下一个可用的文件名
    count = 0
    while True:
        img_name = f"{source}_{count}.jpg"
        img_path = os.path.join(output_folder, img_name)
        if not os.path.exists(img_path):
            break
        count += 1

    if cv2.imwrite(img_path, face_roi):
        print(f"人脸已成功保存为：{img_path}")
        return True, f"人脸已成功保存为：{img_path}"
    else:
        print(f"保存图片失败，请检查目录权限：{output_folder}")
        return False, f"保存图片失败，请检查目录权限：{output_folder}"