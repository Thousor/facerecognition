#!/usr/bin/env python
# encoding:utf-8
from __future__ import division
import os
import sys
import time
import queue
import numpy as np
import subprocess
import shutil
import json  # Added for config management
import threading  # Added for training thread
from datetime import datetime  # Added for logging timestamp
# Updated Flask imports
import threading
import time
import cv2
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, send_file, make_response
from werkzeug.utils import secure_filename
from beauty_processor import BeautyProcessor
import io
import logging
from queue import Queue
from PIL import Image

from collect_data import collect_faces, process_and_save_face
from dataHelper import read_name_list
from faceRegnigtionModel import train_and_save_model, FaceRecognitionModel, get_datasets, IMAGE_SIZE, MODEL_FILE_PATH
from prepare_masked_dataset import prepare_masked_dataset
from masked_face_model import train_and_save_model as train_masked_model, MaskedFaceModel



BEAUTY_GAN_MODEL_PATH = 'D:/PythonProject/face-recognition-001/model'



# --- Global variables for masked model training ---
masked_training_thread = None
masked_stop_training_flag = threading.Event()
masked_training_in_progress = False
masked_model = None
last_beauty_frame = None # Global to store the last processed frame for capture



# --- Global variables for training control ---
training_thread = None
stop_training_flag = threading.Event()
training_in_progress = False
# Imports from the model and camera demo
from faceRegnigtionModel import FaceRecognitionModel, TrainingProgressCallback, get_datasets, IMAGE_SIZE, BATCH_SIZE,     DATA_DIR
from dataHelper import read_name_list

CONFIG_FILE = 'config.json'
RECOGNITION_LOG_FILE = 'recognition_log.txt'  # New: Log file for recognition events
current_config = {}


def load_config():
    global current_config
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            current_config = json.load(f)
    else:
        # Default configuration if file doesn't exist
        current_config = {
            "threshold": 0.5,  # Changed default threshold
            "num_images_to_collect": 50,
            "image_size": 128,
            "batch_size": 32,
            "data_dir": "dataset/",
            "model_file_path": "face.keras",
            "collection_fps": 1  # Added default for collection_fps
        }
        save_config(current_config)


def save_config(config_data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)


def log_recognition_event(name, confidence):
    """Logs a recognition event to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}, {name}, {confidence:.2f}\n"
    with open(RECOGNITION_LOG_FILE, 'a') as f:
        f.write(log_entry)


# Load config at application start
load_config()

app = Flask(__name__, static_url_path='/static', static_folder='static')

# --- Globals ---
# Camera is now managed within each generator function, not globally.
face_recognition_model = None
masked_face_model = None
name_list_camera = []
masked_name_list = []
img_size = 128  # Will be updated from config

# 初始化美颜处理器
beauty_processor = BeautyProcessor()
beauty_options = {
    'smooth': False,
    'whiten': False,
    'slim': False,
    'acne': False,
    'smooth_level': 0.7,
    'whiten_level': 0.3,
    'slim_level': 0.2
}


def initialize_globals():
    """Initializes recognition models and name list."""
    global face_recognition_model, name_list_camera, img_size, masked_face_model, masked_name_list
    try:
        # Update globals from config
        img_size = current_config.get('image_size', 128)

        # Load standard model
        name_list_camera = read_name_list(current_config['data_dir'])
        if len(name_list_camera) > 0 and os.path.exists(MODEL_FILE_PATH):
            face_recognition_model = FaceRecognitionModel(num_classes=len(name_list_camera))
            face_recognition_model.load()
        else:
            face_recognition_model = None

        # Load masked face model
        masked_name_list = read_name_list('mask_dataset/')
        if len(masked_name_list) > 0 and os.path.exists("masked_face.keras"):
            masked_face_model = MaskedFaceModel(num_classes=len(masked_name_list))
            masked_face_model.load()
        else:
            masked_face_model = None

    except Exception as e:
        print(f"Error during model initialization: {e}")
        face_recognition_model = None
        masked_face_model = None


def gen_frames():
    print("Starting gen_frames...")
    """Generator function for video streaming. Manages its own camera instance."""
    global face_recognition_model, name_list_camera

    if not all([face_recognition_model, name_list_camera]):
        print("Streaming stopped: model not ready.")
        return

    try:
        print("正在尝试打开摄像头...")
        camera = cv2.VideoCapture(0)

        # 尝试多个摄像头索引
        camera_index = 0
        while not camera.isOpened() and camera_index < 3:
            print(f"尝试打开摄像头 {camera_index} 失败，尝试下一个...")
            camera.release()
            camera_index += 1
            camera = cv2.VideoCapture(camera_index)

        if not camera.isOpened():
            print("错误：无法打开摄像头。请检查：")
            print("1. 摄像头是否已正确连接")
            print("2. 是否被其他程序占用")
            print("3. 是否有摄像头访问权限")
            return

        print(f"成功打开摄像头 {camera_index}")

        # 设置摄像头属性
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        if face_cascade.empty():
            print("错误：无法加载人脸检测模型文件")
            return

        frame_counter = 0
        last_known_faces = []

        while True:
            success, frame = camera.read()
            if not success:
                print("无法读取摄像头画面")
                break

            if frame_counter % 5 == 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    last_known_faces = []
                    for (x, y, w, h) in faces:
                        ROI = gray[y:y + h, x:x + w]
                        ROI = cv2.resize(ROI, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

                        label_unmasked, prob_unmasked = -1, 0.0
                        label_masked, prob_masked = -1, 0.0

                        if face_recognition_model:
                            label_unmasked, prob_unmasked = face_recognition_model.predict(ROI)

                        if masked_face_model:
                            label_masked, prob_masked = masked_face_model.predict(ROI)

                        # Heuristic to decide if a mask is worn
                        if prob_masked > prob_unmasked and prob_masked > current_config['threshold']:
                            # Masked model is more confident
                            show_name = f"{masked_name_list[label_masked]} (Masked)"
                            final_prob = prob_masked
                            show_text = f"{show_name}: {final_prob:.2f}"
                            log_recognition_event(show_name, final_prob)
                        elif prob_unmasked > current_config['threshold']:
                            # Unmasked model is more confident
                            show_name = name_list_camera[label_unmasked]
                            final_prob = prob_unmasked
                            show_text = f"{show_name}: {final_prob:.2f}"
                            log_recognition_event(show_name, final_prob)
                        else:
                            show_text = "Unknown"

                        last_known_faces.append(((x, y, w, h), show_text))
                except Exception as e:
                    print(f"处理帧时发生错误：{str(e)}")
                    continue

            for (face_coords, text) in last_known_faces:
                x, y, w, h = face_coords
                cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            frame_counter += 1
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("无法编码图像帧")
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"编码或传输帧时发生错误：{str(e)}")
                continue

    except Exception as e:
        print(f"视频流发生错误：{str(e)}")
    finally:
        print("正在释放摄像头...")
        camera.release()
        print("摄像头已释放")


def gen_frames_collect():
    """Generator function for video streaming during face collection. Manages its own camera instance."""
    global is_collecting, current_collection_name, collected_image_count, target_image_count, last_save_time

    # print("Attempting to open camera for collection...") # Commented out
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        # print("ERROR: Cannot open webcam for collection. Please check if camera is in use or drivers are installed.") # Commented out
        return

    # print("Collection camera opened successfully.") # Commented out

    try:
        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        if face_cascade.empty():
            # print("ERROR: Could not load face cascade classifier. Check path: config/haarcascade_frontalface_alt.xml") # Commented out
            return

        save_interval_seconds = 1.0 / current_config.get('collection_fps', 1)  # Default to 1 image per second
        # print(f"Collection FPS set to: {current_config.get('collection_fps', 1)}, save interval: {save_interval_seconds} seconds.") # Commented out

        while True:
            success, frame = camera.read()
            if not success:
                # print("Failed to read frame from camera during collection. Stream might have ended or camera disconnected.") # Commented out
                break

            display_frame = frame.copy()  # Create a copy for display

            if is_collecting and current_collection_name:
                # print(f"Collecting for {current_collection_name}. Current count: {collected_image_count}/{target_image_count}") # Commented out
                faces = face_cascade.detectMultiScale(frame, 1.3, 5)
                output_folder = os.path.join('data', current_collection_name)

                if not os.path.exists(output_folder):
                    # print(f"Creating output directory: {output_folder}") # Commented out
                    os.makedirs(output_folder)

                if len(faces) > 0:
                    # print(f"Detected {len(faces)} face(s).") # Commented out
                    # Only process the largest face
                    (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])

                    # Draw rectangle on the display frame
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save the detected face (color image)
                    face_roi = frame[y:y + h, x:x + w]

                    current_time = time.time()
                    if current_time - last_save_time >= save_interval_seconds:
                        if w > 100 and h > 100:  # Ensure reasonable size
                            if collected_image_count < target_image_count:
                                collected_image_count += 1
                                img_path = os.path.join(output_folder, f"{collected_image_count}.jpg")
                                # print(f"Attempting to save image: {img_path}") # Commented out
                                try:
                                    cv2.imwrite(img_path, face_roi)
                                    # print(f"Successfully saved {img_path}") # Commented out
                                    last_save_time = current_time
                                except Exception as img_e:
                                    # print(f"ERROR: Failed to save image {img_path}: {img_e}") # Commented out
                                    pass  # Suppress error for now
                            else:
                                # print("Target image count reached, not saving more.") # Commented out
                                pass  # Suppress message for now
                        else:
                            # print(f"Face too small to save: width={w}, height={h}") # Commented out
                            pass  # Suppress message for now
                    else:
                        # print(f"Not enough time passed since last save. Remaining: {save_interval_seconds - (current_time - last_save_time):.2f}s") # Commented out
                        pass  # Suppress message for now
                else:
                    # print("No faces detected.") # Commented out
                    pass  # Suppress message for now

                # Update status text on display frame
                status_text = f"Collecting: {collected_image_count}/{target_image_count}"
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if collected_image_count >= target_image_count:
                    is_collecting = False  # Stop collection automatically
                    # print(f"Collection for {current_collection_name} finished. {collected_image_count} images saved.") # Commented out

            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                # print("Failed to encode frame to JPG.") # Commented out
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        # print(f"An error occurred during collection streaming: {e}") # Commented out
        pass  # Suppress error for now
    finally:
        # print("Releasing collection camera.") # Commented out
        camera.release()
        # Also reset collection state on disconnect
        is_collecting = False
        current_collection_name = None
        collected_image_count = 0  # Reset count on stream end


# --- Old Functions for API (can be kept for other purposes) ---

def endwith(s, *endstring):
    resultArray = map(s.endswith, endstring)
    if True in resultArray:  # Removed extra parenthesis
        return True
    else:
        return False





# --- Globals for Face Collection ---
is_collecting = False
current_collection_name = None
collected_image_count = 0
target_image_count = 0
collection_start_time = 0
last_save_time = 0


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template('index.html')


# Removed /register and /collect_faces routes
# @app.route("/register")
# def register():
#     return render_template('register.html')

# @app.route("/collect_faces")
# def collect_faces_page():
#     return render_template('collect_faces.html')

@app.route("/register_collect")
def register_collect_page():
    return render_template('register_and_collect.html')


@app.route("/show_camera")
def show_camera():
    return render_template('show.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_collect')
def video_feed_collect():
    return Response(gen_frames_collect(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/start_face_collection", methods=["POST"])
def start_face_collection():
    global is_collecting, current_collection_name, collected_image_count, target_image_count, collection_start_time, last_save_time
    data = request.json
    name = data.get('name')
    if not name or not name.strip():
        return jsonify({'success': False, 'message': 'Name is required.'})

    # Camera is now initialized by the generator, so no need to call anything here.
    is_collecting = True
    current_collection_name = name.strip()
    collected_image_count = 0
    target_image_count = current_config.get('num_images_to_collect', 50)
    collection_start_time = time.time()
    last_save_time = time.time()  # Initialize last save time

    output_folder = os.path.join('data', current_collection_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return jsonify({'success': True, 'message': f'Starting collection for {current_collection_name}'})


@app.route("/api/stop_face_collection", methods=["POST"])
def stop_face_collection():
    global is_collecting, current_collection_name
    is_collecting = False
    current_collection_name = None
    # The camera will be released by the generator's finally block when the client disconnects.
    return jsonify({'success': True, 'message': 'Collection stopped.'})


@app.route("/api/collection_status", methods=["GET"])
def collection_status():
    global is_collecting, collected_image_count, target_image_count, current_collection_name
    return jsonify({
        'is_collecting': is_collecting,
        'collected_count': collected_image_count,
        'target_count': target_image_count,
        'name': current_collection_name
    })


# --- Globals for Training Progress ---
training_queue = queue.Queue()
training_process = None
training_thread = None


@app.route("/users")
def users_page():
    return render_template('users.html')


@app.route("/api/users")
def get_users():
    users = read_name_list(current_config['data_dir'])
    return jsonify(users)


@app.route("/api/users/delete/<name>", methods=["POST"])
def delete_user(name):
    user_path = os.path.join('dataset', name)
    if os.path.exists(user_path) and os.path.isdir(user_path):
        try:
            shutil.rmtree(user_path)
            return jsonify({'success': True,
                            'message': f'User {name} and their data deleted successfully. Remember to retrain the model.'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error deleting user {name}: {str(e)}'})
    else:
        return jsonify({'success': False, 'message': f'User {name} not found.'})


@app.route("/api/users/rename/<old_name>", methods=["POST"])
def rename_user(old_name):
    new_name = request.json.get('new_name')
    if not new_name or not new_name.strip():
        return jsonify({'success': False, 'message': 'New name is required.'})

    old_path = os.path.join(current_config['data_dir'], old_name)
    new_path = os.path.join(current_config['data_dir'], new_name.strip())

    if not os.path.exists(new_path):
        return jsonify({'success': False, 'message': f'User {new_name.strip()} already exists.'})

    try:
        os.rename(old_path, new_path)
        return jsonify({'success': True,
                        'message': f'User {old_name} renamed to {new_name.strip()} successfully. Remember to retrain the model.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error renaming user {old_name}: {str(e)}'})


@app.route("/settings")
def settings_page():
    return render_template('settings.html', config=current_config)


@app.route('/register_by_upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'})
        file = request.files['file']
        user_name = request.form.get('user_name')
        if not user_name:
            return jsonify({'success': False, 'message': 'Please provide a user name.'})
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
        if file:
            # Read the image directly from the file stream
            np_img = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if image is None:
                return jsonify(
                    {'success': False, 'message': 'Could not decode image. Please ensure it is a valid image file.'})

            success, message = process_and_save_face(image, user_name, source='upload')

            if success:
                return jsonify({'success': True, 'message': message})
            else:
                return jsonify({'success': False, 'message': message})

    return redirect(url_for('register_collect_page'))


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global current_config
    if request.method == "GET":
        return jsonify(current_config)
    elif request.method == "POST":
        new_settings = request.json
        # Validate and update only allowed settings
        for key, value in new_settings.items():
            if key in current_config:
                # Basic type conversion/validation
                if isinstance(current_config[key], int):
                    try:
                        current_config[key] = int(value)
                    except ValueError:
                        return jsonify({'success': False, 'message': f'Invalid integer value for {key}'}), 400
                elif isinstance(current_config[key], float):
                    try:
                        current_config[key] = float(value)
                    except ValueError:
                        return jsonify({'success': False, 'message': f'Invalid float value for {key}'}), 400
                else:
                    current_config[key] = value
        save_config(current_config)
        # Re-initialize globals if critical settings changed (e.g., image_size, threshold)
        # For simplicity, we'll just return success and note that restart might be needed.
        return jsonify({'success': True,
                        'message': 'Settings updated successfully. Some changes may require application restart to take effect.'})


@app.route("/users/<name>")
def user_detail_page(name):
    return render_template('user_detail.html', user_name=name)


@app.route("/api/users/<name>/images")
def get_user_images(name):
    user_path = os.path.join('dataset', name)
    images = []
    if os.path.exists(user_path) and os.path.isdir(user_path):
        for filename in os.listdir(user_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append(f'/dataset/{name}/{filename}')
    return jsonify(images)


@app.route("/history")
def history_page():
    return render_template('history.html')


@app.route("/api/history")
def get_history():
    history_entries = []
    if os.path.exists(RECOGNITION_LOG_FILE):
        with open(RECOGNITION_LOG_FILE, 'r') as f:
            history_entries = f.readlines()
    return jsonify(history_entries)





@app.route('/beauty')
def beauty():
    return render_template('beauty.html')


@app.route('/api/update_beauty_options', methods=['POST'])
def update_beauty_options():
    global beauty_options
    beauty_options = request.json
    return jsonify({'status': 'success'})


def beauty_frame(frame):
    """处理视频帧的美颜效果"""
    return beauty_processor.process_frame(frame, beauty_options)


@app.route('/video_feed_beauty')
def video_feed_beauty():
    """生成美颜处理后的视频流"""

    def generate():
        global camera_active
        cap = cv2.VideoCapture(0)
        while True:
            if not camera_active:
                # If camera is not active, release it and break the loop
                if cap.isOpened():
                    cap.release()
                break

            success, frame = cap.read()
            if not success:
                break

            # 应用美颜效果
            frame = beauty_frame(frame)

            # 转换为JPEG格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cap.isOpened():
            cap.release()

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    data = request.json
    folder_name = data.get('folder_name')
    current_beauty_options = data.get('beauty_options', {})  # Get current beauty options from frontend

    if not folder_name:
        return jsonify({'success': False, 'message': 'Folder name is required.'}), 400

    cap = cv2.VideoCapture(0)  # Open camera directly for capture
    if not cap.isOpened():
        return jsonify({'success': False, 'message': 'Failed to open camera for capture.'}), 500

    success, frame = cap.read()
    cap.release()  # Release camera immediately after capture

    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture frame from camera.'}), 500

    # Apply beauty effects to the captured frame
    processed_frame = beauty_processor.process_frame(frame, current_beauty_options)

    save_dir = os.path.join('beauty', folder_name)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(save_dir, f'captured_{timestamp}.jpg')

    try:
        cv2.imwrite(file_path, processed_frame)
        return jsonify({'success': True, 'message': 'Frame captured successfully.', 'file_path': file_path})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to save frame: {str(e)}'}), 500


@app.route('/api/process_image', methods=['POST'])
def process_image():
    """处理上传的图片"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    file = request.files['image']
    options = json.loads(request.form.get('options', '{}'))
    original_filename = request.form.get('original_filename', 'unknown')  # Get original filename

    # 读取图片
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 处理图片
    processed = beauty_processor.process_frame(image, options)

    # 转换回字节流
    is_success, buffer = cv2.imencode(".jpg", processed)
    if not is_success:
        return jsonify({'error': '图片处理失败'}), 500

    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    response.headers['X-Original-Filename'] = original_filename  # Send original filename back
    return response


@app.route('/api/save_processed_image', methods=['POST'])
def save_processed_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided.'}), 400

    file = request.files['image']
    original_filename = request.form.get('original_filename', 'unknown')

    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'success': False, 'message': 'Could not decode image.'}), 400

    # 获取原始文件所在目录
    original_dir = os.path.dirname(os.path.join('uploads', original_filename))
    if not os.path.exists(original_dir):
        original_dir = 'beauty/processed_images'  # 如果原目录不存在，使用默认目录

    os.makedirs(original_dir, exist_ok=True)

    # 生成新的文件名（添加时间戳和 _beauty 后缀）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(original_filename)[0]
    file_path = os.path.join(original_dir, f'{base_name}_beauty_{timestamp}.jpg')

    try:
        cv2.imwrite(file_path, image)
        return jsonify({
            'success': True,
            'message': 'Processed image saved successfully.',
            'file_path': file_path
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to save processed image: {str(e)}'
        }), 500


@app.route('/train_masked_model', methods=['POST'])
def train_masked_model_route():
    """开始训练口罩识别模型"""
    global masked_training_thread, masked_training_in_progress, masked_stop_training_flag

    if masked_training_in_progress:
        return jsonify({"status": "error", "message": "模型训练已在进行中"}), 400

    try:
        # 准备带口罩的数据集
        prepare_masked_dataset()

        # 重置停止标志
        masked_stop_training_flag.clear()
        masked_training_in_progress = True

        # 启动训练线程
        masked_training_thread = threading.Thread(
            target=train_masked_model,
            args=('mask_dataset', masked_stop_training_flag)
        )
        masked_training_thread.start()

        return jsonify({"status": "success", "message": "口罩模型训练已开始"})
    except Exception as e:
        masked_training_in_progress = False
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/masked_train_status')
def masked_train_status():
    """获取口罩模型训练状态"""
    global masked_training_in_progress
    return jsonify({
        "is_training": masked_training_in_progress
    })


@app.route('/stop_masked_training', methods=['POST'])
def stop_masked_training():
    """停止口罩模型训练"""
    global masked_stop_training_flag, masked_training_in_progress
    masked_stop_training_flag.set()
    masked_training_in_progress = False
    return jsonify({"status": "success", "message": "正在停止训练"})


# 从模型文件中导入训练函数
from faceRegnigtionModel import train_and_save_model as train_normal_model
from masked_face_model import train_and_save_model as train_masked_model

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
training_status = {
    "is_training": False,
    "message": "点击按钮开始训练模型",
    "phase": None,
    "progress": 0
}
training_thread = None
stop_training_flag = threading.Event()
progress_queue = Queue()


def update_training_status(is_training, message, phase=None, progress=None):
    """更新训练状态的辅助函数"""
    global training_status
    training_status.update({
        "is_training": is_training,
        "message": message,
        "phase": phase
    })
    if progress is not None:
        training_status["progress"] = progress
    logger.info(f"Training status updated: {training_status}")


def train_worker():
    """训练工作线程"""
    global training_status, training_thread
    try:
        # 准备口罩数据集
        update_training_status(True, "正在准备口罩数据集...", "preparing_mask_data", 0)
        if not prepare_masked_dataset():
            update_training_status(False, "准备口罩数据集失败", None, 0)
            return

        # 检查数据集
        if not os.path.exists('dataset/'):
            update_training_status(False, "找不到普通人脸数据集", None, 0)
            return
        if not os.path.exists('mask_dataset/'):
            update_training_status(False, "找不到口罩数据集", None, 0)
            return

        # 开始训练普通模型
        update_training_status(True, "正在训练普通模型...", "normal", 0)
        success = train_normal_model(stop_training_flag, progress_queue)

        if stop_training_flag.is_set():
            update_training_status(False, "训练已停止", None, 25)
            return

        if not success:
            update_training_status(False, "普通模型训练失败", None, 25)
            return

        # 开始训练口罩模型
        update_training_status(True, "正在训练口罩模型...", "masked", 50)
        success = train_masked_model('mask_dataset/', stop_training_flag, progress_queue)

        if stop_training_flag.is_set():
            update_training_status(False, "训练已停止", None, 75)
            return

        if success:
            update_training_status(False, "所有模型训练完成！", None, 100)
        else:
            update_training_status(False, "口罩模型训练失败", None, 75)

    except Exception as e:
        error_msg = f"训练过程中发生错误: {str(e)}"
        logger.error(error_msg)
        update_training_status(False, error_msg, None, 0)
    finally:
        training_thread = None


@app.route('/train_status')
def get_training_status():
    """获取训练状态"""
    global training_status, training_thread, progress_queue

    # 检查进度队列
    while not progress_queue.empty():
        progress_data = progress_queue.get()
        if isinstance(progress_data, dict):
            # 更新训练状态
            current_phase = training_status.get('phase')
            base_progress = 0 if current_phase == 'normal' else 50
            epoch_progress = (progress_data.get('epoch', 0) / 300) * 50  # 假设最大 epoch 为 300

            update_training_status(
                True,
                f"正在训练{'普通' if current_phase == 'normal' else '口罩'}模型 - Epoch {progress_data.get('epoch')}/300",
                current_phase,
                base_progress + epoch_progress
            )

    # 检查训练线程是否存在且已结束
    if training_thread and not training_thread.is_alive() and training_status['is_training']:
        update_training_status(
            False,
            '训练已完成' if not stop_training_flag.is_set() else '训练已停止',
            None,
            training_status.get('progress', 0)
        )

    return jsonify(training_status)


@app.route('/train', methods=['POST'])
def start_training():
    """开始训练"""
    global training_thread, training_status, stop_training_flag, progress_queue

    # 如果已经在训练中，返回错误
    if training_thread and training_thread.is_alive():
        return jsonify({
            "status": "error",
            "message": "训练已经在进行中"
        })

    # 重置状态
    stop_training_flag.clear()
    while not progress_queue.empty():
        progress_queue.get()

    update_training_status(True, "正在初始化训练环境...", "preparing", 0)

    # 创建并启动训练线程
    training_thread = threading.Thread(target=train_worker)
    training_thread.start()

    return jsonify({
        "status": "success",
        "message": "训练已开始"
    })


@app.route('/stop_training', methods=['POST'])
def stop_training_route():
    """停止训练"""
    global training_thread, training_status

    # 如果没有正在进行的训练，返回错误
    if not training_thread or not training_thread.is_alive():
        return jsonify({
            "status": "error",
            "message": "没有正在进行的训练"
        })

    # 设置停止标志
    stop_training_flag.set()
    update_training_status(True, "正在停止训练...", "stopping", training_status.get('progress', 0))

    return jsonify({
        "status": "success",
        "message": "训练停止信号已发送"
    })

# --- Makeup Transfer Routes ---
@app.route('/makeup_transfer')
def makeup_transfer_page():
    return render_template('makeup_transfer.html')


@app.route('/api/makeup_transfer', methods=['POST'])
def api_makeup_transfer():
    if 'no_makeup_image' not in request.files:
        return jsonify({'status': 'error', 'message': '未找到上传的原始图片文件。'}), 400

    file = request.files['no_makeup_image']
    makeup_style = request.form.get('makeup_style')

    if file.filename == '':
        return jsonify({'status': 'error', 'message': '未选择原始图片文件。'}), 400

    if not makeup_style:
        return jsonify({'status': 'error', 'message': '未选择妆容风格。'}), 400

    # Use a temporary directory for all intermediate files
    base_dir = os.path.abspath(os.path.dirname(__file__))
    temp_dir = os.path.join(base_dir, 'temp_uploads')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        no_makeup_filename = secure_filename(file.filename)
        no_makeup_path = os.path.join(temp_dir, no_makeup_filename)
        file.save(no_makeup_path)

        makeup_style_path = os.path.join(base_dir, 'static', 'makeup_styles', makeup_style)

        output_dir = os.path.join(base_dir, 'static', 'makeup_results')
        os.makedirs(output_dir, exist_ok=True)
        source_name = os.path.splitext(no_makeup_filename)[0]
        output_image_name = '{}_makeup.png'.format(source_name)
        output_path = os.path.join(output_dir, output_image_name)

        # Call makeup_transfer_tf.py as a separate process
        python_executable = sys.executable # Get the current Python executable
        command = [
            python_executable,
            os.path.join(base_dir, 'makeup_transfer_tf.py'),
            '--no_makeup', no_makeup_path,
            '--makeup_style', makeup_style_path,
            '--model_path', BEAUTY_GAN_MODEL_PATH,
            '--output', output_path
        ]
        
        # Run the command and capture output
        process = subprocess.run(command, capture_output=True, text=True)

        if process.returncode != 0:
            logger.error(f"Makeup transfer subprocess failed: {process.stderr}")
            return jsonify({'status': 'error', 'message': f'处理失败: {process.stderr}'}), 500

        return jsonify({'status': 'success', 'result_path': '/static/makeup_results/{}'.format(output_image_name)})

    except Exception as e:
        logger.error("Makeup transfer failed: {}".format(str(e)))
        return jsonify({'status': 'error', 'message': f'处理失败: {str(e)}'}), 500
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)





if __name__ == "__main__":
    initialize_globals()
    app.run(debug=True, threaded=True, use_reloader=False)