#!/usr/bin/env python
#encoding:utf-8
from __future__ import division

'''
功能： 将人脸识别模型暴露为web服务接口，用于演示的demo
'''

import os
import cv2
import sys
import time
import queue
import numpy as np
import subprocess
import shutil
import json # Added for config management
import threading # Added for training thread
from datetime import datetime # Added for logging timestamp
# Updated Flask imports
import threading
import time
import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename

from collect_data import collect_faces, process_and_save_face
from dataHelper import read_name_list
from faceRegnigtionModel import train_and_save_model, FaceRecognitionModel, get_datasets, IMAGE_SIZE, MODEL_FILE_PATH

# --- Global variables for training control ---
training_thread = None
stop_training_flag = threading.Event()
training_in_progress = False
# Imports from the model and camera demo
from faceRegnigtionModel import FaceRecognitionModel, TrainingProgressCallback, get_datasets, IMAGE_SIZE, BATCH_SIZE, DATA_DIR, Model
from dataHelper import read_name_list

CONFIG_FILE = 'config.json'
RECOGNITION_LOG_FILE = 'recognition_log.txt' # New: Log file for recognition events
current_config = {}

def load_config():
    global current_config
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            current_config = json.load(f)
    else:
        # Default configuration if file doesn't exist
        current_config = {
            "threshold": 0.5, # Changed default threshold
            "num_images_to_collect": 50,
            "image_size": 128,
            "batch_size": 32,
            "data_dir": "dataset/",
            "model_file_path": "face.keras",
            "collection_fps": 1 # Added default for collection_fps
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

app=Flask(__name__, static_url_path='/dataset', static_folder='dataset')
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- Globals ---
# Camera is now managed within each generator function, not globally.
face_recognition_model = None
name_list_camera = []
img_size = 128 # Will be updated from config

def initialize_globals():
    """Initializes recognition model and name list."""
    global face_recognition_model, name_list_camera, img_size
    # print("Initializing recognition model...") # Commented out
    try:
        # Update globals from config
        img_size = current_config.get('image_size', 128)

        name_list_camera = read_name_list(current_config['data_dir'])
        if len(name_list_camera) > 0:
            face_recognition_model = FaceRecognitionModel(num_classes=len(name_list_camera))
            face_recognition_model.load()
        else:
            face_recognition_model = None
            # print("Warning: dataset is empty. Live recognition will not work.") # Commented out
        # print("Recognition model initialization complete.") # Commented out
    except Exception as e:
        # print(f"ERROR: Could not initialize recognition model: {e}") # Commented out
        face_recognition_model = None

def gen_frames():
    """Generator function for video streaming. Manages its own camera instance."""
    global face_recognition_model, name_list_camera

    if not all([face_recognition_model, name_list_camera]):
        print("Streaming stopped: model not ready.")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        # print("ERROR: Cannot open webcam for recognition") # Commented out
        return
    
    # print("Recognition camera opened.") # Commented out
    
    try:
        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        frame_counter = 0
        last_known_faces = []

        while True:
            success, frame = camera.read()
            if not success:
                # print("Failed to read frame from camera.") # Commented out
                break
            
            if frame_counter % 5 == 0: # Skip-frame logic
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                last_known_faces = []
                for (x, y, w, h) in faces:
                    ROI = gray[y:y + h, x:x + w]
                    ROI = cv2.resize(ROI, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    label, prob = face_recognition_model.predict(ROI)
                    
                    # Always show the most similar name, and mark as uncertain if below threshold
                    show_name = name_list_camera[label]
                    if prob > current_config['threshold']:
                        show_text = f"{show_name}: {prob:.2f}"
                        log_recognition_event(show_name, prob) # Log recognized face
                    else:
                        show_text = f"{show_name} (Uncertain): {prob:.2f}"
                        # Optionally log uncertain recognitions as well
                        # log_recognition_event(f"{show_name} (Uncertain)", prob)

                    last_known_faces.append(((x, y, w, h), show_text))
            
            for (face_coords, text) in last_known_faces:
                x, y, w, h = face_coords
                cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            frame_counter += 1
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"An error occurred during recognition streaming: {e}")
    finally:
        print("Releasing recognition camera.")
        camera.release()

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

        save_interval_seconds = 1.0 / current_config.get('collection_fps', 1) # Default to 1 image per second
        # print(f"Collection FPS set to: {current_config.get('collection_fps', 1)}, save interval: {save_interval_seconds} seconds.") # Commented out

        while True:
            success, frame = camera.read()
            if not success:
                # print("Failed to read frame from camera during collection. Stream might have ended or camera disconnected.") # Commented out
                break
            
            display_frame = frame.copy() # Create a copy for display
            
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
                    face_roi = frame[y:y+h, x:x+w]
                    
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval_seconds:
                        if w > 100 and h > 100: # Ensure reasonable size
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
                                    pass # Suppress error for now
                            else:
                                # print("Target image count reached, not saving more.") # Commented out
                                pass # Suppress message for now
                        else:
                            # print(f"Face too small to save: width={w}, height={h}") # Commented out
                            pass # Suppress message for now
                    else:
                        # print(f"Not enough time passed since last save. Remaining: {save_interval_seconds - (current_time - last_save_time):.2f}s") # Commented out
                        pass # Suppress message for now
                else:
                    # print("No faces detected.") # Commented out
                    pass # Suppress message for now

                # Update status text on display frame
                status_text = f"Collecting: {collected_image_count}/{target_image_count}"
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if collected_image_count >= target_image_count:
                    is_collecting = False # Stop collection automatically
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
        pass # Suppress error for now
    finally:
        # print("Releasing collection camera.") # Commented out
        camera.release()
        # Also reset collection state on disconnect
        is_collecting = False
        current_collection_name = None
        collected_image_count = 0 # Reset count on stream end

# --- Old Functions for API (can be kept for other purposes) ---

def endwith(s,*endstring):
    resultArray=map(s.endswith,endstring)
    if True in resultArray: # Removed extra parenthesis
        return True
    else:
        return False

def detectOnePicture(path):
    model=Model()
    model.load()
    img=cv2.imread(path)
    img=cv2.resize(img,(128,128))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType,prob=model.predict(img)
    if picType!=-1:
        name_list=read_name_list('dataset/')
        print(name_list[picType],prob)
        res=u"识别为： "+name_list[picType]+u"的概率为： "+str(prob)
    else:
        res=u"抱歉，未识别出该人！请尝试增加数据量来训练模型！"
    return res

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
    last_save_time = time.time() # Initialize last save time

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

def training_worker(q):
    q.put("Starting data preparation...")
    # Run dataHelper.py as a subprocess
    data_process = subprocess.Popen(
        [sys.executable, "dataHelper.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    for line in iter(data_process.stdout.readline, ''):
        q.put(line.strip())
    data_process.stdout.close()
    data_process.wait()
    
    if data_process.returncode != 0:
        q.put(f"Data preparation failed with exit code {data_process.returncode}")
        q.put("TRAINING_COMPLETE") # Signal completion even on failure
        return

    q.put("Data preparation complete. Starting model training...")
    try:
        # 1. Load Datasets
        q.put("Loading datasets...")
        train_dataset, val_dataset, class_names = get_datasets(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
        num_classes = len(class_names)
        q.put(f"Found {num_classes} classes: {class_names}")

        # 2. Build Model
        q.put("Building model...")
        model = FaceRecognitionModel(num_classes=num_classes)
        model.build_model()

        # 3. Train Model
        q.put("Training model...")
        model.train_model(train_dataset, val_dataset, progress_queue=q)

        # 4. Evaluate Model
        q.put("Evaluating model...")
        model.evaluate_model(val_dataset)

        # 5. Save Model
        q.put("Saving model...")
        model.save()
        q.put("Model training and saving complete!")
    except Exception as e:
        q.put(f"Error during model training: {e}")
    finally:
        q.put("TRAINING_COMPLETE") # Signal for completion

def train_model_thread(stop_flag):
    global training_in_progress
    try:
        train_and_save_model(stop_flag)
    finally:
        training_in_progress = False

@app.route('/train', methods=['POST'])
def train():
    global training_thread, stop_training_flag, training_in_progress

    if training_in_progress:
        return jsonify({'status': 'error', 'message': 'Training is already in progress.'})

    training_in_progress = True
    stop_training_flag.clear()

    training_thread = threading.Thread(target=train_model_thread, args=(stop_training_flag,))
    training_thread.start()

    return jsonify({'status': 'success', 'message': 'Training started.'})


@app.route('/train_status')
def train_status():
    global training_in_progress
    return jsonify({'training': training_in_progress})


@app.route('/stop_training', methods=['POST'])
def stop_training():
    global stop_training_flag
    if training_thread and training_thread.is_alive():
        stop_training_flag.set()
        return jsonify({'status': 'success', 'message': 'Training stop signal sent.'})
    return jsonify({'status': 'error', 'message': 'No active training to stop.'})

@app.route('/train_progress')
def train_progress():
    def generate():
        while True:
            try:
                item = training_queue.get(timeout=1) # Wait for an item, with timeout
                if item == "TRAINING_COMPLETE":
                    yield f"data: Training complete! Please restart the main application (main.py) to load the new model.\n\n"
                    break
                elif isinstance(item, dict): # Training progress dictionary
                    yield f"data: Epoch {item['epoch']}: Loss={item['loss']:.4f}, Accuracy={item['accuracy']:.4f}, Val_Loss={item['val_loss']:.4f}, Val_Accuracy={item['val_accuracy']:.4f}\n\n"
                else: # String message from dataHelper.py or other general messages
                    yield f"data: {item}\n\n"
            except queue.Empty:
                # No new data, keep connection alive
                yield "data: \n" # Send a keep-alive message
            except Exception as e:
                print(f"Error in train_progress generator: {e}")
                yield f"data: Error: {e}\n\n"
                break
            time.sleep(0.1) # Small delay to prevent busy-waiting

    return Response(generate(), mimetype='text/event-stream')

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
            return jsonify({'success': True, 'message': f'User {name} and their data deleted successfully. Remember to retrain the model.'})
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

    if not os.path.exists(old_path):
        return jsonify({'success': False, 'message': f'User {old_name} not found.'})
    if os.path.exists(new_path):
        return jsonify({'success': False, 'message': f'User {new_name.strip()} already exists.'})

    try:
        os.rename(old_path, new_path)
        return jsonify({'success': True, 'message': f'User {old_name} renamed to {new_name.strip()} successfully. Remember to retrain the model.'})
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
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            image = cv2.imread(filepath)
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
        return jsonify({'success': True, 'message': 'Settings updated successfully. Some changes may require application restart to take effect.'})

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

@app.route("/detect", methods=["GET"])
def detectFace():
    if request.method=="GET": 
        picture=request.args['picture']
    start_time=time.time()
    res=detectOnePicture(picture)
    end_time=time.time()
    execute_time=str(round(end_time-start_time,2))
    tsg=u' 总耗时为： %s 秒' % execute_time
    msg=res+'\n\n'+tsg
    return msg

if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    initialize_globals()
    # use_reloader=False is important! Otherwise, the initialization runs twice.
    # threaded=True allows handling multiple requests (e.g., serving the page and the stream)
    app.run(debug=True, threaded=True, use_reloader=False)