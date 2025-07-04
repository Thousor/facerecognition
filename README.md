# Face Recognition System

This is a web-based face recognition application built with Python, Flask, OpenCV, and Keras.

The system allows users to register new faces through a web interface, trains a recognition model on the collected data, and performs real-time face recognition directly in the browser.

## Features

- **Web-Based Interface**: All functionalities are accessible through a clean web UI.
- **Online Face Registration**: Easily add new users by collecting face data directly from a webcam.
- **Real-Time Recognition**: Performs live face recognition on a video stream embedded in the web page.
- **Optimized Performance**: Uses a skip-frame mechanism to ensure smooth video streaming during recognition.
- **Clear Feedback**: Displays the recognized name and the model's confidence score directly on the video feed.

## How to Use

### 1. Environment Setup

First, it is highly recommended to create a Python virtual environment.

Then, install all the necessary dependencies. You can create a `requirements.txt` file with the following content:

```txt
flask
numpy
opencv-python
tensorflow # or tensorflow-cpu
keras
# Add other libraries if you use them
```

And install them using pip:
```bash
pip install -r requirements.txt
```

### 2. Running the Application

To start the system, run the main application file:

```bash
python main.py
```

The web server will start, and you can access the application by navigating to `http://127.0.0.1:5000` in your web browser.

### 3. Registering a New Face

1.  On the home page, click on **"Register New Face"**.
2.  Allow the browser to access your webcam.
3.  Enter the person's name in the input field.
4.  Click **"Start Collecting"**.
5.  A new window will pop up, showing the camera feed. Please face the camera and allow the system to automatically capture about 50 images of your face. The window will close automatically when the collection is complete.

### 4. Training the Model

After registering a new face, you need to retrain the model to include the new data.

Run the `dataHelper.py` script from your terminal:

```bash
python dataHelper.py
```

This will update the `face.keras` model file with the newly added faces.

### 5. Performing Face Recognition

1.  Go back to the home page (`http://127.0.0.1:5000`).
2.  Click on **"Open Camera for Recognition"**.
3.  The page will display the live video feed from your webcam, with recognized faces labeled with their name and confidence score.

