from flask import Flask, render_template, Response
import cv2 as cv 
import pyttsx3
import time
import threading
import queue
from gtts import gTTS
import os
import subprocess
app = Flask(__name__)
video_feed_active = False
cap = None

# Object detector constants
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
FONTS = cv.FONT_HERSHEY_COMPLEX

# Load class names
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Load YOLO model
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Initialize pyttsx3 engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Create a queue to store the object names to speak
object_queue = queue.Queue()

def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    desktop_path = os.path.join(os.path.expanduser('~'), '')
    filename = os.path.join(desktop_path, filename)
    # Remove existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)
    tts.save(filename)
    return filename

def play_audio(filename):
    # For Windows
    if os.name == 'nt':
        os.startfile(filename)
    # For macOS
    elif os.name == 'posix':
        subprocess.call(['open', filename])
    # For Linux
    else:
        subprocess.call(['xdg-open', filename])


def object_detector(image):
    global object_queue
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" %(class_names[classid[0]], score)
        la = class_names[classid[0]]
        cv.rectangle(image, box, color, 2)
        cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
        print(la)
        text = la
        filename_text_to_speech = "C://Users//manju//OneDrive//Desktop//Obstacle Detection Using Yolo V4//sample.mp3"
        saved_file = text_to_speech(la, filename_text_to_speech)
        print("Playing audio...")
        play_audio(saved_file)
        time.sleep(0.5)
        if classid in [0, 67, 2, 15, 16, 14, 17, 18, 19, 20, 21, 22, 23, 25, 26, 32, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 55, 56, 63, 64, 65, 66, 73, 74, 76]:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
    return data_list

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global cap
    while True:
        if video_feed_active:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                data = object_detector(frame)
                ret, buffer = cv.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
@app.route('/video_feed')
def video_feed():
    global video_feed_active, cap
    if not video_feed_active:
        cap = cv.VideoCapture(0)
        video_feed_active = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_feed')
def stop_video_feed():
    global video_feed_active, cap
    if video_feed_active:
        cap.release()
        video_feed_active = False
    return "Camera feed stopped"

if __name__ == '__main__':
    app.run(debug=True)
