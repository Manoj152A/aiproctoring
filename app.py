import os
import cv2
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, session
from utils.face_recognition import FaceRecognition
from utils.object_detection import ObjectDetection
from utils.pose_estimation import PoseEstimation
from utils.audio_monitoring import AudioMonitor
from utils.anomaly_detection import AnomalyDetection

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

object_detection = ObjectDetection()
pose_estimation = PoseEstimation()
audio_monitor = AudioMonitor(threshold=0.1, check_interval=1)
anomaly_detection = AnomalyDetection()

@app.route('/')
def index():
    return render_template('capture.html')

@app.route('/capture_reference', methods=['POST'])
def capture_reference():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'reference_image.jpg')
    file.save(filename)
    
    try:
        face_recognition = FaceRecognition(filename)
        session['face_recognition_initialized'] = True
        logging.info("Face recognition initialized successfully")
        return jsonify({'success': True, 'message': 'Reference image captured successfully'})
    except Exception as e:
        logging.error(f"Error initializing face recognition: {str(e)}")
        return jsonify({'error': f"Error initializing face recognition: {str(e)}"}), 500

@app.route('/exam')
def exam():
    if not session.get('face_recognition_initialized'):
        return redirect(url_for('index'))
    audio_monitor.start_monitoring()
    return render_template('exam.html')

@app.route('/verify', methods=['POST'])
def verify():
    if not session.get('face_recognition_initialized'):
        return jsonify({'error': 'Face recognition not initialized. Please capture a reference image.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'current.jpg')
    file.save(filename)
    
    try:
        # Face recognition
        face_recognition = FaceRecognition(os.path.join(app.config['UPLOAD_FOLDER'], 'reference_image.jpg'))
        is_same, distance, error_message = face_recognition.verify_face(filename)
        
        if error_message:
            return jsonify({'error': error_message}), 400

        # Object detection
        img = cv2.imread(filename)
        objects = object_detection.detect(img)
        
        # Pose estimation
        pose_results = pose_estimation.estimate(img)
        
        # Audio events
        audio_events = audio_monitor.get_events()
        
        # Anomaly detection (replace with actual behavioral data in production)
        dummy_data = np.random.rand(10, 5)
        anomalies = anomaly_detection.detect(dummy_data)
        
        return jsonify({
            'is_same': bool(is_same),
            'distance': float(distance) if distance is not None else None,
            'objects': objects,
            'pose_results': pose_results,
            'audio_events': audio_events,
            'anomalies': anomalies.tolist() if anomalies is not None else None
        })
    except Exception as e:
        logging.error(f"Error in verification process: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/end_exam', methods=['POST'])
def end_exam():
    audio_monitor.stop_monitoring()
    return jsonify({'message': 'Exam ended', 'audio_events': audio_monitor.get_events()})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)