from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import numpy as np
import pickle
import mediapipe as mp
import cv2
import base64
import re
import eventlet # Required for SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!'
socketio = SocketIO(app)

# --- LOAD YOUR MODEL AND SETUP MEDIAPIPE ---
model_filename = 'mudra_model_augmented.pkl'
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
# Use two separate Hands instances for different configurations
hands_live = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# --- Main route to serve the HTML page ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Endpoint for SINGLE IMAGE prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    sbuf = base64.b64decode(image_data)
    np_arr = np.frombuffer(sbuf, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = hands_static.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_hand_landmarks: return jsonify({'error': 'No hands detected.'})
    
    left_hand_landmarks, right_hand_landmarks = [0] * 63, [0] * 63
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = results.multi_handedness[idx].classification[0].label
        temp_landmarks = []
        for lm in hand_landmarks.landmark: temp_landmarks.extend([lm.x, lm.y, lm.z])
        if handedness == 'Left': left_hand_landmarks = temp_landmarks
        elif handedness == 'Right': right_hand_landmarks = temp_landmarks
            
    try:
        landmarks = np.array(left_hand_landmarks + right_hand_landmarks).reshape(1, -1)
        prediction = model.predict(landmarks)
        return jsonify({'mudra': prediction[0]})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'})

# --- WebSocket handler for REAL-TIME stream ---
@socketio.on('live_frame')
def handle_live_frame(data_image):
    sbuf = base64.b64decode(data_image.split(',')[1])
    np_arr = np.frombuffer(sbuf, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = hands_live.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    gesture_name = "Detecting..."
    if results.multi_hand_landmarks:
        left_hand_landmarks, right_hand_landmarks = [0] * 63, [0] * 63
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            temp_landmarks = []
            for lm in hand_landmarks.landmark: temp_landmarks.extend([lm.x, lm.y, lm.z])
            if handedness == 'Left': left_hand_landmarks = temp_landmarks
            elif handedness == 'Right': right_hand_landmarks = temp_landmarks
        try:
            landmarks = np.array(left_hand_landmarks + right_hand_landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)
            gesture_name = prediction[0]
        except: pass
    
    socketio.emit('live_prediction', {'mudra': gesture_name})

if __name__ == '__main__':
    socketio.run(app, debug=True)