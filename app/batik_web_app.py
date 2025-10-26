import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import os

# Load Model
model = load_model('../script/mobilenetv2_batik_final.keras')
# Load Labels (pastikan urutannya sesuai dengan training)
labels_path = 'labels.txt'  # Buat file ini jika belum ada
if os.path.exists(labels_path):
    with open(labels_path, 'r') as file:
        labels = [line.strip() for line in file]
else:
    # fallback untuk 3 label
    labels = ['Batik Kawung', 'Batik Parang', 'Batik Sekar Jagad']

# Inisialisasi kamera
camera = cv2.VideoCapture(0)

app = Flask(__name__)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                # Preprocessing
                resized_frame = cv2.resize(frame, (224, 224))
                normalized_frame = resized_frame / 255.0
                reshaped_frame = np.reshape(normalized_frame, (1, 224, 224, 3))

                # Prediction
                prediction = model.predict(reshaped_frame)
                predicted_index = int(np.argmax(prediction))

                # Proteksi jika index tidak ada dalam label
                if predicted_index < len(labels):
                    label = labels[predicted_index]
                else:
                    label = 'Unknown'

                # Tambahkan label ke frame
                cv2.putText(frame, f'{label}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error during prediction: {e}")
                cv2.putText(frame, "Prediction Error", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode frame untuk ditampilkan di web
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
