import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model
import os
import tempfile

# Load Model
model = load_model('../script/mobilenetv2_batik_final.keras')

# Load Labels
labels_path = 'labels.txt'
if os.path.exists(labels_path):
    with open(labels_path, 'r') as file:
        labels = [line.strip() for line in file]
else:
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
                resized_frame = cv2.resize(frame, (224, 224))
                normalized_frame = resized_frame / 255.0
                reshaped_frame = np.reshape(normalized_frame, (1, 224, 224, 3))

                prediction = model.predict(reshaped_frame)
                predicted_index = int(np.argmax(prediction))

                label = labels[predicted_index] if predicted_index < len(labels) else 'Unknown'

                cv2.putText(frame, f'{label}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error during prediction: {e}")
                cv2.putText(frame, "Prediction Error", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

# === TAMBAHAN: Route untuk prediksi upload gambar ===
@app.route('/predict', methods=['POST'])
def predict_uploaded():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File tidak dipilih"}), 400

    try:
        # Baca file sebagai numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "File bukan gambar yang valid"}), 400

        # Preprocessing yang sama seperti live
        resized = cv2.resize(img, (224, 224))
        normalized = resized / 255.0
        input_image = np.expand_dims(normalized, axis=0)

        # Prediksi
        pred = model.predict(input_image)
        idx = int(np.argmax(pred))

        if idx < len(labels):
            label = labels[idx]
            desc_path = os.path.join("descriptions", f"{label}.txt")
            if os.path.exists(desc_path):
                with open(desc_path, "r", encoding="utf-8") as f:
                    description = f.read().strip()
            else:
                description = f"Deskripsi untuk {label} belum tersedia."
        else:
            label = "Unknown"
            description = "Motif tidak dikenali."

        return jsonify({
            "label": label,
            "description": description
        })

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Gagal memproses gambar."}), 500
# ================================================

if __name__ == '__main__':
    app.run(debug=True)