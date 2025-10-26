import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = load_model("script/batik_cnn_model.h5")

# Daftar label (disesuaikan dengan datasetmu)
class_names = ['Motif_Gajah_Lampung', 'Motif_Gamolan', 'Motif_Kapal', 'Motif_Pohon_Hayat', 'Motif_Pramadya', 'Motif_Sembagi'
               'Motif_Siger']  # <- sesuaikan dengan label asli
''
def predict_batik(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # atau (150,150) sesuai training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalisasi

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence
