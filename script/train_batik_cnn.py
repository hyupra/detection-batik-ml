import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 

dataset_path = r"C:\wahyu.prayoga\detection-batik-ml\batik_pattern_dataset\archive\raw_batik_v2" 
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# ======= Image Settings =======
img_width, img_height = 224, 224
batch_size = 32

# ======= Data Augmentation =======
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
    # Jika perlu, Anda bisa tambahkan shear_range=0.2 atau brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)

# ======= Load Pre-trained MobileNetV2 (without top) =======
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False  # Freeze base untuk Tahap 1

# ======= Add Custom Classification Head =======
x = base_model.output
x = GlobalAveragePooling2D()(x)
# DROPOUT LEBIH TINGGI untuk regularisasi
x = Dropout(0.6)(x) 
# Dense Layer dengan L2 Regularization
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x) 
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ======= Tahap 1: Compile Model (Head Training) =======
print("====================================")
print("TAHAP 1: TRAINING CUSTOM HEAD (FREEZE BASE MODEL)")
print("====================================")

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.summary()

# ======= Callbacks Tahap 1 =======
callbacks = [
    ModelCheckpoint("mobilenetv2_batik_head_best.keras", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# ======= Train Model Tahap 1 =======
epochs = 50

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=callbacks
)

# ======= Save Model Tahap 1 =======
model.save("mobilenetv2_batik_head_trained.keras")


# ==========================================================
# ======= Tahap 2: Fine-Tuning MobileNetV2 =======
# ==========================================================

print("\n\n====================================")
print("TAHAP 2: FINE-TUNING MOBILE-NETV2 (UNFREEZE TOP LAYERS)")
print("====================================")

# 1. Unfreeze base model
base_model.trainable = True

# ...
# 2. Bekukan kembali sebagian besar layer terbawah 
# Coba hanya unfreeze 15-20 layer terakhir
for layer in base_model.layers[:-20]: # Kurangi dari 40 ke 20
    layer.trainable = False
# ...
# 3. Re-compile model dengan learning rate sangat kecil
model.compile(
    optimizer=Adam(learning_rate=0.000002), # Turunkan lagi!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# ...

print(f"Number of trainable weights for fine-tuning: {len(model.trainable_weights)}")


# 4. Lanjutkan pelatihan (Training Tambahan)
callbacks_ft = [
    ModelCheckpoint("mobilenetv2_batik_final_best.keras", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

# Jalankan Fine-Tuning selama beberapa epoch tambahan (misalnya 15 epoch)
fine_tune_epochs = 15
total_epochs = history.epoch[-1] + fine_tune_epochs

history_ft = model.fit(
    train_generator,
    validation_data=test_generator,
    # Lanjutkan dari epoch terakhir training Tahap 1
    epochs=total_epochs, 
    initial_epoch=history.epoch[-1], 
    callbacks=callbacks_ft
)

# ======= Gabungkan Riwayat Pelatihan (untuk plot) =======
for metric in history.history.keys():
    history.history[metric].extend(history_ft.history[metric])
    
# Ganti model.save yang lama
model.save("mobilenetv2_batik_final.keras") 
print("Final Fine-Tuned Model Saved as mobilenetv2_batik_final.keras")

# ======= Plot Training History =======
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Two-Stage Training)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("mobilenetv2_training_accuracy.png")
# plt.show() # Dikomen karena ini adalah script yang dijalankan di server