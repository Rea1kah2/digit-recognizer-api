import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("="* 50)
print("                 Digit Recognizer")
print("="* 50)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"Training samples : {len(x_train)}")
print(f"Testing samples  : {len(x_test)}")
print(f"Image shape      : {x_train[0].shape} (28x28 pixels)")

# Preprocessing Data
print("\nPreprocessing data...")
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(f"Shape setelah reshape : {x_train.shape}")
print(f"Nilai pixel range     : {x_train.min():.1f} - {x_train.max():.1f}")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.10,
    height_shift_range=0.10,
)

# fit() -> Menghitung statistik internal dari data training
datagen.fit(x_train)
print("\nAugmentasi siap!")
print("-> Rotasi: 10 derajat")
print("-> Zoom: 10%")
print("-> Shift Horizontal: 10%")
print("-> Shift Vertical: 10%")

# Arsitektur Model
model = keras.Sequential([
    # Blok Konvolusi 1
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    keras.layers.BatchNormalization(), # Untuk normalisasi output agar training lebih stabil
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    # Blok Konvolusi 2
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    # Fully COnnected Layer
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax'),
])

model.summary()

# Compile Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks 1
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Callbacks 2
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

print("\nCallbacks aktif!")
print('-> EarlyStopping: berhenti jika val_loss stagnan selama 5 epoch')
print('-> ReduceLROnPlateau: kecilkan LR jika stagnan selama 3 epoch')

# Training dengan Augmentasi Data
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    steps_per_epoch=len(x_train) // 64,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluasi Model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Total Epochs: {len(history.history['loss'])}")

# Simpan Model
save_path = 'model/mnist_model.h5'
model.save(save_path)
print(f"\nModel disimpan ke {save_path}")