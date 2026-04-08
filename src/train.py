import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- SETTINGS ---
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15 

# --- DATA PREP ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    '../dataset_final/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

val_gen = test_datagen.flow_from_directory(
    '../dataset_final/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# --- THE DEEP BRAIN ---


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- START TRAINING ---
print("Training the Deep Manual Model...")
model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

# --- SAVE ---
os.makedirs('../models', exist_ok=True)
model.save('../models/cat_dog_manual_best.keras')
print("New deep model saved!")