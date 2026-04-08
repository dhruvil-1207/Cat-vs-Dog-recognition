import tensorflow as tf
import os

# 1. Load the existing "smart" model
print("Loading existing model...")
model = tf.keras.models.load_model('models/cat_dog_manual_best.keras')

# 2. Save it as .h5 (This is instant!)
print("Converting to .h5 format...")
model.save('models/cat_dog_manual_best.h5')

print("Done! You now have the .h5 file in your models folder.")