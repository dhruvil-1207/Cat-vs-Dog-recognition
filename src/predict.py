import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# --- CONFIGURATION ---
MODEL_PATH = '../models/cat_dog_manual_best.keras'
IMG_SIZE = (150, 150)

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find {MODEL_PATH}")
        print("Make sure your training has finished successfully!")
        return None
    
    print("Loading model... please wait.")
    return tf.keras.models.load_model(MODEL_PATH)

def make_prediction(model, img_path):
    try:
        # 1. Load and resize the image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        
        # 2. Convert to array and normalize (0-1)
        img_array = image.img_to_array(img) / 255.0
        
        # 3. Add batch dimension: [1, 150, 150, 3]
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Predict!
        prediction = model.predict(img_array, verbose=0)
        
        # Since we used 'sigmoid', result > 0.5 is Dog, < 0.5 is Cat
        score = prediction[0][0]
        if score > 0.5:
            print(f"\n>>> RESULT: DOG 🐶 (Confidence: {score*100:.2f}%)")
        else:
            print(f"\n>>> RESULT: CAT 🐱 (Confidence: {(1-score)*100:.2f}%)")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    my_model = load_trained_model()
    
    if my_model:
        print("\n--- Prediction Script Ready ---")
        while True:
            img_input = input("\nEnter image path (or type 'exit' to quit): ").strip()
            
            if img_input.lower() == 'exit':
                break
            
            if os.path.exists(img_input):
                make_prediction(my_model, img_input)
            else:
                print("File not found. Please check the path and try again.")