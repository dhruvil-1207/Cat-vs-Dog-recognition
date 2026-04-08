import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

# --- LOAD MODEL ---
@st.cache_resource
@st.cache_resource
def load_my_model():
    # Change .keras to .h5
    return tf.keras.models.load_model('models/cat_dog_manual_best.h5')

model = load_my_model()

st.title("🐾 Cat vs Dog Recognition")

# --- DYNAMIC SAMPLE LOADING ---
SAMPLE_DIR = 'samples'
if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)

# Get all images from the folder, replacing underscores with spaces for the UI
sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
sample_options = ["Upload my own"] + [f.replace('_', ' ').split('.')[0] for f in sample_files]

# --- UI LAYOUT ---
st.write("### 📸 Select an Image")
selection = st.selectbox("Choose a sample image or upload your own:", sample_options)

img_to_process = None

if selection == "Upload my own":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_to_process = Image.open(uploaded_file)
else:
    # Find the original filename that matches the selection
    original_filename = sample_files[sample_options.index(selection) - 1]
    img_path = os.path.join(SAMPLE_DIR, original_filename)
    img_to_process = Image.open(img_path)

# --- PREDICTION LOGIC ---
if img_to_process is not None:
    # Center the image display
    st.image(img_to_process, caption=f'Selected: {selection}', use_container_width=True)
    
    if st.button('🚀 Identify Animal', use_container_width=True):
        # Pre-process
        img_resized = img_to_process.resize((150, 150))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Thinking...'):
            prediction = model.predict(img_array)
            score = float(prediction[0][0])
            
            if score > 0.5:
                st.subheader(f"Result: **DOG** 🐶")
                st.progress(score)
                st.write(f"Confidence: {score*100:.2f}%")
            else:
                st.subheader(f"Result: **CAT** 🐱")
                st.progress(1.0 - score)
                st.write(f"Confidence: {(1-score)*100:.2f}%")