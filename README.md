# Cat vs. Dog Recognition 🐾

A Deep Learning project built with **TensorFlow** and **Keras** to classify images of cats and dogs using a custom-built Convolutional Neural Network (CNN).

## 🚀 Overview
This repository contains a manual CNN architecture designed to identify pets. To ensure high generalization and prevent overfitting, the model utilizes **Data Augmentation** and **Dropout layers**, achieving a final training accuracy of **84.94%**.

---

## 🧠 Model Architecture
* **4 Convolutional Layers:** Extracts features from simple edges to complex shapes.
* **MaxPooling:** Reduces dimensionality while retaining spatial features.
* **Dropout (0.5):** Prevents memorization by randomly deactivating neurons.
* **Data Augmentation:** Flips, rotates, and zooms images to improve real-world performance.

---

## 📊 Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Final Training Accuracy** | 84.94% |
| **Validation Accuracy** | ~82.10% |
| **Epochs** | 15 |
| **Optimizer** | Adam |

---

## 🛠️ Project Structure
```text
Cat-vs-Dog-recognition/
├── data/               # Raw Kaggle images (Git Ignored)
├── dataset_final/      # Organized train/test split (Git Ignored)
├── models/             # Saved .keras model files
├── src/
│   ├── setup_data.py   # Data organization script
│   ├── train.py        # Deep Manual training script
│   └── predict.py      # Inference script for testing
└── requirements.txt
```

🚦 Getting Started
1. Clone the Repository
```
git clone [https://github.com/dhruvil-1207/Cat-vs-Dog-recognition.git](https://github.com/dhruvil-1207/Cat-vs-Dog-recognition.git)
cd Cat-vs-Dog-recognition
```

2. Install Dependencies
```
pip install -r requirements.txt
```

3. Setup Dataset (Optional)
If you wish to re-train the model, place the Kaggle "Dogs vs Cats" images in the data/ folder and run:
```
python src/setup_data.py
```

🔍 Training & Prediction
Phase 1: Training (Optional)
The repository includes pre-trained weights in the models/ folder. To re-train the model from scratch:

```
python src/train.py
```

Phase 2: Prediction
To test the model on any image, run the inference script:

```
python src/predict.py
```

Instructions after execution:
The script will prompt you for an Image Path. You can provide:

A Relative Path: ../my_pet.jpg

An Absolute Path: D:\Images\dog_test.jpg

A Test Image: ../dataset_final/test/cats/cat.1001.jpg

Output:
The script returns the predicted animal and a Confidence Score (e.g., RESULT: DOG 🐶 (Confidence: 92.45%)).
