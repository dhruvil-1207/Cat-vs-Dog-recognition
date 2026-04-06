# Cat vs. Dog Recognition 🐾

A Deep Learning project built with **TensorFlow** and **Keras** to classify images of cats and dogs using a custom-built Convolutional Neural Network (CNN).

## 🚀 Overview
This repository contains a manual CNN architecture designed to identify pets. To ensure high generalization and prevent overfitting, the model utilizes **Data Augmentation** and **Dropout layers**, achieving a final training accuracy of **84.94%**.

---

## 🧠 Model Architecture
The model was built from scratch to balance depth and computational efficiency on a CPU:
* **4 Convolutional Layers:** Progressively extracting features from simple edges to complex shapes like ears and tails.
* **MaxPooling:** Reducing dimensionality while retaining important spatial features.
* **Dropout (0.5):** A regularization technique that randomly shuts off neurons during training to force the model to learn global patterns instead of memorizing pixels.
* **Data Augmentation:** Randomly flipping, rotating, and zooming images during training to improve real-world performance.

---

## 📊 Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Final Training Accuracy** | 84.94% |
| **Validation Accuracy** | ~82.10% |
| **Epochs** | 15 |
| **Optimizer** | Adam |
| **Loss Function** | Binary Crossentropy |

---

## 💻 Environment
* **Device:** Asus Vivobook 16X
* **Processor:** Intel Core i7
* **RAM:** 16GB
* **Platform:** Windows (Conda/TensorFlow)

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
├── requirements.txt    # Python dependencies
└── README.md