# imbd-wiki_age_prediction
[Download Model from Google Drive]
https://drive.google.com/file/d/1cGHFZmeHDVWvOCzOO5BbgS_TxA_Vt7kk/view?usp=sharing
dataset link:
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

📌 README.md - Age Prediction Using CNN

# 🧑‍🔬 Age Prediction Using CNN  

This project builds a **Convolutional Neural Network (CNN)** to predict a person's age from images. The model is trained on the **IMDB-WIKI dataset**, utilizing **TensorFlow and Keras** for deep learning.

---


## 📊 Exploratory Data Analysis (EDA)  

**EDA was performed using:**
- `matplotlib` & `seaborn` for visualizing age distribution.
- **Key Findings:**
  - **Gender Distribution:** 72% Male, 27% Female.
  - **Age Range:** 20 to 100 years.
  - **Majority Age Group:** 20 to 40 years.
  - **Outlier Removal:** No outliers removed.

---

## 🛠️ Model Architecture  

The CNN model consists of:  
✅ **Convolutional Layers (Conv2D + ReLU + MaxPooling)**  
✅ **Fully Connected Layers (Dense, Dropout)**  
✅ **Adam Optimizer with Categorical Cross-Entropy Loss**  

---

## 🚀 Training & Performance  

✅ **Dataset:** IMDB-WIKI (Filtered: Face Score ≥ 2.5)  
✅ **Training:** 80% Train - 20% Validation Split  
✅ **Loss Function:** MSE (Mean Squared Error)  
✅ **Performance:**  
- **MAE 📈 `7.8`
- **Best Performance:** **20-30 Age Group**
- **Challenges:** Higher error in **70+ age group**  

---

## 📥 Download Model  

You can download the **trained model** from Google Drive:  

📥 **[Download age_prediction_model.h5](https://drive.google.com/uc?export=download&id=YOUR_FILE_ID)**  

---

## 📌 Installation & Usage  

### **1️⃣ Install Dependencies**  

pip install -r requirements.txt
2️⃣ Run Predictions

from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model


# Load and preprocess image
img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (128, 128)) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
age = model.predict(img)
print(f"Predicted Age: {age}")
