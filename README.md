# Facial_Emotion_recognition_Assignment1_21i-1736_DeepLearning


This project implements a **Facial Emotion Recognition System** using three different CNN-based architectures for both **classification** (emotion categories) and **regression** (continuous emotion values).  
The goal is to evaluate and compare model performance across multiple state-of-the-art architectures while ensuring reproducibility and proper experimentation.

---

##  Project Structure

The project is divided into **4 main parts**:

1. **Data Loader**  
   - Handles image loading and annotation parsing.  
   - Prepares datasets for training, validation, and testing.  

2. **Model C – DenseNet**  
   - Used for both **classification** and **regression**.  
   - Baseline deep CNN with strong feature extraction capabilities.  

3. **Model B – MobileNet**  
   - Lightweight and efficient CNN for both tasks.  
   - Suitable for deployment on resource-constrained devices.  

4. **Model A – ResNet50**  
   - Deep residual network for classification and regression.  
   - Handles vanishing gradient problems effectively.  

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/areeba0/Facial_Emotion_recognition_Assignment1_21i-1736_DeepLearning.git
cd https://github.com/areeba0/Facial_Emotion_recognition_Assignment1_21i-1736_DeepLearning.git
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
- On **Windows**:
```bash
venv\Scripts\activate
```
- On **Mac/Linux**:
```bash
source venv/bin/activate
```

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

---


## Output & Results

- Training and validation metrics (accuracy, loss, MSE, etc.) are logged.  
- Reports and plots generated for comparison across models.  
- You can find evaluation metrics.

---

## Notes
- This project is implemented in **Jupyter Notebook style**, so you can also open the `.ipynb` files for interactive experimentation.  
- If GPU is available, TensorFlow/Keras will automatically use it for training.  

---

## Features

- Supports **both classification and regression** tasks.  
- Implements **DenseNet, MobileNet, and ResNet50**.  
- Includes **data preprocessing & augmentation** to reduce overfitting.  
- Modular structure to compare models easily.  

---

