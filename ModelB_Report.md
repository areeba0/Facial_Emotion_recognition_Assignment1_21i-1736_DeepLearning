
# README: Facial Emotion Recognition with MobileNetV2

## Introduction
This project implements a dual-model system for Facial Emotion Recognition using the MobileNetV2 architecture.

The task is divided into two parts:
1. **Classification Model**: Predicts categorical emotions (e.g., Happy, Sad, Angry, etc.).
2. **Regression Model**: Predicts continuous emotional dimensions (Valence and Arousal).

The dataset consists of facial images and corresponding annotation files. We utilize MobileNetV2 due to its efficiency in transfer learning and suitability for limited hardware resources.

---

## Dataset Challenges
The dataset used is inherently difficult and noisy, leading to low accuracy across models. Some major issues are:

- **Class Imbalance**: Certain emotions are underrepresented, causing biased learning.  
- **Low-Quality Annotations**: Labels may be noisy, particularly for valence/arousal values.  
- **Limited Data Samples**: A relatively small dataset size restricts the model's ability to generalize well.  
- **Inherent Ambiguity**: Human emotions are complex, and multiple expressions can look similar visually.  

These issues contribute significantly to the lower classification accuracy (0.37 test accuracy, 0.24 validation accuracy).

---

## Code Features & Modules

### 1. **Imports & Setup**
- Loads core libraries: `numpy`, `pandas`, `cv2`, `matplotlib`, `seaborn` for data processing and visualization.  
- Uses TensorFlow/Keras for deep learning and scikit-learn for splitting datasets.

### 2. **Dataset Loading & Preprocessing**
- Loads a pre-cleaned CSV (`cleaned_dataset.csv`) containing image file paths, labels, valence, and arousal values.  
- Creates label mappings for categorical emotion classes.  
- Splits data into **train**, **validation**, and **test** sets using stratified sampling.  

### 3. **Data Augmentation**
- Employs `ImageDataGenerator` with transformations (rotation, shifts, shear, zoom, flips, brightness).  
- Ensures better generalization and combats overfitting.  

### 4. **Classification Model (MobileNetV2)**
- Base model: `MobileNetV2` pretrained on ImageNet (top layer removed).  
- Adds GlobalAveragePooling + Dropout + Dense layer with softmax for classification.  
- Compiles with Adam optimizer, categorical crossentropy loss, and accuracy metric.  
- Fine-tuning: unfreezes last ~40 layers for better domain adaptation.  

### 5. **Regression Model (MobileNetV2)**
- Base model: `MobileNetV2` pretrained on ImageNet.  
- Adds GlobalAveragePooling + Dropout + Dense layer with **tanh** activation for valence & arousal prediction.  
- Compiles with Adam optimizer, MSE loss, and tracks MSE as a metric.  

### 6. **Custom Regression Data Generator**
- Custom `Sequence` class built to feed (image, valence, arousal) batches for regression training.  
- Ensures scalability and proper shuffling at epoch boundaries.  

### 7. **Callbacks**
- **EarlyStopping**: Stops training when validation metric plateaus.  
- **ReduceLROnPlateau**: Dynamically reduces learning rate.  
- **ModelCheckpoint**: Saves the best-performing weights.  

### 8. **Evaluation**
- Classification: Reports accuracy, F1-score, Cohen’s Kappa, AUC-ROC, AUC-PR.  
- Regression: Reports RMSE, Correlation, SAGR, and CCC.  
- Results saved as CSV files (`classification_report_mobilenet.csv`, `regression_report_mobilenet.csv`).  

---

## Why is Accuracy Low?
The classification accuracy of ~37% is significantly below ideal performance. Key reasons include:

- **Data Noise**: Incorrect or ambiguous labels reduce the model’s ability to learn meaningful features.  
- **Intra-class Variability**: Emotions like *Sad* or *Neutral* overlap visually, confusing the model.  
- **Overfitting Risk**: With limited samples, the model can memorize training data but fail on unseen data.  
- **Transfer Learning Limitations**: While pre-trained weights help, the ImageNet domain is very different from facial emotion data.  

The regression task performs slightly better (MSE ~0.15) because predicting continuous values is less sensitive to small labeling errors than strict categorical prediction.

---

## Dealing with Overfitting & Improving Generalization
We employ multiple strategies to reduce overfitting and improve generalization:

- **Data Augmentation**: Random rotations, shifts, flips, zooms, and brightness adjustments increase dataset variability.  
- **Transfer Learning**: Freezing initial MobileNetV2 layers and fine-tuning deeper layers improves feature reuse.  
- **Regularization**: Dropout layers and L2 regularization discourage over-reliance on specific neurons.  
- **Callbacks**: EarlyStopping prevents over-training, while ReduceLROnPlateau adjusts learning rates dynamically.  
- **Stratified Splits**: Ensures balanced representation of all classes across train, validation, and test sets.  

Despite these, the dataset’s inherent difficulty caps performance. In practice, larger and cleaner datasets (e.g., FER+, AffectNet) would significantly improve results.

---

## Conclusion
This project demonstrates the challenges of facial emotion recognition using deep learning, especially on limited and noisy datasets.  
While classification accuracy remains low (~37%), regression achieves more stable performance (MSE ~0.15).  
The methods employed highlight how transfer learning, augmentation, and careful training strategies can still yield useful insights despite dataset limitations.

### Future Improvements:
- Collect more balanced, high-quality emotion datasets.  
- Explore advanced architectures (e.g., EfficientNet, Vision Transformers).  
- Incorporate multimodal data (e.g., audio + facial cues) for richer emotion recognition.  
