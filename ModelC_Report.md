# Facial Emotion Recognition — DenseNet121 (Classification & Regression)

> This project trains **two separate models** on the same face dataset using **DenseNet121** backbones:
> - **Classification**: predicts one of 8 emotions (`Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt`).
> - **Regression**: predicts continuous **Valence** and **Arousal** in the range `[-1, 1]`.
>
> The pipeline includes **stratified splits**, **Keras generators**, **data augmentation**, **transfer learning + fine‑tuning**, **early stopping**, **learning‑rate scheduling**, **L2 weight decay**, and **dropout**. It also exports **CSV reports** with rich metrics (Accuracy/F1/Kappa/AUCs for classification and RMSE/CCC/SAGR/Correlation for regression).


## 2) features of the code

**Data**
- Reads a `cleaned_dataset.csv` with at least these columns:
  - `file` → path to image file.
  - `expression` → integer label in `[0..7]` mapping to 8 emotions.
  - `valence`, `arousal` → floats in `[-1, 1]` for regression.
- **Stratified split** (70/15/15) using `train_test_split(..., stratify=expression)` keeps class balance across **train/val/test**.

**Labeling**
- A clear `labels_map` dictionary maps ints → strings for readability and reporting.

**Input pipeline**
- `ImageDataGenerator.flow_from_dataframe` for **classification**.
- A custom `RegressionDataGenerator (tf.keras.utils.Sequence)` for **regression** (loads with OpenCV, resizes, rescales to `[0,1]`).

**Augmentation**
- Mild to moderate transforms (rotation/shift/zoom/brightness/horizontal flip) to increase robustness while aiming to **preserve the face semantics**.

**Models**
- **Two separate Keras `Model`s**, both with **DenseNet121 (include_top=False, ImageNet weights)**:
  - Classification head: `GlobalAveragePooling → Dropout(0.5) → Dense(num_classes, softmax, L2)`.
  - Regression head: `GlobalAveragePooling → Dropout(0.5) → Dense(2, tanh, L2)` (bounded outputs for `[-1,1]`).

**Optimization**
- Stage‑1: **freeze** the backbone, train top layers (fast convergence, avoids catastrophic forgetting).
- Stage‑2: **fine‑tune** the **last ~30 layers** with a **small LR** to adapt domain‑specific features.

**Regularization & stability**
- **Dropout(0.5)**, **L2 weight decay** on the final Dense layer, explicit **EarlyStopping**, and **ReduceLROnPlateau**.
- Validation generators do **not** shuffle and are only rescaled.

**Losses & metrics**
- Classification: `categorical_crossentropy`, `accuracy`.
- Regression: `Huber` (robust to outliers), `mse` metric.
- Extra reports after training:
  - **Classification**: Accuracy, **F1 (weighted)**, **Cohen’s κ**, (approx) **Krippendorff’s α**, **AUC‑ROC** (OvR), **AUC‑PR**.
  - **Regression**: **RMSE**, **Pearson correlation** (per channel & average), **SAGR** (sign agreement), **CCC** (Concordance Correlation Coefficient).

**Artifacts**
- Weights checkpoints: `densenet_classification.h5`, `densenet_regression.h5` (best on validation).
- Reports: `densenet_classification_report.csv`, `densenet_regression_report.csv`.

---

## 3) Current results (from your latest run)

- **Classification**
  - Train Accuracy ≈ **0.23**
  - Val Accuracy ≈ **0.24**
  - *Baseline note*: with **8 classes**, random guessing is **0.125** — so the model is above naive but still low.
- **Regression**
  - Train MSE ≈ **0.19** (RMSE ≈ 0.436)
  - Val MSE ≈ **0.17** (RMSE ≈ 0.412)
  - *Scale note*: outputs live in `[-1,1]` (range=2). An RMSE ≈ 0.41 means average absolute error ≈ 0.33–0.35 in valence/arousal units (roughly).

These numbers reflect a **difficult dataset** and likely **domain gap** (see below).

---

## 4) Why is the accuracy low? 

1. **Dataset difficulty / label quality**  
   - Facial expressions contain **subtle cues**; classes like *Contempt* vs *Neutral*, or *Disgust* vs *Anger* are often **ambiguous** even for humans.
   - If labels are noisy (common in web‑scraped or student‑curated sets), **cross‑entropy struggles**: the model can’t discover a consistent pattern.

2. **Resolution & micro‑expressions**  
   - Input size is **128×128**. Many reliable facial cues live in **wrinkles/eye corners/lip shape** that are lost at lower resolutions.
   - Pretrained DenseNet121 expects ~**224x224**. Down‑scaling reduces the benefit of pretrained filters.

3. **Domain shift from ImageNet**  
   - ImageNet features are **object‑centric** and general. Faces are a **specialized domain** (alignment, local textures, skin tones, lighting). Without strong fine‑tuning or face‑specific pretraining, transfer is limited.

4. **Augmentation semantics**  
   - Strong brightness or rotation may **distort expressions** (e.g., heavy shadow flips happy ↔ neutral). If the augmentation distribution doesn’t match real test conditions, it can **harm generalization**.

5. **Annotation misalignment for regression**  
   - Valence/arousal annotations might be **inconsistent** across annotators. The model uses **Huber** to be robust, but noise still sets a floor for achievable error.

7. **Feature mis‑localization**  
   - Faces may not be **centered or aligned**; the model wastes capacity on background. Without detection/align steps, it sees non‑face pixels as signal.

---

## 5) What are we doing to handle overfitting 

- **Transfer Learning first (frozen backbone)** → ensures we learn a good linear head without destroying pretrained features. This reduces variance and helps the model find *generic* discriminative features quickly.
- **Fine‑tuning a *thin* slice (last ~30 layers) with tiny LR** → adapts higher‑level filters to facial nuances while avoiding over‑specializing the entire network.
- **Dropout(0.5)** on the pooled features → discourages co‑adaptation and forces the head to be robust to partial feature dropouts.
- **L2 weight decay** on the last Dense layer → keeps weights small and smoother decision boundaries; mitigates memorization.
- **EarlyStopping (patience=10)** → stops before the model starts fitting noise on the validation set.
- **ReduceLROnPlateau** → lowers LR automatically when progress stalls; this typically helps escape **sharp minima** and stabilize training.
- **Moderate augmentations** → we use geometry/photometric jitter, but keep them *face‑friendly* (no vertical flip, controlled brightness/rotate).

---

## 6) What else can we try next 


 **Increase input size to 224×224**  
   - Matches DenseNet’s pretraining; preserves micro‑expressions. Expect a **meaningful uplift** in classification.

 **Face detection + alignment (pre‑processing)**  
   - Crop to facial landmarks; optionally **align eyes** horizontally. Removes background noise; stabilizes features.

**Label smoothing (ε=0.05~0.1)** for classification  
   - Reduces overconfidence, yields better calibrated probabilities and sometimes **higher top‑1** when labels are noisy.

**Aug policy tuning**  
   - Replace ad‑hoc aug with **RandAugment**/**AugMix**-style policies (or at least reduce brightness jitter if it hurts).  
   - Add **CutMix/MixUp** for regularization (often helps robustness without architecture changes).

**LR schedule**  
   - Use **warmup + cosine decay** or **one‑cycle policy** (small warmup steps, peak LR, cosine down) → smoother convergence than step LR reductions.

**Longer fine‑tuning**  
   - Unfreeze **more** than 30 layers gradually (e.g., 30 → 80 → all) while **decreasing LR**; monitor early stopping to avoid overfitting.

**Better targets for regression**  
   - Standardize valence/arousal per‑dataset (`z‑score`) and **scale back** with `tanh` bound. Sometimes helps optimization and CCC.


10. **Data curation**  
   - Relabel a small **clean subset**; use it for validation only. A clean val set often **unlocks progress** by giving a truthful signal.


## 7) Code moodules

### a) Stratified Split
```python
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["expression"], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["expression"], random_state=42)
```
- **Why**: Keeps **class ratios** consistent across splits → fair validation and test estimates; reduces variance.

### b) ImageDataGenerator (Classification)
```python
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.3, brightness_range=[0.7, 1.3], horizontal_flip=True, fill_mode="nearest"
)
```
- **Why**: Regularizes the model and teaches it **invariance** to small camera/lighting changes typical in the wild.

### c) Custom Sequence (Regression)
- Loads with **OpenCV**, resizes, rescales; yields `X` and `y=[valence, arousal]`.
- **Why**: Flexible control for continuous targets; avoids label encoding and supports custom preprocessing.

### d) Transfer Learning → Fine‑Tuning
- `include_top=False, weights="imagenet"`; **freeze** encoder → train the head.
- Then **unfreeze last ~30 layers** with LR=1e‑5.
- **Why**: Safest path on **small/noisy datasets**; avoids overfitting while allowing gentle domain adaptation.

### e) Regularization & Callbacks
- **Dropout(0.5)** + **L2** → classic regularizers against overfitting.
- **EarlyStopping** (monitor val metric) → halts when generalization stops improving.
- **ReduceLROnPlateau** → finer convergence near minima; prevents oscillations.

### f) Losses
- **Cross‑entropy** for classification — theoretically optimal for multi‑class log-likelihood.
- **Huber** for regression — **robust** to annotation outliers, blends L1 and L2.

### g) Metrics & Reports
- **Classification**: besides Accuracy, we log **F1, κ, AUC‑ROC/PR** because Accuracy alone **hides per‑class failures** and calibration issues.
- **Regression**: **RMSE, CCC, SAGR, Pearson** — CCC measures both correlation **and** scale/location agreement; SAGR checks **directional correctness** (sign match).

---

## 11) Reproducibility tips

- Fix random seeds where possible (`numpy`, `tf`, split `random_state`).  
- Keep a **frozen validation set**; don’t tune hyper‑params on the test set.  
- Log **exact image resolution**, **augment policies**, **LR schedule**, and **unfrozen layers count** for each run.

---

## 12) Troubleshooting

- **Training stuck / NaNs** → check learning rate (too high), remove extreme augmentations, confirm images load correctly (no corrupt files).
- **Overfitting quickly** → reduce LR, add **label smoothing**, try **MixUp/CutMix**, increase dropout slightly (0.6), or unfreeze fewer layers.
- **Underfitting (both train & val low)** → raise input size to **224**, train longer with cosine decay, unfreeze **more** layers progressively.
- **Valence/arousal out of range** → tanh head enforces bounds; ensure labels are in `[-1,1]` and no scaling mistakes in CSV.


