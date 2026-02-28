# 🧠 Brain Tumor Segmentation using Efficient-b3-UNet

WEBSITE LINK : https://neuroseg-brain-mri-segmentation-krmujvucnpbhpyw4coyjgy.streamlit.app/

## 📌 Project Overview

This project implements a deep learning–based framework for automatic brain tumor segmentation from MRI scans using a **UNet architecture with a pretrained ResNet34 encoder**.

The objective is to perform accurate **pixel-level tumor segmentation** while effectively handling severe class imbalance using overlap-based loss functions and robust evaluation metrics.

---

## 🎯 Problem Statement

Brain tumor segmentation is a critical task in medical image analysis, assisting in diagnosis, treatment planning, and disease monitoring.

However, it presents significant challenges:

- Severe **class imbalance** (small tumor region vs large background)
- Requirement of **precise boundary localization**
- Avoiding **false negatives** (missing tumor pixels)

Traditional metrics like accuracy are misleading in such scenarios because background pixels dominate the image. Therefore, this project focuses on **Dice Score, IoU, and Sensitivity** for evaluation.

---

## 🏗 Model Architecture

### 🔹 Base Architecture
- **UNet (Encoder–Decoder Structure)**
  - Encoder extracts hierarchical features
  - Decoder reconstructs spatial resolution
  - Skip connections preserve fine-grained spatial details

### 🔹 Encoder
- **ResNet34 (Pretrained on ImageNet)**

### 🔹 Why Use a Pretrained ResNet Encoder?
- Strong low-level feature extraction
- Faster convergence
- Improved generalization
- Reduced training instability
- Better optimization on limited medical datasets

### 🔹 Output
- Binary segmentation mask (Tumor vs Background)
- Pixel-wise probability map

---

## 🧪 Loss Functions Explored

To handle class imbalance and optimize region overlap, multiple loss configurations were experimented with:

### 1️⃣ BCE + Dice Loss
- Combines pixel-wise stability (BCE) with overlap optimization (Dice)

### 2️⃣ Dice + Focal Loss
- Dice optimizes overlap
- Focal reduces dominance of easy background pixels

### 3️⃣ BCE + Dice + Focal (Weighted Combined Loss)
- Balances:
  - Stability (BCE)
  - Overlap accuracy (Dice)
  - Hard-pixel emphasis (Focal)

Ablation experiments were conducted to compare performance across configurations.

---

## 📊 Evaluation Metrics

Due to heavy class imbalance, the following metrics were used:

- **Dice Score (Primary Metric)**
- **IoU (Intersection over Union)**
- **Sensitivity (Recall)**

Accuracy was intentionally excluded as it is dominated by background pixels and does not reflect segmentation quality.

---

## 🚀 Training Details

- **Framework:** PyTorch  
- **Model Library:** segmentation_models_pytorch  
- **Optimizer:** Adam  
- **Learning Rate Scheduler:** ReduceLROnPlateau  
- **Epochs:** 20–25  
- **Hardware:** Google Colab (CUDA GPU acceleration)  
- **Checkpointing:** Best model saved based on validation Dice score  
- **Early Stopping:** Applied to prevent overfitting  

---

## 📈 Performance Results (Best Model)

| Metric        | Test Score |
|--------------|------------|
| Dice Score   | 0.78       |
| IoU Score    | 0.69       |
| Sensitivity  | 0.79       |

These results demonstrate:

- Strong spatial overlap performance
- Good tumor detection capability
- Effective handling of class imbalance

---

## 🔬 Tuning Journey

### Step 1: Baseline UNet
- Moderate performance
- Slower convergence

### Step 2: ResNet34 Pretrained Encoder
- Significant Dice improvement
- Faster convergence
- Better generalization

### Step 3: Loss Function Tuning
- Introduced Dice Loss for overlap optimization
- Added Focal Loss for imbalance handling
- Conducted structured ablation experiments

### Step 4: Threshold Tuning & Validation Monitoring
- Learning rate scheduling applied
- Early stopping implemented
- Best checkpoint selected using validation Dice

---

## 📊 Visualizations Included

- Training vs Validation Loss curves
- Dice Score progression
- Random sample predictions
- Ground truth vs predicted mask comparisons
- Overlayed tumor mask visualizations

---

## 🛠 Tools & Libraries Used

- PyTorch
- segmentation_models_pytorch
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- Torchinfo
- tqdm

---

## 🧠 Key Learnings

- Accuracy is misleading in imbalanced medical segmentation tasks
- Dice and IoU are more reliable overlap metrics
- Pretrained encoders significantly boost performance
- Focal loss effectiveness is dataset-dependent
- Proper checkpointing prevents performance regression
- Monitoring validation Dice is critical for stability

---

## 📌 Future Improvements

- Test-Time Augmentation (TTA)
- Boundary-aware loss functions
- Multi-class tumor segmentation
- K-fold cross-validation
- Small tumor performance analysis
- 3D volumetric segmentation

---

## 🧾 Conclusion

A ResNet-UNet–based segmentation framework was successfully implemented for brain tumor MRI segmentation.

The final model achieved a **Dice Score of 0.74** on unseen test data, demonstrating robust overlap optimization, strong generalization, and effective handling of class imbalance.
