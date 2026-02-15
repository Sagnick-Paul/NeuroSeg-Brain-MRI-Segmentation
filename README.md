🧠 Brain Tumor Segmentation using ResNet-UNet
📌 Project Overview

This project implements a deep learning model for automatic brain tumor segmentation from MRI scans using a UNet architecture with a pretrained ResNet34 encoder.

The objective is to accurately segment tumor regions at the pixel level using overlap-based loss functions and evaluation metrics.

🎯 Problem Statement

Brain tumor segmentation is a critical task in medical imaging. The challenge lies in:

Severe class imbalance (small tumor vs large background)

Precise boundary localization

Avoiding false negatives (missing tumor regions)

Traditional metrics like accuracy are misleading due to imbalance. Therefore, this project focuses on Dice, IoU, and Sensitivity metrics.

🏗 Model Architecture
🔹 Base Architecture:

UNet (Encoder-Decoder structure)

🔹 Encoder:

ResNet34 (Pretrained on ImageNet)

🔹 Why ResNet Encoder?

Strong low-level feature extraction

Faster convergence

Better generalization

Reduced training time

🔹 Output:

Binary segmentation mask (Tumor vs Background)

🧪 Loss Functions Explored

The training process involved experimentation with multiple loss combinations:

1️⃣ BCE + Dice Loss

Baseline configuration.

2️⃣ Dice + Focal Loss

To handle class imbalance and hard pixels.

3️⃣ BCE + Dice + Focal (Combined Loss)

Weighted combination to balance stability and overlap optimization.

📊 Evaluation Metrics

Since segmentation suffers from heavy class imbalance, the following metrics were used:

Dice Score (Primary metric)

IoU (Intersection over Union)

Sensitivity (Recall)

Accuracy was not used as it is dominated by background pixels.

🚀 Training Details

Framework: PyTorch

Model Library: segmentation_models_pytorch

Optimizer: Adam

Scheduler: ReduceLROnPlateau

Epochs: 20–25

GPU: Google Colab CUDA

📈 Performance Results (Best Model)
Metric	Test Score
Dice Score	0.74
IoU Score	0.61
Sensitivity	0.72

These results demonstrate strong overlap accuracy and good tumor detection capability.

🔬 Tuning Journey
Step 1: Baseline UNet

Moderate performance

Slower convergence

Step 2: ResNet34 Pretrained Encoder

Significant Dice improvement

Faster convergence

Better generalization

Step 3: Loss Function Tuning

Introduced Dice Loss for overlap optimization

Added Focal Loss for class imbalance

Conducted ablation experiments

Step 4: Threshold Tuning & Validation Monitoring

Early stopping applied

Learning rate scheduler used

Best checkpoint saved based on validation Dice

📊 Visualization

The project includes:

Training vs Validation Loss Curves

Dice Curve Analysis

Sample Prediction Visualizations

Overlayed Tumor Masks

🛠 Tools & Libraries Used

PyTorch

segmentation_models_pytorch

NumPy

Matplotlib

Seaborn

OpenCV

Torchinfo

tqdm

🧠 Key Learnings

Accuracy is misleading in imbalanced medical segmentation

Dice and IoU are more reliable evaluation metrics

Pretrained encoders significantly boost performance

Focal loss is dataset-dependent

Proper checkpointing prevents performance regression

📌 Future Improvements

Test-Time Augmentation (TTA)

Boundary-aware loss

Multi-class tumor segmentation

Cross-validation

Small tumor performance analysis
