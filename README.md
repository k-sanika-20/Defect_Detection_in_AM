# 🔧 U-Net for Defect Segmentation in Additive Manufacturing

This repository contains a PyTorch implementation of a **U-Net** model designed for **semantic segmentation** of defects in grayscale images from a powder bed fusion additive manufacturing process. The model classifies each pixel into one of **six defect classes**, enabling pixel-wise defect localization.

---

## 🧠 Core Concepts

### 🧩 Semantic Segmentation
- Assigns each pixel in the input image to one of six defect classes.

### 🏗️ U-Net Architecture
- **Encoder-Decoder** structure with **skip connections** to preserve spatial detail.
- Designed for dense prediction tasks like medical imaging and defect detection.

### 🎯 Multi-label Classification
- Uses `BCEWithLogitsLoss`, supporting cases where a pixel may belong to multiple classes.

---

## 🏗️ Model Architecture

| Component         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Input**         | Grayscale image (`1` channel)                                               |
| **Output**        | 6-channel prediction mask (one per defect class)                           |
| **DoubleConv**    | `Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU`                |
| **Encoder**       | 4 downsampling blocks with MaxPooling                                      |
| **Bottleneck**    | High-dimensional feature extractor (1024 channels)                         |
| **Decoder**       | 4 upsampling blocks with `ConvTranspose2d` and skip connections            |
| **Final Conv**    | Outputs 6-channel segmentation map                                         |

### 🔢 Model Size

| Layer           | Parameters (Approx.) |
|----------------|----------------------|
| DoubleConv 1    | ~38K                 |
| DoubleConv 2    | ~221K                |
| DoubleConv 3    | ~885K                |
| DoubleConv 4    | ~3.5M                |
| Bottleneck      | ~14.2M               |
| Decoder         | ~18M                 |
| Final Conv      | ~390                 |
| **Total**       | **~36M**             |

---

## ⚙️ Setup & Usage

### 🐍 Requirements
```bash
pip install torch torchvision tqdm matplotlib
