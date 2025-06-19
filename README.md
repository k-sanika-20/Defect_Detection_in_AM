# U-Net for Defect Segmentation in Additive Manufacturing

This repository contains a PyTorch implementation of a U-Net model designed for semantic segmentation of defects in grayscale images from a powder bed fusion additive manufacturing process. The model classifies each pixel into one of six defect classes, enabling pixel-wise defect localization.

---

## Core Concepts

### Semantic Segmentation
Assigns each pixel in the input image to one of six defect classes.

### U-Net Architecture
- Encoder-decoder structure with skip connections to preserve spatial information.
- Designed for dense prediction tasks like medical imaging and defect detection.

### Multi-label Classification
Uses `BCEWithLogitsLoss`, supporting cases where a pixel may belong to multiple classes.

---

## Model Architecture

| Component      | Description                                                                  |
|----------------|------------------------------------------------------------------------------|
| Input          | Grayscale image (1 channel)                                                  |
| Output         | 6-channel prediction mask (one per defect class)                             |
| DoubleConv     | Two Conv2d layers with BatchNorm and ReLU                                    |
| Encoder        | 4 downsampling blocks with MaxPooling and DoubleConv                         |
| Bottleneck     | High-level feature extractor with 1024 channels                              |
| Decoder        | 4 upsampling blocks using ConvTranspose2d and skip connections               |
| Final Conv     | Outputs a 6-channel segmentation map                                         |

### Model Size

| Layer           | Parameters (Approximate) |
|----------------|--------------------------|
| DoubleConv 1    | ~38K                    |
| DoubleConv 2    | ~221K                   |
| DoubleConv 3    | ~885K                   |
| DoubleConv 4    | ~3.5M                   |
| Bottleneck      | ~14.2M                  |
| Decoder         | ~18M                    |
| Final Conv      | ~390                    |
| **Total**       | **~36M**                |

---

## Setup and Usage

### Requirements

```bash
pip install torch torchvision opencv-python streamlit numpy pillow
