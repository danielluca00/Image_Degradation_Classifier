# Image Degradation Classifier

## Overview

This repository implements a **deep learning–based image degradation classifier** that identifies the **dominant degradation** affecting an image.

The model performs **multi-class, single-label classification** and is intended as a **pre-processing step** for image enhancement or restoration pipelines.

---

## Supported Degradation Classes

The classifier predicts one of the following classes:

- `blur`
- `motion_blur`
- `noise`
- `jpeg`
- `pixelation`
- `low_light`
- `high_light`
- `low_contrast`
- `color_distortion`
- `clean`

---

## Model

- Backbone: **ResNet-18** (ImageNet pretrained)
- Input size: **256 × 256**
- Loss: Cross-Entropy
- Optimizer: Adam

The model is lightweight and can be used for **CPU inference** and **GPU training**.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
### 2. Select clean images from ImageNet
Extract a subset of clean images from ImageNet to be used as a base dataset:
```bash
python scripts/select_clean_images.py
```
### 3. Generate the degradation dataset
Generate a complete dataset with synthetic degradations, automatically split into train, validation, and test sets:
```bash
python scripts/generate_dataset.py
```
### 4. Train the classifier
```bash
python scripts/train_classifier.py
```
### 5. Run inference
```bash
python scripts/inference.py
```
