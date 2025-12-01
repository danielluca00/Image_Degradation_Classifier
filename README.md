# Image Degradation Classifier

## Overview

This project implements a **deep learning-based image degradation classifier**, capable of automatically detecting the type of degradation affecting an input image.  

The classifier can recognize multiple types of degradation, such as:

- Blur  
- Gaussian Noise  
- Low Light  
- JPEG Artifacts  
- Pixelation  
- Clean images (no degradation)  

Once trained, the classifier can be used as a **pre-processing step** for image enhancement or restoration pipelines, selecting the appropriate enhancement method based on the detected degradation.

The repository includes the full pipeline, including:

1. **Dataset generation** – create synthetic degraded images from clean images for training and validation.  
2. **Data loaders** – for efficiently feeding images to the network.  
3. **Model architecture** – a convolutional neural network (e.g., ResNeXt50) fine-tuned for multi-class degradation classification.  
4. **Training scripts** – to train the classifier on synthetic or real-world datasets.  
5. **Inference scripts** – to classify new images and optionally output visual reports or degradation probabilities.  
