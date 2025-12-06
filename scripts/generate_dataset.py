import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# =========================================================
#                 CONFIGURAZIONE
# =========================================================
input_dir = "./clean_images"          # cartella immagini clean
output_dir = "./dataset"              # cartella finale dataset
img_size = (256, 256)

# Split del dataset
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Classi finali del dataset
degradations = [
    "clean",
    "blur",
    "noise",
    "low_light",
    "jpeg",
    "pixelation",
    "motion_blur",
    "high_light",
    "low_contrast",
    "color_distortion"
]

# =========================================================
#         FUNZIONI DI DEGRADAZIONE DINAMICHE
# =========================================================
def apply_blur(img):
    k = random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_noise(img):
    std = random.uniform(10, 50)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_low_light(img):
    factor = random.uniform(0.05, 0.4)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def apply_jpeg(img):
    quality = random.randint(10, 50)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_pixelation(img):
    h, w = img.shape[:2]
    factor = random.randint(4, 16)
    temp = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_motion_blur(img):
    k = random.randint(5, 25)
    angle = random.uniform(0, 360)
    kernel = np.zeros((k, k))
    kernel[int((k-1)/2), :] = np.ones(k)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k/2, k/2), angle, 1.0), (k, k))
    kernel /= np.sum(kernel)
    return cv2.filter2D(img, -1, kernel)

def apply_high_light(img):
    factor = random.uniform(1.5, 3.0)
    bright = img.astype(np.float32) * factor
    return np.clip(bright, 0, 255).astype(np.uint8)

def apply_low_contrast(img):
    alpha = random.uniform(0.3, 0.7)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return np.clip(alpha * img + (1 - alpha) * mean, 0, 255).astype(np.uint8)

def apply_color_distortion(img):
    factors = np.random.uniform(0.6, 1.4, size=(1, 1, 3))
    distorted = img.astype(np.float32) * factors
    return np.clip(distorted, 0, 255).astype(np.uint8)

# Mappa delle funzioni
degradation_funcs = {
    "clean": lambda x: x,
    "blur": apply_blur,
    "noise": apply_noise,
    "low_light": apply_low_light,
    "jpeg": apply_jpeg,
    "pixelation": apply_pixelation,
    "motion_blur": apply_motion_blur,
    "high_light": apply_high_light,
    "low_contrast": apply_low_contrast,
    "color_distortion": apply_color_distortion
}

# =========================================================
#           CARICAMENTO IMMAGINI CLEAN
# =========================================================
all_clean_images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
print(f"Trovate {len(all_clean_images)} clean_images.")

# Split train/val/test sulle clean images
train_imgs, temp_imgs = train_test_split(all_clean_images, test_size=(1-train_ratio), random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio/(test_ratio+val_ratio)), random_state=42)

splits = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs
}

# =========================================================
#       CREAZIONE CARTELLE dataset/train/class/
# =========================================================
for split in splits.keys():
    for d in degradations:
        Path(f"{output_dir}/{split}/{d}").mkdir(parents=True, exist_ok=True)

# =========================================================
#              GENERAZIONE IMMAGINI
# =========================================================
for split, img_list in splits.items():
    print(f"\nGenerazione split: {split} ({len(img_list)} immagini clean)")

    for img_name in tqdm(img_list):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)

        # Genera per ogni classe
        for cls in degradations:
            transformed = degradation_funcs[cls](img)
            save_path = f"{output_dir}/{split}/{cls}/{img_name}"
            Image.fromarray(transformed).save(save_path)

print("\nDataset creato e suddiviso in train/val/test con successo!")
