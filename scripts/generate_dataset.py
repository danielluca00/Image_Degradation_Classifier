import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

# --- Configurazione ---
input_dir = "./clean_images"        # cartella con immagini clean
output_dir = "./dataset"                # cartella dove salvare il dataset generato
img_size = (256, 256)                   # ridimensiona tutte le immagini

# Tipi di degradazioni finali
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

# --- Funzioni di degradazione dinamiche ---
def apply_blur(img):
    k = random.choice([3,5,7,9])
    return cv2.GaussianBlur(img, (k,k), 0)

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
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def apply_pixelation(img):
    h, w = img.shape[:2]
    factor = random.randint(4, 16)
    temp = cv2.resize(img, (w//factor, h//factor), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_motion_blur(img):
    k = random.randint(5, 25)
    angle = random.uniform(0, 360)
    kernel = np.zeros((k, k))
    kernel[int((k-1)/2), :] = np.ones(k)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k/2, k/2), angle, 1.0), (k, k))
    kernel = kernel / np.sum(kernel)
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
    factors = np.random.uniform(0.6, 1.4, size=(1,1,3))
    distorted = img.astype(np.float32) * factors
    return np.clip(distorted, 0, 255).astype(np.uint8)

# --- Preparazione cartelle ---
for d in degradations:
    Path(os.path.join(output_dir, d)).mkdir(parents=True, exist_ok=True)

# --- Generazione dataset ---
for img_name in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)

    # salva immagine clean
    Image.fromarray(img).save(os.path.join(output_dir, "clean", img_name))

    # applica e salva tutte le degradazioni
    Image.fromarray(apply_blur(img)).save(os.path.join(output_dir, "blur", img_name))
    Image.fromarray(apply_noise(img)).save(os.path.join(output_dir, "noise", img_name))
    Image.fromarray(apply_low_light(img)).save(os.path.join(output_dir, "low_light", img_name))
    Image.fromarray(apply_jpeg(img)).save(os.path.join(output_dir, "jpeg", img_name))
    Image.fromarray(apply_pixelation(img)).save(os.path.join(output_dir, "pixelation", img_name))
    Image.fromarray(apply_motion_blur(img)).save(os.path.join(output_dir, "motion_blur", img_name))
    Image.fromarray(apply_high_light(img)).save(os.path.join(output_dir, "high_light", img_name))
    Image.fromarray(apply_low_contrast(img)).save(os.path.join(output_dir, "low_contrast", img_name))
    Image.fromarray(apply_color_distortion(img)).save(os.path.join(output_dir, "color_distortion", img_name))

print("Dataset sintetico generato con successo!")
