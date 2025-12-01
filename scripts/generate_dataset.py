import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# --- Configurazione ---
input_dir = "./clean_images"        # cartella con immagini clean
output_dir = "./dataset"            # cartella dove salvare il dataset generato
img_size = (256, 256)               # ridimensiona tutte le immagini
noise_std = 25                       # deviazione standard per Gaussian noise
jpeg_quality = 20                     # qualità JPEG per artefatti
blur_kernel = (5, 5)                 # kernel per Gaussian blur

# Tipi di degradazioni
degradations = ["blur", "noise", "low_light", "jpeg", "pixelation", "clean"]

# --- Funzioni di degradazione ---
def apply_blur(img):
    return cv2.GaussianBlur(img, blur_kernel, 0)

def apply_noise(img):
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_low_light(img):
    factor = np.random.uniform(0.3, 0.7)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def apply_jpeg(img, quality=jpeg_quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def apply_pixelation(img):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (w//8, h//8), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

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

print("Dataset sintetico generato con successo!")
