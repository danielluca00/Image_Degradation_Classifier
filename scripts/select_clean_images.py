import os
import random
import shutil
from pathlib import Path

IMAGENET_DIR = r"C:\Users\dani2\Downloads\archive"  # Cartella radice delle classi
OUTPUT_DIR = "./few_clean_images"   # Dove copiare le immagini
NUM_CLASSES = 10
IMAGES_PER_CLASS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ottieni tutte le classi (cartelle)
all_classes = [p for p in Path(IMAGENET_DIR).iterdir() if p.is_dir()]
random.shuffle(all_classes)

selected_classes = all_classes[:NUM_CLASSES]

for cls in selected_classes:
    images = list(cls.glob("*.JPEG")) + list(cls.glob("*.jpg"))
    random.shuffle(images)
    selected_imgs = images[:IMAGES_PER_CLASS]

    for img_path in selected_imgs:
        shutil.copy(img_path, Path(OUTPUT_DIR))
