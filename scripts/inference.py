import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================================================
# CONFIG
# ==========================================================
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    'blur',
    'clean',
    'color_distortion',
    'high_light',
    'jpeg',
    'low_contrast',
    'low_light',
    'motion_blur',
    'noise',
    'pixelation'
]

# ==========================================================
# TRANSFORM (uguale a val/test)
# ==========================================================
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# ==========================================================
# MODEL
# ==========================================================
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

# ==========================================================
# INFERENCE
# ==========================================================
def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return CLASSES[pred.item()], conf.item()

def predict_folder(model, folder_path):
    print("\n📂 Inference su cartella:\n")
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(folder_path, file)
            label, conf = predict_image(model, path)
            print(f"{file:30s} → {label:15s} ({conf*100:.2f}%)")

# ==========================================================
# INTERACTIVE MAIN
# ==========================================================
def main():
    print("\n🔍 IMAGE DEGRADATION CLASSIFIER – INFERENCE MODE")
    print("--------------------------------------------------")
    print(f"Using device: {DEVICE}\n")

    # ---- modello ----
    model_path = input("📦 Inserisci il path del modello (.pth):\n> ").strip()
    model = load_model(model_path)

    print("\n✅ Modello caricato correttamente.\n")

    # ---- tipo di inference ----
    print("Scegli modalità di inference:")
    print("1 → Singola immagine")
    print("2 → Cartella di immagini")

    choice = input("\n> ").strip()

    if choice == "1":
        image_path = input("\n📸 Inserisci path dell'immagine:\n> ").strip()

        if not os.path.exists(image_path):
            print("❌ Immagine non trovata.")
            return

        label, conf = predict_image(model, image_path)

        print("\n🧠 RISULTATO INFERENCE")
        print("---------------------")
        print(f"Classe predetta : {label}")
        print(f"Confidenza      : {conf*100:.2f}%")

    elif choice == "2":
        folder_path = input("\n📂 Inserisci path della cartella:\n> ").strip()

        if not os.path.isdir(folder_path):
            print("❌ Cartella non trovata.")
            return

        predict_folder(model, folder_path)

    else:
        print("❌ Scelta non valida.")

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    main()
