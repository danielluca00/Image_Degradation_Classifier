import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from datetime import datetime

# ===============================
# AUTO-LOGGING: salva tutto l’output su file
# ===============================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("training_%Y-%m-%d_%H-%M-%S.log")
log_path = os.path.join(LOG_DIR, log_filename)

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# attiva logging globale
sys.stdout = Logger(log_path)

print(f"📄 Logging attivo → {log_path}")

# ===============================
# CONFIGURAZIONE
# ===============================
DATASET_DIR = "./dataset"
BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE = "image_degradation_classifier.pth"

print("Using device:", DEVICE)

# ===============================
# DATA AUGMENTATION
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

# ===============================
# LOAD DATASETS
# ===============================
train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), train_transform)
val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), val_test_transform)
test_ds  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), val_test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_ds.classes)

# ===============================
# MODEL: ResNet18 (Transfer Learning)
# ===============================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===============================
# TRAINING LOOP
# ===============================
def train_one_epoch(epoch):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ===============================
# TRAINING + VALIDATION
# ===============================
best_val_acc = 0

for epoch in range(1, NUM_EPOCHS+1):
    train_one_epoch(epoch)
    val_acc = evaluate(val_loader)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        torch.save(model.state_dict(), MODEL_SAVE)
        best_val_acc = val_acc
        print("💾 Model improved and saved.")

print("Training finished.")

# ===============================
# FINAL TEST ACCURACY
# ===============================
model.load_state_dict(torch.load(MODEL_SAVE))
test_acc = evaluate(test_loader)
print(f"\n🎉 FINAL TEST ACCURACY: {test_acc*100:.2f}%\n")

print(f"📄 Log salvato in: {log_path}")
