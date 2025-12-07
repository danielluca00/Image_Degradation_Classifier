import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# ==========================================================
# CREA CARTELLA DI OUTPUT PER QUESTA RUN
# ==========================================================
RUN_BASE = "runs"
os.makedirs(RUN_BASE, exist_ok=True)

RUN_DIR = os.path.join(RUN_BASE, datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S"))
os.makedirs(RUN_DIR, exist_ok=True)

# ==========================================================
# AUTO-LOGGING (terminal + file)
# ==========================================================
log_path = os.path.join(RUN_DIR, "training.log")

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

sys.stdout = Logger(log_path)

print(f"📄 Logging attivo → {log_path}")

# ==========================================================
# CONFIG
# ==========================================================
DATASET_DIR = "./dataset"
BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 15
LR = 1e-4
PATIENCE = 5  # early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE = os.path.join(RUN_DIR, "best_model.pth")

print("Using device:", DEVICE)

# ==========================================================
# TRANSFORM
# ==========================================================
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

# ==========================================================
# DATASET
# ==========================================================
train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), train_transform)
val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), val_test_transform)
test_ds  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), val_test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_ds.classes)

# ==========================================================
# MODELLO RESNET18 (Transfer Learning)
# ==========================================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================================
# METRICHE PER I GRAFICI
# ==========================================================
train_losses, val_losses = [], []
train_accs, val_accs = [], []
epoch_times = []

# ==========================================================
# FUNZIONI DI TRAIN / EVAL
# ==========================================================
def train_one_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# ==========================================================
# TRAINING LOOP CON EARLY STOPPING + TIMER
# ==========================================================
best_val_acc = 0
patience_counter = 0
training_start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n===== EPOCH {epoch}/{NUM_EPOCHS} =====")

    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate(val_loader)

    epoch_duration = time.time() - epoch_start
    epoch_times.append(epoch_duration)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
    print(f"⏱️  Tempo epoca: {epoch_duration:.2f} sec")

    if val_acc > best_val_acc:
        torch.save(model.state_dict(), MODEL_SAVE)
        best_val_acc = val_acc
        patience_counter = 0
        print("💾 Miglior modello salvato!")
    else:
        patience_counter += 1
        print(f"⏳ Early stopping counter: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\n🛑 EARLY STOPPING ATTIVATO")
            break

training_total_time = time.time() - training_start_time
print(f"\n⏱️ Tempo totale di training: {training_total_time:.2f} sec")
print("\nTraining terminato.\n")

# ==========================================================
# TEST FINALE + Confusion Matrix
# ==========================================================
model.load_state_dict(torch.load(MODEL_SAVE))
test_loss, test_acc = evaluate(test_loader)
print(f"🎉 TEST ACCURACY: {test_acc*100:.2f}%")

# ---- Predizioni per confusion matrix ----
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_ds.classes, yticklabels=train_ds.classes)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"))
plt.close()

# ==========================================================
# GRAFICI LOSS, ACCURACY, TEMPI
# ==========================================================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(RUN_DIR, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(RUN_DIR, "accuracy_curve.png"))
plt.close()

plt.figure()
plt.plot(epoch_times)
plt.title("Epoch Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Seconds")
plt.savefig(os.path.join(RUN_DIR, "epoch_times.png"))
plt.close()

print(f"\n📂 Tutti i risultati salvati in: {RUN_DIR}")
print(f"📄 Log completo: {log_path}\n")
