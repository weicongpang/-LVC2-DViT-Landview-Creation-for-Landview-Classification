import os, warnings

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)


import time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.optim import AdamW          
from timm.utils import accuracy, AverageMeter
import matplotlib.pyplot as plt
import seaborn as sns
from models.build import build_model
from torchsummary import summary


DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 8     
NUM_EPOCHS   = 20 
NUM_CLASSES  = 5     
LR           = 1e-4                        
IM_SIZE      = 512

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Save trained model
TRAINED_MODEL = '/root/Water_Resource/trained_models/data_record.pth'

# Dataset path
TRAIN_DATASET_DIR = '/root/autodl-tmp/Dataset-6-15/train_new'
VALID_DATASET_DIR = '/root/autodl-tmp/Dataset-6-15/valid_new'
TEST_DATASET_DIR = '/root/autodl-tmp/Dataset-6-15/test'


tfm = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
train_ds = datasets.ImageFolder(TRAIN_DATASET_DIR, tfm)
val_ds   = datasets.ImageFolder(VALID_DATASET_DIR, tfm)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=True)  
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True)


model = build_model().to(DEVICE)

print("Model Summary:")
summary(model, input_size=(3, IM_SIZE, IM_SIZE), device=str(DEVICE))



criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)   # 无正则


def train_one_epoch():
    model.train()
    running_loss, running_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss    += loss.item() * imgs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc  = running_correct / len(train_loader.dataset) * 100  
    return epoch_loss, epoch_acc

def validate():
    model.eval()
    running_loss, running_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss    += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_acc  = running_correct / len(val_loader.dataset) * 100
    return val_loss, val_acc


best_acc = 0.0
history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch()
    val_loss, val_acc = validate()
    dt = time.time() - t0

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:03d} | "
        f"Train Accuracy: {tr_acc:.2f}%, Train Loss: {tr_loss:.4f} | "
        f"Val Accuracy: {val_acc:.2f}%, Val Loss: {val_loss:.4f} | "
        f"Time: {dt:.1f}s"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "epoch": epoch
        }, "best_vit_dcnv4.pth")
        print(f"*** New best Acc@1 {best_acc:.2f}% @ epoch {epoch} ***")

print(f"\nFinished. Best Acc@1 {best_acc:.2f}%")


sns.set_theme(style="whitegrid")
def plot_curves(hist, out="curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(len(hist["train_loss"]))

    # Loss 曲线
    ax1.plot(epochs, hist["train_loss"], label="Train Loss")
    ax1.plot(epochs, hist["val_loss"],   label="Val Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy 曲线
    ax2.plot(epochs, hist["train_acc"], label="Train Acc@1")
    ax2.plot(epochs, hist["val_acc"],   label="Val Acc@1")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

plot_curves(history)
print("Curves saved.")
