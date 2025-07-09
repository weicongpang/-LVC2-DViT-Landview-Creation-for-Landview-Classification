import os
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.utils.metrics import accuracy, AverageMeter
import matplotlib.pyplot as plt
import seaborn as sns
from models.build import build_model
from torchinfo import summary
from timm.optim import Lamb
from torch.optim import AdamW
from datetime import datetime

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)




# -------------------- Configuration and File Path --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_CLASSES = 5
LR = 1e-4
IM_SIZE = 512
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_DATASET_DIR = '/root/autodl-tmp/Dataset-6-15/train'
VALID_DATASET_DIR = '/root/autodl-tmp/Dataset-6-15/val'
RESULTS_ROOT = '/root/Water_Resource/train_tasks'
CHECKPOINT_INTERVAL = 25  




# -------------------- Data Preparation --------------------
def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(TRAIN_DATASET_DIR, transform)
    val_ds = datasets.ImageFolder(VALID_DATASET_DIR, transform)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader




# -------------------- Model, Loss, Optimizer --------------------
def get_model():
    model = build_model().to(DEVICE)
    return model

def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(model):
    # return Lamb(model.parameters(), lr=LR)
    return AdamW(model.parameters(), lr=LR)



# -------------------- Training & Validation --------------------
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss, running_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_correct / len(train_loader.dataset) * 100
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    model.eval()
    running_loss, running_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_correct / len(val_loader.dataset) * 100
    return val_loss, val_acc



# -------------------- Plotting --------------------
def plot_curves(history, out_path="curves.png"):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(len(history["train_loss"]))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], label="Train Acc@1")
    ax2.plot(epochs, history["val_acc"], label="Val Acc@1")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



# -------------------- Main --------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # 1. Create unique result folder
    timestamp = datetime.now().strftime("run_flashinternimage_%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    train_loader, val_loader = get_data_loaders()
    model = get_model()
    print("Model Summary:")
    summary(model, input_size=(1,3, IM_SIZE, IM_SIZE), device=str(DEVICE))
    criterion = get_criterion()
    optimizer = get_optimizer(model)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    log_path = os.path.join(result_dir, "train_log.txt")
    log_file = open(log_path, "w")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log_str = (
            f"Epoch {epoch:03d} | "
            f"Train Accuracy: {tr_acc:.2f}%, Train Loss: {tr_loss:.4f} | "
            f"Val Accuracy: {val_acc:.2f}%, Val Loss: {val_loss:.4f} | "
            f"Time: {dt:.1f}s"
        )
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(result_dir, "best_flashinternimage.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch
            }, best_model_path)
            print(f"*** New best Acc@1 {best_acc:.2f}% @ epoch {epoch} ***")
            log_file.write(f"*** New best Acc@1 {best_acc:.2f}% @ epoch {epoch} ***\n")
            log_file.flush()

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = os.path.join(result_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history
            }, ckpt_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
            log_file.write(f"Checkpoint saved at epoch {epoch+1}\n")
            log_file.flush()

    print(f"\nFinished. Best Acc@1 {best_acc:.2f}%")
    log_file.write(f"\nFinished. Best Acc@1 {best_acc:.2f}%\n")
    log_file.close()

    # Save curves
    curve_path = os.path.join(result_dir, "curves.png")
    plot_curves(history, out_path=curve_path)
    print(f"Curves saved at {curve_path}")
