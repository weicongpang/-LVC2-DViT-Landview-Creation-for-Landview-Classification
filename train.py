# train_flash_internimage.py
import os, time, json, datetime, random, math, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config as user_config          # <-- 保留你的全局配置
from models.build import build_model   # <-- Flash-InternImage 的构建函数

sns.set_theme(style='whitegrid')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 1. 配置与超参
# --------------------------------------------------
cfg = user_config._C.clone()
cfg.defrost()
cfg.MODEL.TYPE = "flash_intern_image"
cfg.MODEL.NUM_CLASSES = user_config.NUM_CLASSES
cfg.freeze()

IM_SIZE      = 384                          # ←① 改成 384
BATCH_SIZE   = user_config.BATCH_SIZE
NUM_EPOCHS   = user_config.NUM_EPOCHS
NUM_CLASSES  = user_config.NUM_CLASSES
INIT_LR      = 1e-3                          # ←② 学习率调回 1e-3
WEIGHT_DECAY = 1e-2
CLIP_GRAD    = 1.0
scaler       = GradScaler()

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# --------------------------------------------------
# 2. 数据加载与增强  (完全复刻原脚本)
# --------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tf = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.ImageFolder(user_config.TRAIN_DATASET_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(user_config.VALID_DATASET_DIR, transform=val_tf)

class_map    = {v: k for k, v in train_ds.class_to_idx.items()}
TRAIN_SIZE   = len(train_ds)
VAL_SIZE     = len(val_ds)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"[Info] Train / Val size = {TRAIN_SIZE} / {VAL_SIZE}")

# --------------------------------------------------
# 3. 模型 & 优化器
# --------------------------------------------------
model     = build_model(cfg).to(DEVICE)      # Flash-InternImage
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
# ⚠️ 去掉 CosineAnnealingLR，先用恒定 LR；若验证不稳再按需加回

# --------------------------------------------------
# 4. 训练 & 验证
# --------------------------------------------------
def run_epoch():
    model.train()
    total_loss = total_correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        scaler.step(optimizer); scaler.update()

        total_loss   += loss.item() * imgs.size(0)
        total_correct+= (logits.argmax(1) == labels).sum().item()

    return total_loss / TRAIN_SIZE, total_correct / TRAIN_SIZE

@torch.no_grad()
def evaluate():
    model.eval()
    loss_sum = correct = 0
    y_true, y_pred = [], []
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        loss_sum += loss.item() * imgs.size(0)
        preds     = logits.argmax(1)
        correct  += (preds == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    return loss_sum / VAL_SIZE, correct / VAL_SIZE, np.array(y_true), np.array(y_pred)

# --------------------------------------------------
# 5. 主循环
# --------------------------------------------------
BEST_ACC, best_epoch = 0., 0
history, best_state  = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}, None
print(f"[Info] Start training  |  LR={INIT_LR}  BS={BATCH_SIZE}  Epochs={NUM_EPOCHS}")

for epoch in range(1, NUM_EPOCHS+1):
    t0 = time.time();  print(f"\n=== Epoch [{epoch}/{NUM_EPOCHS}] ===")
    tr_loss, tr_acc = run_epoch()
    val_loss, val_acc, y_t, y_p = evaluate()

    history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss);  history['val_acc'].append(val_acc)

    if val_acc > BEST_ACC:
        BEST_ACC  = val_acc;  best_epoch = epoch
        best_state= (model.state_dict(), y_t, y_p)
        torch.save(model.state_dict(), "best_flash_internimage.pth")
        print(f"*** New Best Acc: {BEST_ACC*100:.2f}% ***")

    print(f"Train Loss {tr_loss:.4f}  Acc {tr_acc*100:.2f}%")
    print(f"Val   Loss {val_loss:.4f}  Acc {val_acc*100:.2f}%  |  Best {BEST_ACC*100:.2f}% @Ep{best_epoch}")
    print(f"Time  {(time.time()-t0):.1f}s")

# --------------------------------------------------
# 6. 结果保存（曲线 / CM 与原脚本一致，可复用）
# --------------------------------------------------
torch.save(history, 'history_flash_internimage.pth')
np.savez('y_true_pred.npz', y_true=best_state[1], y_pred=best_state[2])

# （绘图函数略，与原相同）
print(f"\n[Complete] Best Val Acc = {BEST_ACC*100:.2f}%  (Epoch {best_epoch})")

# Plot training curves (based on original approach)
def plot_curves(hist, save='training_curves.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(hist['train_loss'], label='Train Loss')
    ax1.plot(hist['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(hist['train_acc'], label='Train Acc')
    ax2.plot(hist['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cm(y_true, y_pred, save='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f',
                xticklabels=[class_map[i] for i in range(len(class_map))],
                yticklabels=[class_map[i] for i in range(len(class_map))],
                cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.close()

def plot_distribution(y_true, save='class_distribution.png'):
    counts = np.bincount(y_true, minlength=len(class_map))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(counts)), counts, tick_label=[class_map[i] for i in range(len(counts))])
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h, str(int(h)), ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.ylabel('#Samples')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.close()

# Generate plots
plot_curves(history)
plot_cm(best_state[1], best_state[2])
plot_distribution(best_state[1])

print(f"Training history saved to: history_flash_internimage.pth")
print(f"Best model saved to: best_flash_internimage.pth")
print(f"Visualization results saved to: training_curves.png, confusion_matrix.png, class_distribution.png")
print("\nTraining Completed, already saved in best_flash_internimage.pth")
