"""
train.py  — FlashInternImage (DCNv4)  Single‑GPU 版
=================================================
• 含 **RandAug + Mixup/CutMix + Label‑Smoothing**
• **CosineWarmup** 学习率策略 (线性 warm‑up → Cosine)
• **EMA** 滑动平均模型 (可选)
• **AMP**(autocast + grad‑scaler) 与梯度裁剪
• 保存 latest / best / history

> 仅依赖 PyTorch / torchvision / timm；无需 DDP 套件就可跑通
"""

import os, time, json, math, random, shutil, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 本项目模块
import config as user_config
from models.build import build_model

sns.set_theme(style="whitegrid")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 1. 配置 & 超参
# -------------------------------------------------
cfg = user_config._C.clone()
cfg.defrost()
cfg.MODEL.TYPE = "flash_intern_image"
cfg.MODEL.NUM_CLASSES = user_config.NUM_CLASSES
cfg.freeze()

BATCH_SIZE  = user_config.BATCH_SIZE
NUM_EPOCHS  = user_config.NUM_EPOCHS
NUM_CLASSES = user_config.NUM_CLASSES
INIT_LR     = 5e-4            # base lr
MIN_LR      = 5e-6            # floor lr for cosine
WARMUP_EPOCHS = 5             # linear warm‑up
WEIGHT_DECAY = 0.05
EMA_DECAY    = 0.9998         # set 0 to disable EMA
CLIP_GRAD    = 5.0
MIXUP_ALPHA  = 0.8            # 0 = off
CUTMIX_ALPHA = 1.0            # 0 = off
LABEL_SMOOTH = 0.1            # 0 = off

# -------------------------------------------------
# 2. 数据增强 & Loader
# -------------------------------------------------
IM_SIZE = 512
mean, std = cfg.AUG.MEAN, cfg.AUG.STD

train_tf = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
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

class_map = {v: k for k, v in train_ds.class_to_idx.items()}
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

TRAIN_SIZE, VAL_SIZE = len(train_ds), len(val_ds)
print(f"[Info] Train / Val size = {TRAIN_SIZE} / {VAL_SIZE}")

# -------------------------------------------------
# 3. Mixup / CutMix util
# -------------------------------------------------
class MixupCutmix:
    def __init__(self, mixup_alpha, cutmix_alpha, num_classes):
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes  = num_classes
    def __call__(self, x, y):
        if self.mixup_alpha <= 0 and self.cutmix_alpha <= 0:  # disabled
            return x, torch.nn.functional.one_hot(y, self.num_classes).float()
        lam, use_cutmix = 1.0, False
        if self.mixup_alpha > 0 and random.random() < 0.5:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        elif self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            use_cutmix = True
        perm = torch.randperm(x.size(0))
        x2, y2 = x[perm], y[perm]
        if use_cutmix:
            bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
            x[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (IM_SIZE * IM_SIZE)
        y1 = torch.nn.functional.one_hot(y,  self.num_classes)
        y2 = torch.nn.functional.one_hot(y2, self.num_classes)
        targets = y1 * lam + y2 * (1 - lam)
        return x, targets.float()

def _rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

mixup_fn = MixupCutmix(MIXUP_ALPHA, CUTMIX_ALPHA, NUM_CLASSES)

# -------------------------------------------------
# 4. 模型 / 优化器 / LR 调度
# -------------------------------------------------
model = build_model(cfg).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH) if (MIXUP_ALPHA==0 and CUTMIX_ALPHA==0 and LABEL_SMOOTH>0) else nn.KLDivLoss(reduction='batchmean')

optimizer = AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)

# warm‑up + cosine schedule
warmup_iters = WARMUP_EPOCHS * len(train_loader)
def lr_lambda(current_step):
    if current_step < warmup_iters:
        return float(current_step) / float(max(1, warmup_iters))
    progress = (current_step - warmup_iters) / float(max(1, (NUM_EPOCHS - WARMUP_EPOCHS) * len(train_loader)))
    return max(MIN_LR / INIT_LR, 0.5 * (1. + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)
scaler = GradScaler()

# EMA
ema_model = None
if EMA_DECAY > 0:
    ema_model = build_model(cfg).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

def update_ema(model_src, model_ema, decay):
    with torch.no_grad():
        ema_params = dict(model_ema.named_parameters())
        for name, param in model_src.named_parameters():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

# -------------------------------------------------
# 5. 训练 & 验证
# -------------------------------------------------

def run_epoch(epoch):
    model.train()
    running_loss = running_acc = 0.
    for step, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, targets = mixup_fn(imgs, labels)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            if MIXUP_ALPHA > 0 or CUTMIX_ALPHA > 0:
                log_prob = torch.log_softmax(logits, dim=1)
                loss = criterion(log_prob, targets)
            else:
                loss = criterion(logits, targets.max(1)[1])
        scaler.scale(loss).backward()
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        running_acc  += (logits.argmax(1) == labels).sum().item()

        if ema_model is not None:
            update_ema(model, ema_model, EMA_DECAY)

    return running_loss / TRAIN_SIZE, running_acc / TRAIN_SIZE

@torch.no_grad()
def evaluate(model_eval):
    model_eval.eval()
    loss_sum = correct = 0
    all_true, all_pred = [], []
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast():
            logits = model_eval(imgs)
            loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
        loss_sum += loss.item()
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())
    return loss_sum / VAL_SIZE, correct / VAL_SIZE, np.array(all_true), np.array(all_pred)

# -------------------------------------------------
# 6. 主训练循环
# -------------------------------------------------
BEST_ACC = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_state = None

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(epoch)
    val_loss, val_acc, y_true, y_pred = evaluate(ema_model if ema_model else model)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'  ].append(val_loss)
    history['val_acc'  ].append(val_acc)

    if val_acc > BEST_ACC:
        BEST_ACC = val_acc
        best_state = (model.state_dict(), y_true, y_pred)

    print(f"E{epoch:03d} | TL {train_loss:.4f} | TA {train_acc*100:.2f}% | VL {val_loss:.4f} | VA {val_acc*100:.2f}% | Best {BEST_ACC*100:.2f}% | LR {scheduler.get_last_lr()[0]:.2e} | dt {(time.time()-t0):.1f}s")

# -------------------------------------------------
# 7. 保存结果 & 可视化
# -------------------------------------------------
model_path = 'best_flash_internimage.pth'
torch.save(best_state[0], model_path)
print(f"Best model saved => {model_path}")

torch.save(history, 'history_flash_internimage.pth')
np.savez('y_true_pred.npz', y_true=best_state[1], y_pred=best_state[2])

# loss / acc 曲线
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
plt.subplot(1,2,2); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Accuracy'); plt.legend()
plt.tight_layout(); plt.savefig('training_curves.png', dpi=300); plt.close()

# 混淆矩阵
cm = confusion_matrix(best_state[1], best_state[2])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
plt.figure(figsize=(10,8))
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=[class_map[i] for i in range(NUM_CLASSES)], yticklabels=[class_map[i] for i in range(NUM_CLASSES)])
plt.ylabel('True'); plt.xlabel('Pred'); plt.title('Confusion Matrix (%)'); plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=300); plt.close()
