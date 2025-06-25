import os, json, time
import numpy as np
import torch
from models.build import build_model  
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- 1. 常量（保持与 train.py 同步） ----------
BATCH_SIZE         = 8
IM_SIZE            = 512
MEAN, STD          = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TEST_DATASET_DIR   = '/root/autodl-tmp/Dataset-6-15/test'
# WEIGHTS_PATH       = '/root/Water_Resource/best_flash_internimage.pth'        # 训练脚本保存的权重
WEIGHTS_PATH       = '/root/Water_Resource/logs/resnet18_06-25-1-50epoch/best_resnet18.pth'        # 训练脚本保存的权重
CLASS_MAP_PATH     = 'classification_to_name.json'       # 类别映射（idx ➜ 中文/英文名）

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True    # 启用卷积算法自动调优

# ---------- 2. 数据预处理 ----------
test_transforms = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ---------- 3. 数据集 & DataLoader ----------
test_set = datasets.ImageFolder(TEST_DATASET_DIR, transform=test_transforms)
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ---------- 4. 加载类别名称 ----------
with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
    class_map = json.load(f)
num_classes = len(class_map)


model = build_model().to(DEVICE)
model.eval()

# ---------- 6. 加载权重 ----------
ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
# 训练脚本保存时用了 {"state_dict": ...}
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
model.load_state_dict(state_dict)
print(f'Loaded weights from: {WEIGHTS_PATH}')

# ---------- 7. 评估函数 ----------
def evaluate():
    y_true, y_pred = [], []
    total_per_class   = [0] * num_classes
    correct_per_class = [0] * num_classes

    model.eval()
    start = time.time()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs  = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(imgs)
            preds  = logits.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            for t, p in zip(labels, preds):
                total_per_class[t]   += 1
                correct_per_class[t] += int(t == p)

    elapsed = time.time() - start
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)

    # ---- 整体指标 ----
    oa    = accuracy_score(y_true, y_pred)
    class_accs = [
        np.mean(y_pred[y_true == i] == y_true[y_true == i])
        for i in range(num_classes) if np.sum(y_true == i) > 0
    ]
    macc  = np.mean(class_accs)
    kappa = cohen_kappa_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec   = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # ---- 打印 ----
    print(f"\nTest completed. Samples: {len(y_true)}  Time: {elapsed:.2f}s")
    print(f"Overall Accuracy (OA) : {oa:.4f}")
    print(f"Mean Accuracy (mAcc)  : {macc:.4f}")
    print(f"Cohen-Kappa           : {kappa:.4f}")
    print(f"Precision (macro)     : {prec:.4f}")
    print(f"Recall    (macro)     : {rec:.4f}")
    print(f"F1-score  (macro)     : {f1:.4f}")

    print("\nPer-class accuracy:")
    for idx in range(num_classes):
        total   = total_per_class[idx]
        correct = correct_per_class[idx]
        acc_cls = 100.0 * correct / total if total else 0.0
        cls_name = class_map.get(str(idx), f'class_{idx}')
        print(f"  [{idx:02d}] {cls_name:<20s}: {acc_cls:6.2f}%  ({correct}/{total})")

# ---------- 8. 主入口 ----------
if __name__ == '__main__':
    evaluate()
