import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import warnings
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix
)
from PIL import Image
from models.build import build_model
from einops import repeat


# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- Configuration and File Path --------------------
BATCH_SIZE = 8
IM_SIZE = 512
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TEST_DATASET_DIR = '/root/autodl-tmp/Dataset-6-15/test'
WEIGHTS_PATH = '/root/Water_Resource/train_tasks/run_flashinternimage_20250703_112824/checkpoint_epoch_50.pth'
CLASS_MAP_PATH = '/root/Water_Resource/classification_to_name.json'
RESULTS_DIR = '/root/Water_Resource/test_tasks/flashinternimage_20250708'
CM_FILENAME = 'confusion_matrix_flashinternimage.png'   # Confusion Matrix Filename
RESULTS_FILENAME = 'results.txt'                # Results Filename

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True



# -------------------- Data Preparation --------------------
def get_test_loader():
    test_transforms = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_set = datasets.ImageFolder(TEST_DATASET_DIR, transform=test_transforms)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader



def load_class_map():
    with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    return class_map



def load_model(num_classes):
    model = build_model().to(DEVICE)
    model.eval()
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f'Loaded weights from: {WEIGHTS_PATH}')
    return model


'''
For Vit-DCNv4 Only
'''
#
def generate_cam(model, img_tensor, class_idx=None):
    """
    Generate CAM for a single image tensor (shape: 1, 3, H, W)  CAM: Class Activation Map
    """
    model.eval()
    with torch.no_grad():
        # 1. Get backbone features
        feat = model.backbone(img_tensor)  # (1, 512, 32, 32)
        # 2. patch embedding
        patch_emb = model.to_patch_embedding(feat)  # (1, 1024, 512)
        b, n, c = patch_emb.shape
        # 3. Concatenate cls token
        cls_token = model.cls_token.expand(b, -1, -1)  # (1, 1, 512)
        x = torch.cat((cls_token, patch_emb), dim=1)   # (1, 1025, 512)
        x = x + model.pos_embedding[:, :n+1]
        x = model.dropout(x)
        # 4. transformer
        x = model.transformer(x)  # (1, 1025, 512)
        # 5. Take all patch tokens (excluding cls token)
        patch_tokens = x[:, 1:, :]  # (1, 1024, 512)
        # 6. Classification head
        logits = model.mlp_head(x[:, 0])  # (1, num_classes)
        pred = logits.argmax(dim=1).item() if class_idx is None else class_idx

        # 7. Get classification head weights
        weight = model.mlp_head[-1].weight  # (num_classes, 512)
        cam_weights = weight[pred].detach().cpu().numpy()  # (512,)

        # 8. Weighted sum of patch tokens
        patch_tokens = patch_tokens.squeeze(0).detach().cpu().numpy()  # (1024, 512)
        cam = np.dot(patch_tokens, cam_weights)  # (1024,)
        cam = cam.reshape(32, 32)
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        return cam, pred



def save_cam_figure(img_path, cam, save_path, alpha=0.4):
    """
    Save a 2x1 figure: top is original image, bottom is CAM overlay
    """
    img = Image.open(img_path).convert('RGB').resize((IM_SIZE, IM_SIZE))
    img_np = np.array(img)
    cam_resized = cv2.resize(cam, (IM_SIZE, IM_SIZE))
    cam_uint8 = np.uint8(255 * cam_resized)
    # Ensure C_CONTIGUOUS 2D uint8
    cam_uint8 = np.ascontiguousarray(cam_uint8)
    if cam_uint8.ndim == 3:
        cam_uint8 = cam_uint8[..., 0]
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    # Generate 2x1 subplot
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title('CAM Overlay')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_grad_cam_heatmap(image_path, model, device, save_path=None):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    original_img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(original_img)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Select hook point
    target_layers = [model.transformer.net.layers[-2][1].fn]  # FeedForward body
    # target_layers = [model.transformer.net.layers[-1][0].fn]  # SelfAttention body

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform
    )

    # Register debug hooks
    feature_maps = {}
    grads = {}

    def save_features_hook(module, input, output):
        feature_maps['value'] = output.detach().cpu()

    def save_grads_hook(module, grad_input, grad_output):
        grads['value'] = grad_output[0].detach().cpu()

    handle_feat = target_layers[0].register_forward_hook(save_features_hook)
    handle_grad = target_layers[0].register_full_backward_hook(save_grads_hook)

    with torch.no_grad():
        output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    targets = [ClassifierOutputTarget(pred_class)]

    cam_mask = cam(input_tensor, targets=targets)
    print("CAM before normalization min/max/mean:", cam_mask.min(), cam_mask.max(), cam_mask.mean())
    if cam_mask.ndim == 3:
        cam_mask = cam_mask[0]
    elif cam_mask.ndim == 2:
        pass
    else:
        raise ValueError(f"cam_mask shape abnormal: {cam_mask.shape}")
    print("CAM before normalization(2) min/max/mean:", cam_mask.min(), cam_mask.max(), cam_mask.mean())
    cam_mask = np.maximum(cam_mask, 0)
    cam_mask = cam_mask / (cam_mask.max() + 1e-8)
    print("CAM after normalization min/max/mean:", cam_mask.min(), cam_mask.max(), cam_mask.mean())

    # Print features and gradients
    if 'value' in feature_maps:
        print("Feature output min/max/mean:", feature_maps['value'].min().item(), feature_maps['value'].max().item(), feature_maps['value'].mean().item())
    if 'value' in grads:
        print("Gradient output min/max/mean:", grads['value'].min().item(), grads['value'].max().item(), grads['value'].mean().item())
    else:
        print("No gradient captured")

    # Unregister hooks
    handle_feat.remove()
    handle_grad.remove()

    cam_mask = cam_mask.astype(np.float32)
    cam_mask = cv2.resize(cam_mask, (224, 224), interpolation=cv2.INTER_LINEAR)
    mask_uint8 = np.uint8(255 * cam_mask)
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_uint8 = np.array(original_img.resize((224, 224))).astype(np.uint8)
    overlay = cv2.addWeighted(img_uint8, 0.4, heatmap, 0.6, 0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 16))
    axes[0].imshow(img_uint8)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title('CAM Overlay')
    axes[1].axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print("Heatmap generation completed!")



# -------------------- Evaluation --------------------
def evaluate(model, test_loader, class_map, results_dir):
    num_classes = len(class_map)
    y_true, y_pred = [], []
    total_per_class = [0] * num_classes
    correct_per_class = [0] * num_classes

    model.eval()
    start = time.time()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            for t, p in zip(labels, preds):
                total_per_class[t] += 1
                correct_per_class[t] += int(t == p)
    elapsed = time.time() - start
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(font_scale=1.2)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=[class_map.get(str(i), f'class_{i}') for i in range(num_classes)],
        yticklabels=[class_map.get(str(i), f'class_{i}') for i in range(num_classes)]
    )
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, CM_FILENAME)
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nConfusion matrix saved as {cm_path}')

    # Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(font_scale=1.2)
    ax2 = sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=[class_map.get(str(i), f'class_{i}') for i in range(num_classes)],
        yticklabels=[class_map.get(str(i), f'class_{i}') for i in range(num_classes)]
    )
    ax2.set_xlabel('Predicted Label', fontsize=14)
    ax2.set_ylabel('True Label', fontsize=14)
    ax2.set_title('Normalized Matrix Heatmap', fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_norm_path = os.path.join(results_dir, 'normalized_confusion_matrix_flashinternimage.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Normalized confusion matrix saved as {cm_norm_path}')

    # Metrics
    oa = accuracy_score(y_true, y_pred)
    class_accs = [
        np.mean(y_pred[y_true == i] == y_true[y_true == i])
        for i in range(num_classes) if np.sum(y_true == i) > 0
    ]
    macc = np.mean(class_accs)
    kappa = cohen_kappa_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division='warn')
    rec = recall_score(y_true, y_pred, average='macro', zero_division='warn')
    f1 = f1_score(y_true, y_pred, average='macro', zero_division='warn')

    # Save results to txt
    results_path = os.path.join(results_dir, RESULTS_FILENAME)
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Test completed. Samples: {len(y_true)}  Time: {elapsed:.2f}s\n")
        f.write(f"Overall Accuracy (OA) : {oa:.4f}\n")
        f.write(f"Mean Accuracy (mAcc)  : {macc:.4f}\n")
        f.write(f"Cohen-Kappa           : {kappa:.4f}\n")
        f.write(f"Precision (macro)     : {prec:.4f}\n")
        f.write(f"Recall    (macro)     : {rec:.4f}\n")
        f.write(f"F1-score  (macro)     : {f1:.4f}\n\n")
        f.write("Per-class accuracy:\n")
        for idx in range(num_classes):
            total = total_per_class[idx]
            correct = correct_per_class[idx]
            acc_cls = 100.0 * correct / total if total else 0.0
            cls_name = class_map.get(str(idx), f'class_{idx}')
            f.write(f"  [{idx:02d}] {cls_name:<20s}: {acc_cls:6.2f}%  ({correct}/{total})\n")
    print(f"\nResults saved as {results_path}")

    # Print results
    print(f"\nTest completed. Samples: {len(y_true)}  Time: {elapsed:.2f}s")
    print(f"Overall Accuracy (OA) : {oa:.4f}")
    print(f"Mean Accuracy (mAcc)  : {macc:.4f}")
    print(f"Cohen-Kappa           : {kappa:.4f}")
    print(f"Precision (macro)     : {prec:.4f}")
    print(f"Recall    (macro)     : {rec:.4f}")
    print(f"F1-score  (macro)     : {f1:.4f}")
    print("\nPer-class accuracy:")
    for idx in range(num_classes):
        total = total_per_class[idx]
        correct = correct_per_class[idx]
        acc_cls = 100.0 * correct / total if total else 0.0
        cls_name = class_map.get(str(idx), f'class_{idx}')
        print(f"  [{idx:02d}] {cls_name:<20s}: {acc_cls:6.2f}%  ({correct}/{total})")



# -------------------- Main --------------------
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_loader = get_test_loader()
    class_map = load_class_map()
    model = load_model(len(class_map))


    transform = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    evaluate(model, test_loader, class_map, RESULTS_DIR)