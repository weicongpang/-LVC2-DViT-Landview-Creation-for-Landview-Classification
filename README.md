# LVC2-DViT: LandView Creation for Landview Classification

## Project Introduction

Remote sensing land-cover classification is impeded by limited annotated data and pronounced geometric distortion, hindering its value for environmental monitoring and land planning. We introduce LVC2-DViT (Landview Creation for Landview Classification with Deformable Vision Transformer), an end - to- end framework evaluated on five Aerial Image Dataset (AID) scene types, including Beach, Bridge, Pond, Port and River. LVC2-DViT fuses two modules: (i) a data creation pipeline that converts ChatGPT-4o- generated textual scene descriptions into class- balanced, high-fidelity images via Stable Diffusion, and (ii) DViT, a deformation - aware Vision Transformer dedicated to land - use 
classification whose adaptive receptive fields more faithfully model irregular landform geometries. Without increasing model size, LVC2 -DViT improves Overall Accuracy by 2.13 percentage points and Cohen’s Kappa by 2.66 percentage points over a strong vanilla ViT baseline, and also surpasses FlashAttention variant. These results confirm the effectiveness of combining generative augmentation with deformable attention for robust landuse mapping. 

## Project Structure

```
-LVC2-DViT-Landview-Creation-for-Landview-Classification/
├── config.py                          # Global configuration file
├── train.py                           # Training script
├── test.py                            # Testing script
├── classification_to_name.json        # Class name mapping file
├── vit_train_val_test.ipynb           # Vision Transformer Train, Valid, and Test
├── models/                            # Model definition directory
│   ├── build.py                       # Model Parameters Definition
│   ├── flash_intern_image.py          # FlashInternImage model
│   ├── intern_image.py                # InternImage model
│   ├── vit_dcnv4.py                   # DViT model
│   └── DCNv4/                         # DCNv4 operation module
├── DCNv4_op/                          # Functions of DCNv4
├── ops_dcnv3/                         # Functions of DCNv3
├── environment.yaml                   # Project's environment setup
```


## Supported Models

1. **Vision Transformer**
2. **FlashInternImage**
3. **DViT (Deformable Vision Transformer)**



## Environment Requirements

- Python 3.12+
- PyTorch 2.5.1+cu124
- Other dependencies: torchvision, timm, matplotlib, seaborn, sklearn, PIL, opencv-python

## Installation Steps

1. **Clone the project**
```bash
git clone https://github.com/weicongpang/-LVC2-DViT-Landview-Creation-for-Landview-Classification.git
cd -LVC2-DViT-Landview-Creation-for-Landview-Classification
```

2. **Install relevant dependencies**

Please follow environment.yaml to install relevant packages. 
```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn pillow opencv-python
```

3. **Train and Test** 
Open ```models/build.py```, modify this code ```model_type = 'flash_intern_image' ``` based on the model you want to train.
You can also modify the training parameters in ```models/build.py```.

Modify the following configurations based on the path of your dataset and the path of your weights file.
You can specify the directories that you want to save your results. 
```
TEST_DATASET_DIR = '/root/autodl-tmp/Dataset-6-15/test'
WEIGHTS_PATH = '/root/Water_Resource/train_tasks/run_flashinternimage_20250703_112824/checkpoint_epoch_50.pth'
CLASS_MAP_PATH = '/root/Water_Resource/classification_to_name.json'
RESULTS_DIR = '/root/Water_Resource/test_tasks/flashinternimage_20250708'
CM_FILENAME = 'confusion_matrix_flashinternimage.png'   # Confusion Matrix Filename
RESULTS_FILENAME = 'results.txt'                # Results Filename
```

Run ```python train.py``` for training.

During training, the script will:
- Automatically create timestamped result directories
- Save the best model weights
- Periodically save checkpoints
- Record training logs
- Generate training curve plots


Run ```python test.py``` for testing. 

The testing script will:
- Load trained model weights
- Evaluate performance on test set
- Generate confusion matrices
- Calculate various classification metrics
- Save detailed result reports

For Vision Transformer, you should go into vit_train_val_test.ipynb to start the training and testing by using Vision Transformer model. 

## Pretrained Models
We also provided our proposed pretrained checkpoint DViT model, which is available at *[Checkpoint Model](https://drive.google.com/file/d/14cEaFWWmT0-B8wrZdbLROYxWx3HjpTdJ/view?usp=sharing)*.

## Performance Metrics

The testing script calculates the following metrics:
- **Overall Accuracy (OA)**: Overall accuracy
- **Mean Accuracy (mAcc)**: Mean class accuracy
- **Cohen-Kappa**: Kappa coefficient
- **Precision/Recall/F1-score**: Precision/Recall/F1-score
- **Per-class Accuracy**: Accuracy for each class






