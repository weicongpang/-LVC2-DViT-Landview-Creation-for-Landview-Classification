# LVC2-DViT: LandView Creation for Landview Classification

## Project Introduction

Remote sensing land-cover classification is impeded by limited annotated data and pronounced geometric distortion, hindering its value for environmental monitoring and land planning. We introduce LVC2-DViT (Landview Creation for Landview Classification with Deformable Vision Transformer), an end - to- end framework evaluated on five Aerial Image Dataset (AID) scene types, including Beach, Bridge, Pond, Port and River. LVC2-DViT fuses two modules: (i) a data creation pipeline that converts ChatGPT-4o- generated textual scene descriptions into class- balanced, high-fidelity images via Stable Diffusion, and (ii) DViT, a deformation - aware Vision Transformer dedicated to land - use 
classification whose adaptive receptive fields more faithfully model irregular landform geometries. Without increasing model size, LVC2 -DViT improves Overall Accuracy by 2.13 percentage points and Cohen’s Kappa by 2.66 percentage points over a strong vanilla ViT baseline, and also surpasses FlashAttention variant. These results confirm the effectiveness of combining generative augmentation with deformable attention for robust landuse mapping. 

## Project Structure

```
Water_Resource/
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
```

## Core Files Description

### Configuration Files
- **`config.py`**: Global configuration management, containing all parameter settings for data, models, training, etc.
- **`configs/`**: Contains YAML configuration files for different model architectures

### Training Related
- **`train.py`**: Main training script, supporting model training, validation, and saving best models
- **`vit_train_val_test.ipynb`**: Jupyter notebook provided for Vision Transformer Training and Testing

### Testing Related
- **`test.py`**: Model testing script, generating confusion matrices and calculating various metrics
- **`classification_to_name.json`**: Class ID to name mapping file

### Model Definitions
- **`models/build.py`**: Model builder, creating different architecture models based on configuration
- **`models/flash_intern_image.py`**: FlashInternImage model implementation
- **`models/intern_image.py`**: InternImage model implementation
- **`models/vit_dcnv4.py`**: ViT-DCNv4 hybrid model implementation

### CUDA Extensions
- **`DCNv4_op/`**: DCNv4 CUDA implementation, providing deformable convolution operations
- **`ops_dcnv3/`**: DCNv3 CUDA implementation

## Supported Models

1. **FlashInternImage**: Advanced image classification model based on DCNv4
2. **InternImage**: Classic image classification model based on DCNv3
3. **ViT-DCNv4**: Hybrid model combining Vision Transformer and DCNv4

## Dataset Format

The project uses ImageFolder format datasets with the following directory structure:
```
Dataset-6-15/
├── train/
│   ├── Beach/
│   ├── Bridge/
│   ├── Pond/
│   ├── Port/
│   └── River/
├── val/
│   ├── Beach/
│   ├── Bridge/
│   ├── Pond/
│   ├── Port/
│   └── River/
└── test/
    ├── Beach/
    ├── Bridge/
    ├── Pond/
    ├── Port/
    └── River/
```

## Environment Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU training)
- Other dependencies: torchvision, timm, matplotlib, seaborn, sklearn, PIL, opencv-python

## Installation Steps

1. **Clone the project**
```bash
git clone <repository_url>
cd Water_Resource
```

2. **Install dependencies**
```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn pillow opencv-python
```

3. **Compile CUDA extensions** (optional, for DCNv4/DCNv3)
```bash
cd DCNv4_op
bash make.sh
cd ../ops_dcnv3
bash make.sh
```

## Usage Guide

### 1. Data Preparation

Organize the dataset in ImageFolder format and update the paths in the configuration file:

```python
# Modify in config.py
TRAIN_DATASET_DIR = '/path/to/your/dataset/train'
VALID_DATASET_DIR = '/path/to/your/dataset/val'
TEST_DATASET_DIR = '/path/to/your/dataset/test'
```

### 2. Model Training

```bash
# Train with default configuration
python train.py

# Training results will be saved in train_tasks/ directory
```

During training, the script will:
- Automatically create timestamped result directories
- Save the best model weights
- Periodically save checkpoints
- Record training logs
- Generate training curve plots

### 3. Model Testing

```bash
# Test model performance
python test.py
```

The testing script will:
- Load trained model weights
- Evaluate performance on test set
- Generate confusion matrices
- Calculate various classification metrics
- Save detailed result reports

### 4. Using Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook vit_train_val_test.ipynb
```

The notebook provides interactive training and testing functionality.

## Configuration Guide

### Training Parameters (config.py)

```python
# Basic training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-4
IM_SIZE = 512

# Data augmentation
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
```

### Model Selection (models/build.py)

```python
# Modify model_type in build.py to select different models
model_type = 'flash_intern_image'  # Options: 'intern_image', 'vit_dcnv4'
```

## Output Results

### Training Output
- **Best Model**: `best_flashinternimage.pth`
- **Checkpoints**: `checkpoint_epoch_X.pth`
- **Training Log**: `train_log.txt`
- **Training Curves**: `curves.png`

### Testing Output
- **Confusion Matrix**: `confusion_matrix_flashinternimage.png`
- **Normalized Confusion Matrix**: `normalized_confusion_matrix_flashinternimage.png`
- **Detailed Results**: `results.txt`

## Performance Metrics

The testing script calculates the following metrics:
- **Overall Accuracy (OA)**: Overall accuracy
- **Mean Accuracy (mAcc)**: Mean class accuracy
- **Cohen-Kappa**: Kappa coefficient
- **Precision/Recall/F1-score**: Precision/Recall/F1-score
- **Per-class Accuracy**: Accuracy for each class

## Important Notes

1. **GPU Memory**: Adjust BATCH_SIZE based on GPU VRAM
2. **Data Paths**: Ensure dataset paths are correct
3. **Model Weights**: Ensure weight file paths are correct before testing
4. **CUDA Extensions**: Compile CUDA extensions first if using DCNv4/DCNv3

## Troubleshooting

### Common Issues

1. **Insufficient CUDA Memory**
   - Reduce BATCH_SIZE
   - Lower image size IM_SIZE

2. **Model Loading Failure**
   - Check weight file paths
   - Confirm model architecture compatibility

3. **Data Loading Errors**
   - Check dataset paths
   - Confirm ImageFolder format is correct

## Contributing

Welcome to submit Issues and Pull Requests to improve the project.

## License

This project is open source under the MIT License. 