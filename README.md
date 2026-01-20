# Off-Road Semantic Scene Segmentation - Desert Environment
## Track 2: Duality AI Falcon Challenge

This repository contains repurposed notebooks for the Off-Road Semantic Scene Segmentation challenge using synthetic desert imagery from Duality AI's Falcon platform.

## 📋 Challenge Overview

**Objective:** Build a robust semantic segmentation model that generalizes across different synthetic desert environments.

**Key Requirements:**
- Train on one synthetic desert environment
- Evaluate on a previously unseen desert environment
- Test domain transfer and generalization capabilities

**Evaluation Metric:** Intersection over Union (IoU)

## 🎯 Target Classes (10 Total)

1. Trees
2. Lush Bushes
3. Dry Grass
4. Dry Bushes
5. Ground Clutter
6. Flowers
7. Logs
8. Rocks
9. Landscape
10. Sky

## 📁 Project Structure

```
.
├── 01_data_preprocessing_desert_segmentation.ipynb
├── 02_model_train_desert_segmentation.ipynb
├── README.md
├── checkpoints/          # Model checkpoints (created during training)
├── outputs/              # Training curves, predictions, etc.
├── falcon_desert_dataset/  # Dataset from Falcon (download separately)
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       └── images/
└── dataset_config.json   # Generated dataset configuration
```

## 🚀 Getting Started

### 1. Dataset Setup

1. Create a Falcon account:
   - Visit: https://falcon.duality.ai/auth/sign-up
   
2. Download the dataset:
   - Login and navigate to: https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert
   
3. Extract the dataset to `./falcon_desert_dataset/` directory

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib opencv-python pillow tqdm
pip install jupyter notebook
```

### 3. Running the Notebooks

**Step 1: Data Preprocessing**
```bash
jupyter notebook 01_data_preprocessing_desert_segmentation.ipynb
```

This notebook will:
- Load and explore the Falcon dataset
- Create custom Dataset classes
- Implement data augmentation
- Visualize samples and class distributions
- Generate dataset configuration

**Step 2: Model Training**
```bash
jupyter notebook 02_model_train_desert_segmentation.ipynb
```

This notebook will:
- Build U-Net architecture
- Implement combined loss function (CE + Dice)
- Train the model with validation
- Evaluate performance with IoU metrics
- Generate predictions for test set

## 🏗️ Model Architecture

### U-Net with Skip Connections

The implementation uses a U-Net architecture, which is particularly effective for semantic segmentation:

**Key Features:**
- **Encoder-Decoder Structure:** Captures context while preserving spatial information
- **Skip Connections:** Combines low-level and high-level features
- **Bilinear Upsampling:** Smooth upsampling in the decoder
- **Batch Normalization:** Improves training stability

**Architecture Details:**
- Input: RGB images (3 channels)
- Output: Per-pixel class predictions (10 channels)
- Total Parameters: ~31M trainable parameters

```
Input (3, 512, 512)
    ↓
Encoder (64 → 128 → 256 → 512 → 1024)
    ↓
Bottleneck (1024)
    ↓
Decoder (512 → 256 → 128 → 64)
    ↓
Output (10, 512, 512)
```

## 🎨 Data Augmentation Strategy

The preprocessing pipeline includes robust augmentation specifically designed for desert scenes:

### Training Augmentations
1. **Geometric Transformations:**
   - Random horizontal flips (50% probability)
   - Random vertical flips (50% probability)
   - Random 90° rotations (25% probability)

2. **Color Augmentations:**
   - Brightness adjustment (0.8-1.2×)
   - Contrast adjustment (0.8-1.2×)
   - Saturation adjustment (0.8-1.2×)
   - Hue adjustment (±0.1)

3. **Normalization:**
   - ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Why These Augmentations?
- **Flips & Rotations:** Desert terrain has no canonical orientation
- **Color Jittering:** Simulates different lighting conditions and times of day
- **Helps generalization** to the unseen test environment

## 📊 Loss Function

### Combined Cross-Entropy + Dice Loss

The model uses a combination of two complementary loss functions:

1. **Cross-Entropy Loss (50% weight)**
   - Standard pixel-wise classification loss
   - Good for well-separated classes
   - Handles class imbalance through optional weighting

2. **Dice Loss (50% weight)**
   - Measures overlap between prediction and ground truth
   - More robust to class imbalance
   - Particularly effective for segmentation tasks

**Formula:**
```
L_total = 0.5 × L_CE + 0.5 × L_Dice

where:
L_Dice = 1 - (2 × |X ∩ Y|) / (|X| + |Y|)
```

## 📈 Training Strategy

### Hyperparameters
- **Batch Size:** 8 (adjust based on GPU memory)
- **Learning Rate:** 1e-4 (AdamW optimizer)
- **Weight Decay:** 1e-5 (L2 regularization)
- **Epochs:** 50
- **Scheduler:** ReduceLROnPlateau (reduces LR when validation plateaus)

### Training Loop
1. Forward pass through the model
2. Calculate combined loss
3. Backward pass and optimization
4. Validate every epoch
5. Save best model based on validation IoU
6. Checkpoint every 5 epochs

### Best Practices
- **Early Stopping:** Monitor validation IoU, stop if no improvement
- **Learning Rate Scheduling:** Reduce LR by 0.5× after 5 epochs without improvement
- **Gradient Clipping:** Prevents exploding gradients (optional)

## 🔍 Evaluation Metrics

### Intersection over Union (IoU)

The primary evaluation metric for the challenge.

**Per-Class IoU:**
```
IoU_class = (Intersection) / (Union)
          = (TP) / (TP + FP + FN)
```

**Mean IoU (mIoU):**
```
mIoU = (1/N) × Σ IoU_i
```

where N is the number of classes.

### Why IoU?
- Measures both precision and recall
- Robust to class imbalance
- Standard metric for semantic segmentation
- Ranges from 0 (no overlap) to 1 (perfect overlap)

## 📊 Expected Results

### Baseline Performance
With the provided U-Net architecture and training strategy:

- **Training IoU:** 0.70-0.80
- **Validation IoU:** 0.65-0.75
- **Test IoU (unseen):** 0.60-0.70 (expect some drop due to domain shift)

### Class-Specific Performance
Expected easier classes:
- Sky (large, uniform regions)
- Landscape (dominant class)

Expected harder classes:
- Flowers (small, sparse objects)
- Ground Clutter (heterogeneous textures)

## 🛠️ Potential Improvements

### 1. Architecture Enhancements
- **DeepLabV3+:** Better at multi-scale features
- **SegFormer:** Transformer-based, state-of-the-art
- **Pretrained Encoders:** Use ResNet/EfficientNet pretrained on ImageNet

### 2. Loss Function Variations
```python
# Focal Loss (handles class imbalance)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Tversky Loss (adjustable FP/FN trade-off)
criterion = TverskyLoss(alpha=0.3, beta=0.7)

# Boundary Loss (emphasizes object boundaries)
criterion = CombinedLoss(ce_weight=0.4, dice_weight=0.4, boundary_weight=0.2)
```

### 3. Advanced Techniques
- **Test-Time Augmentation (TTA):** Average predictions over multiple augmented versions
- **Model Ensembling:** Combine predictions from multiple models
- **Pseudo-Labeling:** Use confident predictions on test set for semi-supervised learning
- **Post-Processing:** Conditional Random Fields (CRF) for smoothing

### 4. Domain Adaptation
Since training and test sets are from different desert environments:
- **Style Transfer:** Adapt test images to match training style
- **Domain-Invariant Features:** Add domain adversarial training
- **Self-Training:** Iteratively retrain on test predictions

## 📝 Deliverables Checklist

### A. Model Development & Training
- [x] Trained semantic segmentation model (U-Net)
- [x] Model weights saved in checkpoints/
- [x] Training scripts in notebook format
- [x] Clear training workflow documented

### B. Evaluation & Analysis
- [x] IoU metric implementation
- [x] Training curves (loss and IoU over epochs)
- [ ] Failure case analysis (TODO: analyze worst predictions)
- [ ] Class-wise performance breakdown

### C. Documentation & Reproducibility
- [x] Structured README with methodology
- [x] Clear instructions to reproduce results
- [x] Dataset configuration saved
- [ ] Final report (TODO: create detailed report)

## 🎓 Learning Resources

### Semantic Segmentation Fundamentals
- **Papers with Code:** https://paperswithcode.com/task/semantic-segmentation
- **YouTube Tutorial:** https://www.youtube.com/watch?v=FQ9YV7t7n1U

### Key Papers
1. **U-Net:** [Original Paper](https://arxiv.org/abs/1505.04597)
2. **DeepLabV3+:** [Paper](https://arxiv.org/abs/1802.02611)
3. **SegFormer:** [Paper](https://arxiv.org/abs/2105.15203)

### Additional Resources
- PyTorch Semantic Segmentation: https://pytorch.org/vision/stable/models.html#semantic-segmentation
- Albumentations (advanced augmentation): https://albumentations.ai/

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```python
# Solution: Reduce batch size
BATCH_SIZE = 4  # or even 2

# Or use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

**2. Model Not Learning**
- Check learning rate (try 1e-3 or 1e-5)
- Verify data normalization
- Ensure masks are in correct format (class indices, not RGB)

**3. Poor Generalization**
- Add more augmentation
- Increase weight decay
- Use dropout layers
- Reduce model complexity

**4. Class Imbalance**
```python
# Calculate class weights from training set
class_counts = ...  # count pixels per class
weights = 1.0 / class_counts
weights = weights / weights.sum() * len(weights)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

## 📧 Contact & Support

For issues or questions:
- Falcon Platform: https://falcon.duality.ai
- Competition Forum: [Link to forum if available]

## 📄 License

This project is for educational and competition purposes.

## 🙏 Acknowledgments

- Duality AI for providing the Falcon platform and synthetic dataset
- PyTorch community for excellent documentation
- U-Net authors for the foundational architecture

---

**Good luck with the competition! 🚀**

*Remember: The key to winning is not just having the best model, but understanding the data, documenting your process clearly, and being able to explain your methodology.*
