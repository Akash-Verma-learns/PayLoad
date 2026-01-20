# Quick Start Guide
## Off-Road Desert Semantic Segmentation Challenge

This guide will get you up and running in 15 minutes.

## ⚡ 5-Step Quick Start

### Step 1: Download Dataset (5 min)
1. Create Falcon account: https://falcon.duality.ai/auth/sign-up
2. Login and download dataset: https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert
3. Extract to `./falcon_desert_dataset/`

**Expected structure:**
```
falcon_desert_dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

### Step 2: Setup Environment (3 min)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run Data Preprocessing (2 min)
```bash
jupyter notebook 01_data_preprocessing_desert_segmentation.ipynb
```

**In the notebook:**
1. Update `DATA_ROOT` path to your dataset location
2. Run all cells (Ctrl+Enter for each cell, or Cell → Run All)
3. Verify visualizations look correct
4. Check that `dataset_config.json` is created

### Step 4: Train Model (1 min to start, hours to complete)
```bash
jupyter notebook 02_model_train_desert_segmentation.ipynb
```

**In the notebook:**
1. Verify configuration (batch size, epochs, etc.)
2. Uncomment the training loop section
3. Run training cells
4. Monitor training progress (loss curves, IoU)

**Training time estimates:**
- With GPU: ~2-4 hours for 50 epochs
- Without GPU: ~10-20 hours (not recommended)

### Step 5: Generate Predictions (2 min)
```bash
# Still in the training notebook
# Run the test set prediction cells
```

This will create predictions in `./outputs/test_predictions/`

## 🎯 What You'll Get

After completing all steps:

1. **Trained Model**
   - Best model checkpoint: `checkpoints/best_model.pth`
   - Training history: loss and IoU curves

2. **Evaluation Results**
   - Overall mIoU score
   - Per-class IoU breakdown
   - Training curves visualization

3. **Test Predictions**
   - Segmentation masks for all test images
   - Ready for submission

## 📊 Expected Timeline

| Task | Time | GPU Required |
|------|------|--------------|
| Dataset download | 5 min | No |
| Environment setup | 3 min | No |
| Data preprocessing | 2 min | No |
| Model training | 2-4 hours | Yes (recommended) |
| Evaluation | 5 min | Yes |
| Test prediction | 5 min | Yes |
| **Total** | **~3 hours** | **Yes** |

## 🚨 Common First-Time Issues

### Issue 1: "Dataset not found"
**Solution:** Update `DATA_ROOT` in preprocessing notebook
```python
DATA_ROOT = './falcon_desert_dataset'  # Adjust to your path
```

### Issue 2: "CUDA out of memory"
**Solution:** Reduce batch size
```python
BATCH_SIZE = 4  # or 2 if still failing
```

### Issue 3: "Module not found"
**Solution:** Ensure virtual environment is activated
```bash
source venv/bin/activate  # Activate first!
pip install -r requirements.txt
```

### Issue 4: No GPU available
**Solution:** Use Google Colab (free GPU)
1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU

## 🎓 Understanding the Code

### Key Components

**1. Dataset Class (`DesertSegmentationDataset`)**
- Loads images and masks
- Applies augmentation
- Returns tensors ready for training

**2. U-Net Model**
- Encoder: Downsamples and extracts features
- Decoder: Upsamples and generates segmentation
- Skip connections: Preserves spatial details

**3. Training Loop**
- Forward pass: Image → Model → Predictions
- Loss calculation: Compare predictions to ground truth
- Backward pass: Update model weights
- Validation: Evaluate on validation set

**4. Metrics**
- IoU: Measures prediction quality
- mIoU: Average IoU across all classes

## 🔧 Quick Customization

### Change Image Size
```python
# In both notebooks
HEIGHT = 256  # Smaller = faster training
WIDTH = 256
```

### Adjust Training Duration
```python
NUM_EPOCHS = 30  # Reduce for faster results
```

### Modify Augmentation
```python
# In JointTransform class
if np.random.rand() > 0.3:  # Change probability
    # Apply augmentation
```

## 📈 Monitoring Training

**Good signs:**
- ✅ Training loss decreasing
- ✅ Validation IoU increasing
- ✅ Gap between train/val not too large

**Warning signs:**
- ⚠️ Validation loss increasing (overfitting)
- ⚠️ No improvement after many epochs (plateau)
- ⚠️ Very large train/val gap (overfitting)

**Solutions:**
- Add more augmentation
- Reduce model complexity
- Use early stopping
- Increase weight decay

## 🎯 Submission Checklist

Before submitting:
- [ ] Model trained for sufficient epochs
- [ ] Validation IoU documented
- [ ] Test predictions generated
- [ ] Training curves saved
- [ ] Code documented
- [ ] Methodology explained in README
- [ ] Failure cases analyzed

## 💡 Tips for Better Results

1. **Start Simple**
   - Run with default settings first
   - Understand the baseline performance
   - Then iterate and improve

2. **Monitor Closely**
   - Watch training curves
   - Check validation samples
   - Identify problematic classes

3. **Iterate Strategically**
   - Change one thing at a time
   - Keep notes on what works
   - Save different model versions

4. **Document Everything**
   - What you tried
   - What worked/didn't work
   - Final configuration

## 🎓 Next Steps After Quick Start

Once you have baseline results:

1. **Analyze Performance**
   - Which classes perform poorly?
   - Where does the model fail?
   - What patterns do you see?

2. **Experiment**
   - Try different architectures
   - Adjust hyperparameters
   - Add more augmentation

3. **Optimize**
   - Focus on weak classes
   - Improve generalization
   - Reduce overfitting

4. **Document**
   - Create detailed report
   - Explain methodology
   - Show results

## 📚 Additional Resources

**If you're stuck:**
- Read the detailed README.md
- Check troubleshooting section
- Review PyTorch documentation
- Search for similar implementations

**For deeper understanding:**
- U-Net paper: https://arxiv.org/abs/1505.04597
- Semantic segmentation tutorial
- PyTorch segmentation examples

## 🏆 Final Tips

**Remember:**
- Clean code > complex code
- Simple model working > complex model broken
- Document as you go
- Test early, test often

**Success = Good methodology + Clear documentation + Reproducible results**

Good luck! 🚀
