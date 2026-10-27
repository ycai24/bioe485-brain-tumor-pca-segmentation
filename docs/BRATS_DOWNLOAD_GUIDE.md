# BraTS2020 Dataset Download Guide

## Quick Start Guide for Downloading BraTS2020 from Kaggle

This guide will help you download the BraTS2020 dataset used in this project.

## Prerequisites

- Python environment with pip
- Kaggle account (free to create at kaggle.com)
- ~7 GB free disk space

## Step-by-Step Instructions

### 1. Install Kaggle API

Open your terminal/PowerShell and run:

```bash
pip install kaggle
```

### 2. Get Your Kaggle API Credentials

1. Go to https://www.kaggle.com (create account if needed)
2. Click on your profile picture (top right)
3. Select "Settings" from dropdown
4. Scroll down to "API" section
5. Click "Create New Token"
6. This downloads a file called `kaggle.json`

### 3. Setup Kaggle Credentials

**Windows:**
```powershell
# Create .kaggle directory
New-Item -ItemType Directory -Force -Path $env:USERPROFILE\.kaggle

# Move kaggle.json to the directory
Move-Item -Path Downloads\kaggle.json -Destination $env:USERPROFILE\.kaggle\kaggle.json
```

**Linux/Mac:**
```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move kaggle.json
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download BraTS2020 Dataset

Navigate to your project directory and run:

```bash
# Navigate to project
cd "C:\Download\Work\MS BIC\BIOE485\bioe485 final project"

# Download dataset
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
```

This will download a file called `brats20-dataset-training-validation.zip` (~6.5 GB).

### 5. Extract Dataset

**Windows PowerShell:**
```powershell
# Extract to data/raw
Expand-Archive -Path brats20-dataset-training-validation.zip -DestinationPath data/raw/

# Remove zip file (optional)
Remove-Item brats20-dataset-training-validation.zip
```

**Linux/Mac:**
```bash
# Extract to data/raw
unzip brats20-dataset-training-validation.zip -d data/raw/

# Remove zip file (optional)
rm brats20-dataset-training-validation.zip
```

### 6. Verify Installation

After extraction, your directory structure should look like:

```
data/raw/
└── BraTS2020_TrainingData/
    └── MICCAI_BraTS2020_TrainingData/
        ├── BraTS20_Training_001/
        │   ├── BraTS20_Training_001_t1.nii.gz
        │   ├── BraTS20_Training_001_t1ce.nii.gz
        │   ├── BraTS20_Training_001_t2.nii.gz
        │   ├── BraTS20_Training_001_flair.nii.gz
        │   └── BraTS20_Training_001_seg.nii.gz
        ├── BraTS20_Training_002/
        └── ... (369 subjects total)
```

Run this Python code to verify:

```python
from pathlib import Path

data_path = Path("data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
subjects = list(data_path.glob("BraTS20_Training_*"))
print(f"Found {len(subjects)} subjects")
print(f"First subject: {subjects[0].name}" if subjects else "No subjects found!")
```

### 7. Update Notebook Path (if needed)

If your extracted path is different, update the `BRATS_DATA_PATH` variable in the notebook:

```python
# In cell 2 of the notebook
BRATS_DATA_PATH = '../data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
```

## Troubleshooting

### "401 Unauthorized" Error
- Make sure `kaggle.json` is in the correct location
- Check file permissions (should be 600 on Linux/Mac)
- Verify your Kaggle API token is valid

### "Command not found: kaggle"
- Ensure Kaggle API is installed: `pip install kaggle`
- Try: `python -m kaggle datasets download -d awsaf49/brats20-dataset-training-validation`

### Path Issues
- Use absolute paths if relative paths don't work
- On Windows, use forward slashes `/` or escaped backslashes `\\\\`
- Check that folder names match exactly (case-sensitive on Linux/Mac)

### Disk Space
- Dataset size: ~6.5 GB compressed, ~10 GB extracted
- Make sure you have at least 20 GB free space

## Alternative: Manual Download

If the API method doesn't work:

1. Visit: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
2. Click the "Download" button
3. Extract the downloaded zip file
4. Move contents to `data/raw/BraTS2020_TrainingData/`

## Dataset Information

**Dataset:** BraTS2020 Training Data  
**Source:** MICCAI Brain Tumor Segmentation Challenge 2020  
**Subjects:** 369 cases  
**Format:** NIfTI (.nii.gz)  
**Modalities:** T1, T1ce (contrast-enhanced), T2, FLAIR  
**Labels:** 0 (background), 1 (necrotic/non-enhancing tumor), 2 (edema), 4 (enhancing tumor)

## Citation

If you use this dataset, please cite:

```
Bakas et al., "Advancing The Cancer Genome Atlas glioma MRI collections with 
expert segmentation labels and radiomic features", Nature Scientific Data, 2017.

Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", 
IEEE Transactions on Medical Imaging, 2015.
```

## Next Steps

Once the data is downloaded:

1. Open `notebooks/tissue_segmentation_analysis.ipynb`
2. Set `USE_BRATS = True` in the notebook
3. Run all cells to perform analysis
4. Results will be saved in `results/` folder

## Need Help?

- Check the main [DATA_SOURCES.md](DATA_SOURCES.md) documentation
- Review the [README.md](../README.md) for project overview
- Open an issue on the GitHub repository

Happy analyzing! 🧠
