# Data Directory

## Purpose
This folder contains MRI imaging data for the brain tumor segmentation project using BraTS2020 dataset.

## Structure

### `raw/`
Place your BraTS2020 dataset here after downloading from Kaggle.

**Supported formats:**
- `.nii.gz` - Compressed NIfTI (BraTS2020 standard format)

**Expected BraTS2020 structure:**
```
raw/
└── BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_t1.nii.gz
    │   ├── BraTS20_Training_001_t1ce.nii.gz
    │   ├── BraTS20_Training_001_t2.nii.gz
    │   ├── BraTS20_Training_001_flair.nii.gz
    │   └── BraTS20_Training_001_seg.nii.gz
    ├── BraTS20_Training_002/
    │   └── ...
    └── ...
```

### `processed/`
Processed data outputs will be saved here automatically by the pipeline.

## Where to Get Data

### BraTS2020 from Kaggle

**Dataset**: awsaf49/brats20-dataset-training-validation  
**URL**: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

### Download Steps:

1. Install Kaggle API: `pip install kaggle`
2. Setup credentials from kaggle.com/settings/account
3. Download: `kaggle datasets download -d awsaf49/brats20-dataset-training-validation`
4. Extract to `data/raw/`

See `../docs/DATA_SOURCES.md` for detailed instructions.

## Quick Start

1. Download MRI data from one of the sources listed in DATA_SOURCES.md
2. Place NIfTI files in the `raw/` subdirectory
3. Run the analysis pipeline using the Jupyter notebook or Python scripts

## Example Usage

```python
from src.data_loader import MRIDataLoader

# Initialize loader
loader = MRIDataLoader('data/raw/BraTS2020_TrainingData')

# Load all modalities for a subject
subject = 'BraTS20_Training_001'
base_path = f'data/raw/BraTS2020_TrainingData/{subject}'

t1 = loader.load_nifti(f'{base_path}/{subject}_t1.nii.gz')
t1ce = loader.load_nifti(f'{base_path}/{subject}_t1ce.nii.gz')
t2 = loader.load_nifti(f'{base_path}/{subject}_t2.nii.gz')
flair = loader.load_nifti(f'{base_path}/{subject}_flair.nii.gz')
seg = loader.load_nifti(f'{base_path}/{subject}_seg.nii.gz')

# Extract 2D slice
slice_idx = t1.shape[2] // 2
t1_slice = loader.extract_2d_slice(t1, slice_idx=slice_idx, axis=2)

# Prepare multi-channel data
images = [t1_slice, t2_slice, flair_slice]
X = loader.prepare_data_matrix(images)
```

## Note

**Do not commit large MRI files to Git!**

The `.gitignore` file is configured to exclude:
- `*.nii`
- `*.nii.gz`
- Large binary files

If you need to share data, use:
- External storage (Google Drive, Dropbox)
- Institutional data repositories
- AWS S3 or similar cloud storage
