# BraTS2020 Dataset - Data Sources

## Primary Dataset: BraTS2020 from Kaggle

This project uses the **MICCAI Brain Tumor Segmentation Challenge 2020 (BraTS2020)** dataset for brain tumor segmentation analysis.

## BraTS2020 Dataset Details

### Kaggle Dataset

**Dataset Name:** `awsaf49/brats20-dataset-training-validation`  
**Direct Link:** https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation  
**Dataset Size:** ~6.5 GB (compressed)

**Description:**
- Brain tumor MRI data from multiple institutions
- Multi-parametric MRI sequences
- Expert-annotated tumor segmentations
- Training and validation splits included

**MRI Modalities (4 sequences per subject):**
- **T1-weighted (t1.nii.gz)**: Native T1-weighted scan
- **T1 with contrast enhancement (t1ce.nii.gz)**: T1-weighted with gadolinium contrast
- **T2-weighted (t2.nii.gz)**: T2-weighted scan
- **FLAIR (flair.nii.gz)**: Fluid Attenuated Inversion Recovery

**Ground Truth Segmentation (seg.nii.gz):**
- **Label 0**: Background (healthy tissue)
- **Label 1**: Necrotic and non-enhancing tumor core (NCR/NET)
- **Label 2**: Peritumoral edema (ED)
- **Label 4**: GD-enhancing tumor (ET)

**Format:** NIfTI compressed (.nii.gz)

**Number of Subjects:** 369 training cases

---

## How to Download from Kaggle

### Method 1: Kaggle API (Recommended)

**Step 1: Install Kaggle API**
```bash
pip install kaggle
```

**Step 2: Setup Kaggle Credentials**
1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New Token"
4. This downloads `kaggle.json`
5. Place `kaggle.json` in:
   - **Windows**: `C:\Users\YourName\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
6. Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

**Step 3: Download Dataset**
```bash
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
```

**Step 4: Extract to Project**
```bash
# Windows PowerShell
Expand-Archive -Path brats20-dataset-training-validation.zip -DestinationPath "data/raw/"

# Linux/Mac
unzip brats20-dataset-training-validation.zip -d data/raw/
```

### Method 2: Manual Download

1. Visit https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
2. Click "Download" button (requires Kaggle account)
3. Extract zip file to `data/raw/` folder

---

## Expected Data Structure

After downloading and extracting, your directory should look like:

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
        │   ├── BraTS20_Training_002_t1.nii.gz
        │   ├── BraTS20_Training_002_t1ce.nii.gz
        │   ├── BraTS20_Training_002_t2.nii.gz
        │   ├── BraTS20_Training_002_flair.nii.gz
        │   └── BraTS20_Training_002_seg.nii.gz
        ├── ...
        └── BraTS20_Training_369/
```

**Note:** The exact folder structure may vary slightly depending on how you extract the archive. Adjust paths in your code accordingly.

---

## Alternative Datasets (Optional)

### OASIS (Open Access Series of Imaging Studies)

**Website:** https://www.oasis-brains.org/

**Description:**
- Cross-sectional and longitudinal brain MRI datasets
- Normal aging and Alzheimer's disease subjects
- Well-characterized with demographic and clinical data

**Available Projects:**
- **OASIS-1**: Cross-sectional (ages 18-96)
- **OASIS-2**: Longitudinal (multiple timepoints)
- **OASIS-3**: Large-scale longitudinal

**Format:** NIfTI

**Access:** Free with account registration

**How to Download:**
1. Create account on OASIS website
2. Browse available datasets
3. Download individual scans or bulk data
4. Extract and place in `data/raw/`

---

### 3. Human Connectome Project (HCP)

**Website:** http://www.humanconnectomeproject.org/

**Description:**
- High-resolution brain imaging
- Structural and functional MRI
- Excellent for detailed tissue analysis

**Data Contents:**
- T1-weighted structural MRI
- T2-weighted structural MRI
- Diffusion MRI
- Resting-state and task fMRI

**Format:** NIfTI

**Access:** Free with data use agreement

**How to Download:**
1. Register at ConnectomeDB
2. Accept data use terms
3. Use Amazon S3 or Aspera for downloads
4. Place structural scans in `data/raw/`

---

### 4. ADNI (Alzheimer's Disease Neuroimaging Initiative)

**Website:** http://adni.loni.usc.edu/

**Description:**
- Longitudinal brain MRI for Alzheimer's research
- Multiple scanning protocols
- Rich clinical metadata

**Data Contents:**
- T1-weighted MRI
- T2-weighted MRI
- FLAIR
- PET imaging
- Clinical assessments

**Format:** Various (DICOM, NIfTI available)

**Access:** Registration and approval required

**How to Download:**
1. Register at ADNI website
2. Complete Data Use Agreement
3. Search and download via LONI IDA
4. Convert DICOM to NIfTI if needed
5. Place in `data/raw/`

---

### 5. IXI Dataset

**Website:** https://brain-development.org/ixi-dataset/

**Description:**
- Nearly 600 MR images from healthy subjects
- Multiple imaging modalities
- Ages 20-86

**Data Contents:**
- T1-weighted
- T2-weighted
- PD-weighted
- MRA
- Diffusion MRI

**Format:** NIfTI

**Access:** Freely available (cite if used)

**How to Download:**
1. Visit IXI website
2. Direct download from Imperial College London
3. No registration required
4. Place in `data/raw/`

---

## Working with BraTS2020 Data

### Loading a Single Subject

```python
from src.data_loader import MRIDataLoader
import matplotlib.pyplot as plt

# Initialize loader
loader = MRIDataLoader('data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')

# Define subject
subject = 'BraTS20_Training_001'
base_path = f'data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/{subject}'

# Load all modalities
t1 = loader.load_nifti(f'{base_path}/{subject}_t1.nii.gz')
t1ce = loader.load_nifti(f'{base_path}/{subject}_t1ce.nii.gz')
t2 = loader.load_nifti(f'{base_path}/{subject}_t2.nii.gz')
flair = loader.load_nifti(f'{base_path}/{subject}_flair.nii.gz')
seg = loader.load_nifti(f'{base_path}/{subject}_seg.nii.gz')

print(f"Volume shape: {t1.shape}")  # Typically (240, 240, 155)
```

### Extract and Visualize 2D Slice

```python
# Extract middle slice
slice_idx = t1.shape[2] // 2  # Middle axial slice

t1_slice = loader.extract_2d_slice(t1, slice_idx=slice_idx, axis=2)
t2_slice = loader.extract_2d_slice(t2, slice_idx=slice_idx, axis=2)
flair_slice = loader.extract_2d_slice(flair, slice_idx=slice_idx, axis=2)
seg_slice = loader.extract_2d_slice(seg, slice_idx=slice_idx, axis=2)

# Visualize all modalities
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(t1_slice, cmap='gray')
axes[0, 0].set_title('T1')
axes[0, 1].imshow(t1ce_slice, cmap='gray')
axes[0, 1].set_title('T1ce')
axes[0, 2].imshow(t2_slice, cmap='gray')
axes[0, 2].set_title('T2')
axes[1, 0].imshow(flair_slice, cmap='gray')
axes[1, 0].set_title('FLAIR')
axes[1, 1].imshow(seg_slice, cmap='jet')
axes[1, 1].set_title('Ground Truth Segmentation')
plt.show()
```

### Prepare Multi-Modal Data for PCA

```python
# Normalize intensities
t1_norm = loader.normalize_intensity(t1_slice)
t2_norm = loader.normalize_intensity(t2_slice)
flair_norm = loader.normalize_intensity(flair_slice)

# Create multi-channel data matrix
images = [t1_norm, t2_norm, flair_norm]
X = loader.prepare_data_matrix(images)

print(f"Data matrix shape: {X.shape}")  # (n_pixels, n_modalities)
```

## Data Preprocessing Steps

Before using the data with our pipeline:

1. **Check format**: Ensure files are NIfTI (.nii or .nii.gz)
2. **Verify orientation**: Check if reorientation is needed
3. **Skull stripping** (optional): Remove non-brain tissue
4. **Bias field correction** (optional): Correct intensity inhomogeneity

## Tools for Data Conversion

If you receive DICOM files:

### Using dcm2niix (Recommended)
```bash
# Install
sudo apt-get install dcm2niix  # Linux
brew install dcm2niix          # macOS

# Convert
dcm2niix -o data/raw/ -f subject_001 path/to/dicom/folder/
```

### Using Python (nibabel + pydicom)
```python
import nibabel as nib
import pydicom
from pydicom.data import get_testdata_files

# Load DICOM series and convert
# (Implementation depends on your DICOM structure)
```

## Sample Data for Testing

If you don't have access to real MRI data yet, the code includes synthetic data generation:

```python
from src.utils import create_synthetic_mri_data

# Create synthetic brain-like MRI
image, ground_truth = create_synthetic_mri_data(
    shape=(256, 256),
    n_tissues=3,
    noise_level=0.1
)
```

## Data Size Considerations

- **Single MRI volume**: ~1-50 MB (compressed NIfTI)
- **Complete BraTS dataset**: ~20-30 GB
- **OASIS-3**: ~1 TB (full dataset)
- **Recommended for this project**: Start with 10-50 scans

## Citation Requirements

If you use these datasets, please cite appropriately:

**BraTS:**
```
Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)",
IEEE Transactions on Medical Imaging, 2015
```

**OASIS:**
```
Marcus et al., "Open Access Series of Imaging Studies (OASIS)",
Journal of Cognitive Neuroscience, 2007
```

**HCP:**
```
Van Essen et al., "The WU-Minn Human Connectome Project",
NeuroImage, 2013
```

## Additional Resources

- **NiBabel Documentation**: https://nipy.org/nibabel/
- **FSL (FMRIB Software Library)**: https://fsl.fmrib.ox.ac.uk/
- **FreeSurfer**: https://surfer.nmr.mgh.harvard.edu/
- **ANTs**: http://stnava.github.io/ANTs/

## Troubleshooting

### Common Issues:

1. **File format errors**: Ensure files are valid NIfTI
2. **Memory errors**: Start with smaller 2D slices
3. **Different orientations**: Use nibabel to standardize
4. **Missing data**: Check download completed successfully

### Getting Help:

- Check dataset documentation
- Review NiBabel tutorials
- See project README.md for code examples
- Open issue on GitHub repository

## Next Steps

After downloading data:

1. Place files in `data/raw/`
2. Open `notebooks/tissue_segmentation_analysis.ipynb`
3. Modify data loading cell to point to your files
4. Run the analysis pipeline

Happy analyzing! 🧠
