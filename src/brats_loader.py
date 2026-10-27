"""
BraTS2020 dataset loader for brain tumor segmentation.
Handles multi-modal MRI data loading and preprocessing.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from data_loader import MRIDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BraTSLoader:
    """Load and preprocess BraTS2020 dataset."""
    
    # BraTS2020 tumor labels
    LABEL_BACKGROUND = 0
    LABEL_NCR_NET = 1  # Necrotic and non-enhancing tumor core
    LABEL_ED = 2       # Peritumoral edema
    LABEL_ET = 4       # GD-enhancing tumor
    
    def __init__(self, data_dir: str):
        """
        Initialize BraTS loader.
        
        Args:
            data_dir: Path to BraTS2020_TrainingData directory
        """
        self.data_dir = Path(data_dir)
        self.loader = MRIDataLoader(str(data_dir))
        
        # Check if directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
    
    def get_subject_list(self) -> List[str]:
        """
        Get list of all subjects in dataset.
        
        Returns:
            List of subject IDs
        """
        # Find all subject directories
        subjects = []
        if self.data_dir.exists():
            for path in self.data_dir.iterdir():
                if path.is_dir() and path.name.startswith('BraTS20_Training_'):
                    subjects.append(path.name)
        
        subjects.sort()
        logger.info(f"Found {len(subjects)} subjects")
        return subjects
    
    def load_subject(self, subject_id: str, load_seg: bool = True) -> Dict[str, np.ndarray]:
        """
        Load all modalities for a subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'BraTS20_Training_001')
            load_seg: Whether to load segmentation ground truth
            
        Returns:
            Dictionary with keys: 't1', 't1ce', 't2', 'flair', 'seg' (if load_seg=True)
        """
        subject_path = self.data_dir / subject_id
        
        if not subject_path.exists():
            raise ValueError(f"Subject not found: {subject_id}")
        
        data = {}
        
        # Load all modalities
        modalities = ['t1', 't1ce', 't2', 'flair']
        for mod in modalities:
            # Try both .nii.gz and .nii extensions
            file_path_gz = subject_path / f"{subject_id}_{mod}.nii.gz"
            file_path_nii = subject_path / f"{subject_id}_{mod}.nii"
            
            if file_path_gz.exists():
                data[mod] = self.loader.load_nifti(str(file_path_gz))
            elif file_path_nii.exists():
                data[mod] = self.loader.load_nifti(str(file_path_nii))
            else:
                logger.warning(f"File not found: {file_path_gz} or {file_path_nii}")
        
        # Load segmentation if requested
        if load_seg:
            seg_path_gz = subject_path / f"{subject_id}_seg.nii.gz"
            seg_path_nii = subject_path / f"{subject_id}_seg.nii"
            
            if seg_path_gz.exists():
                data['seg'] = self.loader.load_nifti(str(seg_path_gz))
            elif seg_path_nii.exists():
                data['seg'] = self.loader.load_nifti(str(seg_path_nii))
            else:
                logger.warning(f"Segmentation not found: {seg_path_gz} or {seg_path_nii}")
        
        logger.info(f"Loaded subject {subject_id} with {len(data)} volumes")
        return data
    
    def extract_slice_all_modalities(self, data: Dict[str, np.ndarray],
                                    slice_idx: int,
                                    axis: int = 2) -> Dict[str, np.ndarray]:
        """
        Extract same slice from all modalities.
        
        Args:
            data: Dictionary of volumes from load_subject
            slice_idx: Slice index
            axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
            
        Returns:
            Dictionary of 2D slices
        """
        slices = {}
        for key, volume in data.items():
            slices[key] = self.loader.extract_2d_slice(volume, slice_idx, axis)
        
        return slices
    
    def normalize_modalities(self, slices: Dict[str, np.ndarray],
                           exclude_keys: List[str] = ['seg']) -> Dict[str, np.ndarray]:
        """
        Normalize intensity of MRI slices (excluding segmentation).
        
        Args:
            slices: Dictionary of 2D slices
            exclude_keys: Keys to skip normalization (e.g., segmentation)
            
        Returns:
            Dictionary with normalized slices
        """
        normalized = {}
        for key, slice_data in slices.items():
            if key in exclude_keys:
                normalized[key] = slice_data
            else:
                normalized[key] = self.loader.normalize_intensity(slice_data)
        
        return normalized
    
    def prepare_multimodal_data(self, slices: Dict[str, np.ndarray],
                               modalities: List[str] = ['t1', 't2', 'flair']) -> np.ndarray:
        """
        Prepare multi-modal data matrix for PCA.
        
        Args:
            slices: Dictionary of 2D slices
            modalities: List of modalities to include
            
        Returns:
            Data matrix of shape (n_pixels, n_modalities)
        """
        images = [slices[mod] for mod in modalities if mod in slices]
        
        if not images:
            raise ValueError("No valid modalities found in slices")
        
        X = self.loader.prepare_data_matrix(images)
        logger.info(f"Prepared multi-modal data: {X.shape}")
        return X
    
    def get_tumor_mask(self, seg: np.ndarray) -> np.ndarray:
        """
        Create binary mask of tumor region (all non-background labels).
        
        Args:
            seg: Segmentation array
            
        Returns:
            Binary mask (1 for tumor, 0 for background)
        """
        return (seg > 0).astype(np.uint8)
    
    def get_tumor_subregions(self, seg: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract binary masks for different tumor sub-regions.
        
        Args:
            seg: Segmentation array
            
        Returns:
            Dictionary with masks for each tumor region
        """
        masks = {
            'whole_tumor': (seg > 0).astype(np.uint8),
            'tumor_core': np.isin(seg, [self.LABEL_NCR_NET, self.LABEL_ET]).astype(np.uint8),
            'enhancing_tumor': (seg == self.LABEL_ET).astype(np.uint8),
            'edema': (seg == self.LABEL_ED).astype(np.uint8),
            'necrosis': (seg == self.LABEL_NCR_NET).astype(np.uint8)
        }
        return masks
    
    def find_tumor_slices(self, seg_volume: np.ndarray,
                         axis: int = 2,
                         min_tumor_pixels: int = 100) -> List[int]:
        """
        Find slices that contain tumor.
        
        Args:
            seg_volume: 3D segmentation volume
            axis: Axis along which to check slices
            min_tumor_pixels: Minimum number of tumor pixels to consider
            
        Returns:
            List of slice indices containing tumor
        """
        tumor_slices = []
        
        for i in range(seg_volume.shape[axis]):
            if axis == 0:
                slice_seg = seg_volume[i, :, :]
            elif axis == 1:
                slice_seg = seg_volume[:, i, :]
            else:
                slice_seg = seg_volume[:, :, i]
            
            tumor_pixels = np.sum(slice_seg > 0)
            if tumor_pixels >= min_tumor_pixels:
                tumor_slices.append(i)
        
        logger.info(f"Found {len(tumor_slices)} slices with tumor (axis={axis})")
        return tumor_slices
    
    def load_subject_with_tumor_slice(self, subject_id: str,
                                     axis: int = 2) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Load subject and automatically select slice with most tumor.
        
        Args:
            subject_id: Subject identifier
            axis: Axis to slice along
            
        Returns:
            Tuple of (slices_dict, slice_idx)
        """
        # Load full volumes
        data = self.load_subject(subject_id, load_seg=True)
        
        if 'seg' not in data:
            # If no segmentation, use middle slice
            slice_idx = data['t1'].shape[axis] // 2
        else:
            # Find slice with maximum tumor
            seg = data['seg']
            max_tumor = 0
            best_slice = seg.shape[axis] // 2
            
            for i in range(seg.shape[axis]):
                if axis == 0:
                    slice_seg = seg[i, :, :]
                elif axis == 1:
                    slice_seg = seg[:, i, :]
                else:
                    slice_seg = seg[:, :, i]
                
                tumor_pixels = np.sum(slice_seg > 0)
                if tumor_pixels > max_tumor:
                    max_tumor = tumor_pixels
                    best_slice = i
            
            slice_idx = best_slice
            logger.info(f"Selected slice {slice_idx} with {max_tumor} tumor pixels")
        
        # Extract slices
        slices = self.extract_slice_all_modalities(data, slice_idx, axis)
        slices = self.normalize_modalities(slices)
        
        return slices, slice_idx


if __name__ == "__main__":
    # Example usage
    loader = BraTSLoader('data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
    
    # List subjects
    subjects = loader.get_subject_list()
    if subjects:
        print(f"Found {len(subjects)} subjects")
        print(f"First subject: {subjects[0]}")
        
        # Load first subject
        data = loader.load_subject(subjects[0])
        print(f"Loaded modalities: {list(data.keys())}")
        print(f"Volume shape: {data['t1'].shape}")
