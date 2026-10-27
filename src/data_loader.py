"""
Data loading and preprocessing module for MRI images.
Handles loading NIfTI files and preparing data for PCA analysis.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIDataLoader:
    """Load and preprocess MRI data from NIfTI files."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing MRI data
        """
        self.data_dir = Path(data_dir)
        
    def load_nifti(self, file_path: str) -> np.ndarray:
        """
        Load a NIfTI file and return the image data.
        
        Args:
            file_path: Path to NIfTI file (.nii or .nii.gz)
            
        Returns:
            Numpy array containing image data
        """
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            logger.info(f"Loaded {file_path} with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def normalize_intensity(self, data: np.ndarray, 
                          percentile_range: Tuple[float, float] = (1, 99)) -> np.ndarray:
        """
        Normalize image intensity using percentile-based clipping.
        
        Args:
            data: Input image array
            percentile_range: Tuple of (low, high) percentiles for clipping
            
        Returns:
            Normalized image array
        """
        p_low, p_high = np.percentile(data, percentile_range)
        data_clipped = np.clip(data, p_low, p_high)
        
        # Normalize to [0, 1]
        data_normalized = (data_clipped - p_low) / (p_high - p_low)
        return data_normalized
    
    def extract_2d_slice(self, volume: np.ndarray, 
                        slice_idx: int, 
                        axis: int = 2) -> np.ndarray:
        """
        Extract a 2D slice from a 3D volume.
        
        Args:
            volume: 3D image volume
            slice_idx: Index of slice to extract
            axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
            
        Returns:
            2D slice array
        """
        if axis == 0:
            return volume[slice_idx, :, :]
        elif axis == 1:
            return volume[:, slice_idx, :]
        else:
            return volume[:, :, slice_idx]
    
    def extract_patches(self, image: np.ndarray, 
                       patch_size: Tuple[int, int] = (32, 32),
                       stride: int = 16) -> np.ndarray:
        """
        Extract overlapping patches from a 2D image.
        
        Args:
            image: 2D image array
            patch_size: Size of patches (height, width)
            stride: Stride between patches
            
        Returns:
            Array of shape (n_patches, patch_height * patch_width)
        """
        h, w = image.shape
        ph, pw = patch_size
        
        patches = []
        for i in range(0, h - ph + 1, stride):
            for j in range(0, w - pw + 1, stride):
                patch = image[i:i+ph, j:j+pw]
                patches.append(patch.flatten())
        
        patches = np.array(patches)
        logger.info(f"Extracted {len(patches)} patches of size {patch_size}")
        return patches
    
    def flatten_image(self, image: np.ndarray) -> np.ndarray:
        """
        Flatten image to vectors for PCA.
        
        Args:
            image: 2D or 3D image array
            
        Returns:
            Flattened array where each pixel is a sample
        """
        return image.reshape(-1, 1) if image.ndim == 2 else image.reshape(-1, image.shape[-1])
    
    def prepare_data_matrix(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Prepare data matrix X from multiple images.
        Each row is a pixel, each column is an image/feature.
        
        Args:
            images: List of 2D image arrays (should have same shape)
            
        Returns:
            Data matrix of shape (n_pixels, n_images)
        """
        flattened = [img.flatten() for img in images]
        X = np.column_stack(flattened)
        logger.info(f"Prepared data matrix with shape {X.shape}")
        return X
