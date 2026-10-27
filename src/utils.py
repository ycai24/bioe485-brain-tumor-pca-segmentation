"""
Utility functions for MRI tissue segmentation project.
"""

import numpy as np
from typing import Tuple


def compute_reconstruction_error(X_original: np.ndarray, 
                                 X_reconstructed: np.ndarray) -> float:
    """
    Compute mean squared reconstruction error.
    
    Args:
        X_original: Original data
        X_reconstructed: Reconstructed data from PCA
        
    Returns:
        Mean squared error
    """
    mse = np.mean((X_original - X_reconstructed) ** 2)
    return mse


def compute_dice_coefficient(seg1: np.ndarray, 
                             seg2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two segmentations.
    
    Args:
        seg1: First segmentation
        seg2: Second segmentation
        
    Returns:
        Dice coefficient (0 to 1)
    """
    intersection = np.sum(seg1 == seg2)
    dice = 2 * intersection / (seg1.size + seg2.size)
    return dice


def normalize_segmentation_labels(labels: np.ndarray) -> np.ndarray:
    """
    Normalize segmentation labels to consecutive integers starting from 0.
    
    Args:
        labels: Original labels
        
    Returns:
        Normalized labels
    """
    unique_labels = np.unique(labels)
    normalized = np.zeros_like(labels)
    
    for new_label, old_label in enumerate(unique_labels):
        normalized[labels == old_label] = new_label
    
    return normalized


def compute_tissue_statistics(image: np.ndarray, 
                              segmentation: np.ndarray) -> dict:
    """
    Compute statistics for each tissue class.
    
    Args:
        image: Original image
        segmentation: Segmentation map
        
    Returns:
        Dictionary with tissue statistics
    """
    stats = {}
    unique_labels = np.unique(segmentation)
    
    for label in unique_labels:
        mask = segmentation == label
        tissue_intensities = image[mask]
        
        stats[f'tissue_{int(label)}'] = {
            'mean_intensity': np.mean(tissue_intensities),
            'std_intensity': np.std(tissue_intensities),
            'median_intensity': np.median(tissue_intensities),
            'voxel_count': np.sum(mask),
            'percentage': 100 * np.sum(mask) / segmentation.size
        }
    
    return stats


def create_synthetic_mri_data(shape: Tuple[int, int] = (256, 256),
                             n_tissues: int = 3,
                             noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic MRI-like data for testing (if real data not available).
    
    Args:
        shape: Image shape
        n_tissues: Number of tissue classes
        noise_level: Amount of Gaussian noise
        
    Returns:
        Tuple of (synthetic_image, ground_truth_labels)
    """
    from scipy.ndimage import gaussian_filter
    
    # Create synthetic tissue regions
    labels = np.zeros(shape, dtype=int)
    image = np.zeros(shape, dtype=float)
    
    # Generate regions with different intensities
    center = np.array(shape) // 2
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    # Background/CSF
    labels[:, :] = 0
    image[:, :] = 0.2
    
    # Gray matter (outer ring)
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    gray_matter_mask = (dist_from_center < shape[0] * 0.4) & (dist_from_center > shape[0] * 0.25)
    labels[gray_matter_mask] = 1
    image[gray_matter_mask] = 0.6
    
    # White matter (inner circle)
    white_matter_mask = dist_from_center <= shape[0] * 0.25
    labels[white_matter_mask] = 2
    image[white_matter_mask] = 0.9
    
    # Add smooth transitions
    image = gaussian_filter(image, sigma=3)
    
    # Add noise
    noise = np.random.normal(0, noise_level, shape)
    image = image + noise
    image = np.clip(image, 0, 1)
    
    return image, labels
