"""
Main pipeline for eigenanalysis-based tissue segmentation.
"""

import numpy as np
from pathlib import Path
import logging

from data_loader import MRIDataLoader
from pca_analysis import PCAAnalyzer
from segmentation import TissueSegmenter, compare_segmentations
from visualization import (plot_explained_variance, plot_segmentation, 
                          plot_comparison, plot_pca_projection_2d)
from utils import compute_tissue_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationPipeline:
    """Complete pipeline for MRI tissue segmentation using PCA."""
    
    def __init__(self, n_components: int = 10, n_clusters: int = 3):
        """
        Initialize segmentation pipeline.
        
        Args:
            n_components: Number of principal components
            n_clusters: Number of tissue classes
        """
        self.n_components = n_components
        self.n_clusters = n_clusters
        
        self.data_loader = MRIDataLoader("")
        self.pca = PCAAnalyzer(n_components=n_components)
        self.segmenter = TissueSegmenter(n_clusters=n_clusters, method='gmm')
        
        self.image_original = None
        self.X_original = None
        self.Y_pca = None
        self.labels_pca = None
        self.segmentation_pca = None
        
    def load_and_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Load and preprocess image data.
        
        Args:
            image: Input 2D image array
            
        Returns:
            Preprocessed and flattened data matrix
        """
        logger.info("Preprocessing image...")
        
        # Normalize intensity
        image_normalized = self.data_loader.normalize_intensity(image)
        self.image_original = image_normalized
        
        # Flatten to data matrix (each pixel is a sample)
        X = image_normalized.reshape(-1, 1)
        self.X_original = X
        
        logger.info(f"Data matrix shape: {X.shape}")
        return X
    
    def perform_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Perform PCA on data.
        
        Args:
            X: Data matrix
            
        Returns:
            Transformed data in PC space
        """
        logger.info("Performing PCA...")
        
        # For single-channel data, we need multiple features
        # Create features from local neighborhoods
        if X.shape[1] == 1:
            logger.info("Creating multi-channel features from image patches...")
            X = self._create_patch_features(self.image_original)
            self.X_original = X
        
        Y = self.pca.fit_transform(X)
        self.Y_pca = Y
        
        logger.info(f"PCA transformed data shape: {Y.shape}")
        return Y
    
    def _create_patch_features(self, image: np.ndarray, 
                              patch_size: int = 5) -> np.ndarray:
        """
        Create feature vectors from image patches.
        
        Args:
            image: 2D image
            patch_size: Size of neighborhood patches
            
        Returns:
            Feature matrix where each row is a pixel and columns are patch features
        """
        from scipy.ndimage import uniform_filter
        
        h, w = image.shape
        pad = patch_size // 2
        
        # Pad image
        image_padded = np.pad(image, pad, mode='reflect')
        
        # Create features from different filters
        features = []
        
        # Original intensity
        features.append(image.flatten())
        
        # Local mean with different window sizes
        for size in [3, 5, 7]:
            filtered = uniform_filter(image, size=size)
            features.append(filtered.flatten())
        
        # Gradient magnitude
        from scipy.ndimage import sobel
        grad_x = sobel(image, axis=0)
        grad_y = sobel(image, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        features.append(grad_mag.flatten())
        
        # Stack features
        X = np.column_stack(features)
        logger.info(f"Created {X.shape[1]} features per pixel")
        
        return X
    
    def segment_tissues(self, Y: np.ndarray) -> np.ndarray:
        """
        Perform tissue segmentation in PC space.
        
        Args:
            Y: Data in PC space
            
        Returns:
            Segmentation labels
        """
        logger.info("Performing tissue segmentation...")
        
        labels = self.segmenter.fit_predict(Y)
        self.labels_pca = labels
        
        # Reshape to image
        self.segmentation_pca = labels.reshape(self.image_original.shape)
        
        return self.segmentation_pca
    
    def run_pipeline(self, image: np.ndarray) -> dict:
        """
        Run complete segmentation pipeline.
        
        Args:
            image: Input 2D image
            
        Returns:
            Dictionary with results
        """
        logger.info("="*60)
        logger.info("Starting segmentation pipeline")
        logger.info("="*60)
        
        # Step 1: Preprocess
        X = self.load_and_preprocess(image)
        
        # Step 2: PCA
        Y = self.perform_pca(X)
        
        # Step 3: Segment
        segmentation = self.segment_tissues(Y)
        
        # Step 4: Compute statistics
        stats = compute_tissue_statistics(self.image_original, segmentation)
        
        logger.info("="*60)
        logger.info("Pipeline complete!")
        logger.info("="*60)
        
        results = {
            'image': self.image_original,
            'segmentation': segmentation,
            'pca': self.pca,
            'labels': self.labels_pca,
            'Y_pca': self.Y_pca,
            'statistics': stats
        }
        
        return results
    
    def visualize_results(self, results: dict, save_dir: str = None):
        """
        Visualize all results.
        
        Args:
            results: Results dictionary from run_pipeline
            save_dir: Directory to save figures (optional)
        """
        save_dir = Path(save_dir) if save_dir else None
        
        # Plot explained variance
        plot_explained_variance(
            self.pca,
            save_path=save_dir / 'explained_variance.png' if save_dir else None
        )
        
        # Plot segmentation
        plot_segmentation(
            results['image'],
            results['segmentation'],
            title='Tissue Segmentation (PCA Space)',
            save_path=save_dir / 'segmentation.png' if save_dir else None
        )
        
        # Plot PC projection
        plot_pca_projection_2d(
            results['Y_pca'],
            results['labels'],
            save_path=save_dir / 'pca_projection.png' if save_dir else None
        )
        
        # Print statistics
        logger.info("\nTissue Statistics:")
        for tissue, stats in results['statistics'].items():
            logger.info(f"\n{tissue}:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    # Example usage with synthetic data
    from utils import create_synthetic_mri_data
    
    # Create synthetic MRI data
    image, ground_truth = create_synthetic_mri_data(shape=(256, 256), n_tissues=3)
    
    # Run pipeline
    pipeline = SegmentationPipeline(n_components=5, n_clusters=3)
    results = pipeline.run_pipeline(image)
    
    # Visualize
    pipeline.visualize_results(results)
