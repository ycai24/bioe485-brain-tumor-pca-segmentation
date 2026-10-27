"""
Tissue segmentation module using clustering in PCA space.
Implements Gaussian Mixture Model clustering for tissue classification.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TissueSegmenter:
    """Segment tissues using clustering in PCA-reduced space."""
    
    def __init__(self, n_clusters: int = 3, method: str = 'gmm'):
        """
        Initialize tissue segmenter.
        
        Args:
            n_clusters: Number of tissue classes (e.g., 3 for gray matter, white matter, CSF)
            method: Clustering method ('gmm' or 'kmeans')
        """
        self.n_clusters = n_clusters
        self.method = method
        self.model = None
        self.labels_ = None
        
    def fit(self, X: np.ndarray) -> 'TissueSegmenter':
        """
        Fit clustering model to PCA-transformed data.
        
        Args:
            X: Data in PCA space of shape (n_samples, n_components)
            
        Returns:
            self
        """
        logger.info(f"Fitting {self.method.upper()} with {self.n_clusters} clusters")
        
        if self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
        elif self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        
        logger.info(f"Clustering complete. Label distribution: "
                   f"{np.bincount(self.labels_)}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data in PCA space of shape (n_samples, n_components)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        return self.model.predict(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.
        
        Args:
            X: Data in PCA space of shape (n_samples, n_components)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        self.fit(X)
        return self.labels_
    
    def reshape_labels_to_image(self, labels: np.ndarray, 
                               image_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reshape 1D label array back to image dimensions.
        
        Args:
            labels: 1D array of cluster labels
            image_shape: Original image shape
            
        Returns:
            Segmentation map with shape image_shape
        """
        return labels.reshape(image_shape)
    
    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get cluster membership probabilities (only for GMM).
        
        Args:
            X: Data in PCA space of shape (n_samples, n_components)
            
        Returns:
            Probability array of shape (n_samples, n_clusters)
        """
        if self.method != 'gmm':
            raise ValueError("Probabilities only available for GMM")
        
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        return self.model.predict_proba(X)
    
    def compute_bic(self, X: np.ndarray) -> float:
        """
        Compute Bayesian Information Criterion (only for GMM).
        
        Args:
            X: Data in PCA space
            
        Returns:
            BIC score (lower is better)
        """
        if self.method != 'gmm':
            raise ValueError("BIC only available for GMM")
        
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        return self.model.bic(X)
    
    def compute_aic(self, X: np.ndarray) -> float:
        """
        Compute Akaike Information Criterion (only for GMM).
        
        Args:
            X: Data in PCA space
            
        Returns:
            AIC score (lower is better)
        """
        if self.method != 'gmm':
            raise ValueError("AIC only available for GMM")
        
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        return self.model.aic(X)


def segment_with_original_space(X_original: np.ndarray, 
                               n_clusters: int = 3,
                               method: str = 'gmm') -> np.ndarray:
    """
    Perform segmentation in original image space (for comparison).
    
    Args:
        X_original: Data in original space
        n_clusters: Number of tissue classes
        method: Clustering method
        
    Returns:
        Cluster labels
    """
    logger.info("Performing segmentation in original space")
    segmenter = TissueSegmenter(n_clusters=n_clusters, method=method)
    return segmenter.fit_predict(X_original)


def compare_segmentations(labels_pca: np.ndarray, 
                         labels_original: np.ndarray) -> dict:
    """
    Compare segmentation results between PCA and original space.
    
    Args:
        labels_pca: Labels from PCA-space segmentation
        labels_original: Labels from original-space segmentation
        
    Returns:
        Dictionary with comparison metrics
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari = adjusted_rand_score(labels_original, labels_pca)
    nmi = normalized_mutual_info_score(labels_original, labels_pca)
    
    comparison = {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi
    }
    
    logger.info(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")
    return comparison
