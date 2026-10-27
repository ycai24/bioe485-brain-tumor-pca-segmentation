"""
Principal Component Analysis implementation for MRI tissue segmentation.
Implements eigenanalysis for dimensionality reduction.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAAnalyzer:
    """Perform PCA on MRI data using eigenanalysis."""
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize PCA analyzer.
        
        Args:
            n_components: Number of principal components to keep (None = keep all)
        """
        self.n_components = n_components
        self.mean_ = None
        self.eigenvectors_ = None
        self.eigenvalues_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X: np.ndarray) -> 'PCAAnalyzer':
        """
        Fit PCA model to data using eigenanalysis.
        
        Mathematical formulation:
        1. Center data: X_centered = X - mean(X)
        2. Compute covariance matrix: C = (1/(n-1)) * X^T @ X
        3. Eigendecomposition: C @ v_j = λ_j * v_j
        4. Sort eigenvectors by eigenvalues in descending order
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        logger.info(f"Fitting PCA on data with shape {X.shape}")
        
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        # C = (1/(n-1)) * X^T @ X
        cov_matrix = (1 / (n_samples - 1)) * (X_centered.T @ X_centered)
        
        # Step 3: Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Step 4: Sort by eigenvalues (descending order)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]
        
        # Calculate explained variance
        total_variance = np.sum(self.eigenvalues_)
        self.explained_variance_ = self.eigenvalues_
        self.explained_variance_ratio_ = self.eigenvalues_ / total_variance
        
        # Select top k components if specified
        if self.n_components is not None:
            self.eigenvectors_ = self.eigenvectors_[:, :self.n_components]
            self.eigenvalues_ = self.eigenvalues_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
        
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        logger.info(f"Top {len(self.eigenvalues_)} components explain "
                   f"{cumulative_variance[-1]:.2%} of variance")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components.
        
        Y = X @ V_k where V_k contains k eigenvectors with largest eigenvalues
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if self.mean_ is None:
            raise ValueError("PCA model has not been fitted yet")
        
        X_centered = X - self.mean_
        Y = X_centered @ self.eigenvectors_
        
        logger.info(f"Transformed data to shape {Y.shape}")
        return Y
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal component space.
        
        X_reconstructed = Y @ V_k^T + mean
        
        Args:
            Y: Transformed data of shape (n_samples, n_components)
            
        Returns:
            Reconstructed data of shape (n_samples, n_features)
        """
        if self.mean_ is None:
            raise ValueError("PCA model has not been fitted yet")
        
        X_reconstructed = Y @ self.eigenvectors_.T + self.mean_
        return X_reconstructed
    
    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio.
        
        Returns:
            Array of cumulative variance ratios
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA model has not been fitted yet")
        
        return np.cumsum(self.explained_variance_ratio_)
    
    def get_n_components_for_variance(self, variance_threshold: float = 0.95) -> int:
        """
        Get number of components needed to explain desired variance.
        
        Args:
            variance_threshold: Desired cumulative variance (e.g., 0.95 for 95%)
            
        Returns:
            Number of components needed
        """
        cumulative_variance = self.get_cumulative_variance()
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        logger.info(f"{n_components} components needed for "
                   f"{variance_threshold:.0%} variance")
        return n_components
