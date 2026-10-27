"""
Visualization utilities for PCA analysis and tissue segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import seaborn as sns

sns.set_style('whitegrid')


def plot_explained_variance(pca_analyzer, save_path: Optional[str] = None):
    """
    Plot explained variance by principal components.
    
    Args:
        pca_analyzer: Fitted PCAAnalyzer object
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    n_components = len(pca_analyzer.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)
    
    # Individual explained variance
    ax1.bar(components, pca_analyzer.explained_variance_ratio_, alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Variance Explained by Each PC')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumulative_variance = pca_analyzer.get_cumulative_variance()
    ax2.plot(components, cumulative_variance, 'o-', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_principal_components(eigenvectors: np.ndarray, 
                             image_shape: Tuple[int, int],
                             n_components: int = 4,
                             save_path: Optional[str] = None):
    """
    Visualize principal component eigenvectors as images.
    
    Args:
        eigenvectors: Eigenvector matrix
        image_shape: Original image shape
        n_components: Number of components to display
        save_path: Path to save figure (optional)
    """
    n_cols = min(4, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten() if n_components > 1 else [axes]
    
    for i in range(n_components):
        pc = eigenvectors[:, i].reshape(image_shape)
        im = axes[i].imshow(pc, cmap='RdBu_r', aspect='auto')
        axes[i].set_title(f'PC {i+1}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Hide extra subplots
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Principal Component Eigenvectors', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_segmentation(image: np.ndarray, 
                     segmentation: np.ndarray,
                     title: str = 'Tissue Segmentation',
                     save_path: Optional[str] = None):
    """
    Plot original image and segmentation side by side.
    
    Args:
        image: Original image
        segmentation: Segmentation map
        title: Figure title
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Segmentation
    im = ax2.imshow(segmentation, cmap='viridis')
    ax2.set_title(title)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_comparison(image: np.ndarray,
                   seg_pca: np.ndarray,
                   seg_original: np.ndarray,
                   save_path: Optional[str] = None):
    """
    Compare segmentation results from PCA and original space.
    
    Args:
        image: Original image
        seg_pca: Segmentation from PCA space
        seg_original: Segmentation from original space
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(seg_pca, cmap='viridis')
    axes[1].set_title('Segmentation (PCA Space)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(seg_original, cmap='viridis')
    axes[2].set_title('Segmentation (Original Space)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_pca_projection_2d(Y: np.ndarray, 
                          labels: np.ndarray,
                          save_path: Optional[str] = None):
    """
    Plot 2D projection of data in PC space with cluster labels.
    
    Args:
        Y: Data in PC space (uses first 2 components)
        labels: Cluster labels
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='viridis', 
                        alpha=0.6, s=1)
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Data Projection in PC Space (Colored by Cluster)')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_eigenvalues(eigenvalues: np.ndarray, 
                    n_components: int = 20,
                    save_path: Optional[str] = None):
    """
    Plot eigenvalue spectrum.
    
    Args:
        eigenvalues: Array of eigenvalues
        n_components: Number of eigenvalues to show
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_show = min(n_components, len(eigenvalues))
    components = np.arange(1, n_show + 1)
    
    ax.plot(components, eigenvalues[:n_show], 'o-', linewidth=2)
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Eigenvalue (λ)')
    ax.set_title('Eigenvalue Spectrum')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
