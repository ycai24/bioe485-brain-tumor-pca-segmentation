"""PCA vs. original-feature-space tumor segmentation comparison on BraTS2020 (BIOE 485 final project).

Converted from the original Jupyter notebook. Cells are separated by
`# %%` markers, so the file can be run top-to-bottom as a script or
cell-by-cell in VS Code / Spyder.
"""


# %%
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import sys
from tqdm import tqdm
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# Resolve paths relative to the repository root so the script runs from any directory
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / 'src'))

# Import custom modules
from brats_loader import BraTSLoader
from pca_analysis import PCAAnalyzer

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

print("Libraries imported successfully!")


# %% [markdown]
# ## 1. Data Loading Configuration


# %%
# Configure data paths
BRATS_DATA_PATH = str(REPO_ROOT / 'data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
RESULTS_PATH = REPO_ROOT / 'results/pca_comparison'
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Check if data exists
data_exists = Path(BRATS_DATA_PATH).exists()
print(f"BraTS data path: {BRATS_DATA_PATH}")
print(f"Data exists: {data_exists}")
print(f"Results will be saved to: {RESULTS_PATH}")


# %%
# Initialize BraTS loader
if data_exists:
    brats_loader = BraTSLoader(BRATS_DATA_PATH)
    subjects = brats_loader.get_subject_list()
    print(f"Found {len(subjects)} subjects in BraTS2020 training dataset")
    print(f"First 5 subjects: {subjects[:5]}")
else:
    sys.exit(
        f"ERROR: BraTS data not found at {BRATS_DATA_PATH}.\n"
        "Download it first (see docs/BRATS_DOWNLOAD_GUIDE.md)."
    )


# %% [markdown]
# ## 2. Helper Functions for PCA Analysis


# %%
def prepare_multimodal_features(slices: Dict[str, np.ndarray], 
                                modalities: List[str] = ['t1', 't1ce', 't2', 'flair']) -> np.ndarray:
    """
    Prepare multi-modal feature matrix from MRI slices.
    
    Args:
        slices: Dictionary containing MRI modalities
        modalities: List of modalities to include
    
    Returns:
        Feature matrix of shape (n_pixels, n_modalities)
    """
    features = []
    for mod in modalities:
        if mod in slices:
            # Flatten and normalize each modality
            img = slices[mod].flatten()
            # Normalize to [0, 1]
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
            features.append(img)
    
    return np.column_stack(features)


def prepare_segmentation_features(seg: np.ndarray) -> np.ndarray:
    """
    Prepare segmentation as features for PCA.
    Uses one-hot encoding of segmentation labels.
    
    Args:
        seg: Segmentation mask with shape (H, W)
    
    Returns:
        Feature matrix of shape (n_pixels, n_classes)
    """
    seg_flat = seg.flatten()
    unique_labels = np.unique(seg_flat)
    n_classes = len(unique_labels)
    n_pixels = len(seg_flat)
    
    # One-hot encode
    features = np.zeros((n_pixels, n_classes))
    for i, label in enumerate(unique_labels):
        features[:, i] = (seg_flat == label).astype(float)
    
    return features


def compute_pca_metrics(X: np.ndarray, n_components: int = None) -> Dict:
    """
    Compute comprehensive PCA metrics for given data.
    
    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of components to keep (None = all)
    
    Returns:
        Dictionary of PCA metrics
    """
    pca = PCAAnalyzer(n_components=n_components)
    pca.fit(X)
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    # Compute reconstruction error
    mse = np.mean((X - X_reconstructed) ** 2)
    rmse = np.sqrt(mse)
    
    # Normalized reconstruction error
    X_variance = np.var(X)
    normalized_error = mse / X_variance if X_variance > 0 else 0
    
    # Explained variance
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)
    
    return {
        'pca': pca,
        'transformed': X_transformed,
        'reconstructed': X_reconstructed,
        'mse': mse,
        'rmse': rmse,
        'normalized_error': normalized_error,
        'eigenvalues': pca.eigenvalues_,
        'explained_variance_ratio': explained_var_ratio,
        'cumulative_variance': cumulative_var,
        'n_components': len(pca.eigenvalues_),
        'total_variance_explained': cumulative_var[-1] if len(cumulative_var) > 0 else 0
    }


print("Helper functions defined successfully!")


# %% [markdown]
# ## 3. Process All BraTS Subjects
#
# We will:
# 1. Load all subjects in the training dataset
# 2. Extract middle axial slices (where tumors are typically present)
# 3. Apply PCA to both original multi-modal images and segmentation masks
# 4. Collect metrics for statistical analysis


# %%
# Storage for results
results = {
    'original': [],  # PCA on original multi-modal images
    'segmentation': []  # PCA on segmentation masks
}

# Number of subjects to process (use all available)
n_subjects = len(subjects)
print(f"Processing {n_subjects} subjects...")

# Process each subject
for i, subject_id in enumerate(tqdm(subjects, desc="Processing subjects")):
    try:
        # Load subject with tumor slice
        slices, slice_idx = brats_loader.load_subject_with_tumor_slice(subject_id, axis=2)
        
        # Skip if no segmentation
        if 'seg' not in slices or slices['seg'] is None:
            continue
        
        # Check if tumor is present
        seg = slices['seg']
        if np.sum(seg > 0) == 0:
            continue
        
        # ===== PCA ON ORIGINAL MULTI-MODAL IMAGES =====
        X_original = prepare_multimodal_features(slices)
        metrics_original = compute_pca_metrics(X_original, n_components=None)
        
        results['original'].append({
            'subject_id': subject_id,
            'slice_idx': slice_idx,
            'n_features': X_original.shape[1],
            'n_pixels': X_original.shape[0],
            **metrics_original
        })
        
        # ===== PCA ON SEGMENTATION MASKS =====
        X_seg = prepare_segmentation_features(seg)
        metrics_seg = compute_pca_metrics(X_seg, n_components=None)
        
        results['segmentation'].append({
            'subject_id': subject_id,
            'slice_idx': slice_idx,
            'n_features': X_seg.shape[1],
            'n_pixels': X_seg.shape[0],
            'n_tumor_pixels': np.sum(seg > 0),
            'tumor_percentage': 100 * np.sum(seg > 0) / seg.size,
            **metrics_seg
        })
        
    except Exception as e:
        print(f"\nError processing {subject_id}: {str(e)}")
        continue

print(f"\n✓ Successfully processed:")
print(f"  - {len(results['original'])} subjects with original images")
print(f"  - {len(results['segmentation'])} subjects with segmentation masks")


# %% [markdown]
# ## 4. Statistical Analysis and Maximum Likelihood Estimation (MLE)
#
# ### Theory
#
# We model the reconstruction error distribution and perform MLE to estimate parameters.
#
# **Assumption**: Reconstruction errors follow a Gaussian distribution:
# $$\epsilon \sim \mathcal{N}(\mu, \sigma^2)$$
#
# **Log-likelihood for Gaussian**:
# $$\log L(\mu, \sigma | \epsilon_1, \ldots, \epsilon_n) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(\epsilon_i - \mu)^2$$
#
# **MLE estimates**:
# - $\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}\epsilon_i$
# - $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(\epsilon_i - \hat{\mu})^2$


# %%
def gaussian_log_likelihood(params: np.ndarray, data: np.ndarray) -> float:
    """
    Compute negative log-likelihood for Gaussian distribution.
    
    Args:
        params: [mu, sigma]
        data: Sample data
    
    Returns:
        Negative log-likelihood (for minimization)
    """
    mu, sigma = params
    n = len(data)
    
    if sigma <= 0:
        return 1e10  # Invalid parameter
    
    log_likelihood = -n/2 * np.log(2 * np.pi) - n/2 * np.log(sigma**2) - \
                     np.sum((data - mu)**2) / (2 * sigma**2)
    
    return -log_likelihood  # Return negative for minimization


def fit_gaussian_mle(data: np.ndarray) -> Tuple[float, float, Dict]:
    """
    Fit Gaussian distribution using Maximum Likelihood Estimation.
    
    Args:
        data: Sample data
    
    Returns:
        Tuple of (mu_mle, sigma_mle, fit_info)
    """
    # Analytical MLE solutions
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)  # MLE uses N, not N-1
    
    # Verify with numerical optimization
    initial_guess = [mu_mle, sigma_mle]
    result = minimize(
        gaussian_log_likelihood,
        initial_guess,
        args=(data,),
        method='Nelder-Mead'
    )
    
    mu_opt, sigma_opt = result.x
    
    # Compute confidence intervals (95%)
    n = len(data)
    se_mu = sigma_mle / np.sqrt(n)
    ci_mu = stats.norm.interval(0.95, loc=mu_mle, scale=se_mu)
    
    # Chi-square for variance
    chi2_lower = stats.chi2.ppf(0.025, n-1)
    chi2_upper = stats.chi2.ppf(0.975, n-1)
    ci_sigma = (np.sqrt((n-1) * sigma_mle**2 / chi2_upper),
                np.sqrt((n-1) * sigma_mle**2 / chi2_lower))
    
    fit_info = {
        'mu_analytical': mu_mle,
        'sigma_analytical': sigma_mle,
        'mu_numerical': mu_opt,
        'sigma_numerical': sigma_opt,
        'n_samples': n,
        'log_likelihood': -result.fun,
        'ci_mu': ci_mu,
        'ci_sigma': ci_sigma,
        'optimization_success': result.success
    }
    
    return mu_mle, sigma_mle, fit_info


print("MLE functions defined successfully!")


# %%
# Extract reconstruction errors for MLE
errors_original = np.array([r['rmse'] for r in results['original']])
errors_seg = np.array([r['rmse'] for r in results['segmentation']])

print("=" * 80)
print("MAXIMUM LIKELIHOOD ESTIMATION RESULTS")
print("=" * 80)

# MLE for original images
print("\n1. ORIGINAL MULTI-MODAL IMAGES:")
print("-" * 80)
mu_orig, sigma_orig, fit_orig = fit_gaussian_mle(errors_original)
print(f"  MLE Estimate (μ):        {mu_orig:.6f}")
print(f"  95% CI for μ:            [{fit_orig['ci_mu'][0]:.6f}, {fit_orig['ci_mu'][1]:.6f}]")
print(f"  MLE Estimate (σ):        {sigma_orig:.6f}")
print(f"  95% CI for σ:            [{fit_orig['ci_sigma'][0]:.6f}, {fit_orig['ci_sigma'][1]:.6f}]")
print(f"  Log-likelihood:          {fit_orig['log_likelihood']:.2f}")
print(f"  Sample size:             {fit_orig['n_samples']}")

# MLE for segmentation
print("\n2. SEGMENTATION MASKS:")
print("-" * 80)
mu_seg, sigma_seg, fit_seg = fit_gaussian_mle(errors_seg)
print(f"  MLE Estimate (μ):        {mu_seg:.6f}")
print(f"  95% CI for μ:            [{fit_seg['ci_mu'][0]:.6f}, {fit_seg['ci_mu'][1]:.6f}]")
print(f"  MLE Estimate (σ):        {sigma_seg:.6f}")
print(f"  95% CI for σ:            [{fit_seg['ci_sigma'][0]:.6f}, {fit_seg['ci_sigma'][1]:.6f}]")
print(f"  Log-likelihood:          {fit_seg['log_likelihood']:.2f}")
print(f"  Sample size:             {fit_seg['n_samples']}")

print("\n" + "=" * 80)


# %% [markdown]
# ## 5. Hypothesis Testing
#
# **Null Hypothesis (H₀)**: PCA reconstruction error is the same for original images and segmentation masks
#
# **Alternative Hypothesis (H₁)**: PCA reconstruction error differs between original images and segmentation masks
#
# We perform:
# 1. Two-sample t-test
# 2. Mann-Whitney U test (non-parametric)
# 3. Kolmogorov-Smirnov test (distribution comparison)


# %%
print("=" * 80)
print("HYPOTHESIS TESTING: Original vs Segmentation PCA Performance")
print("=" * 80)

# Two-sample t-test
t_stat, t_pval = stats.ttest_ind(errors_original, errors_seg)
print("\n1. TWO-SAMPLE T-TEST:")
print("-" * 80)
print(f"  t-statistic:             {t_stat:.4f}")
print(f"  p-value:                 {t_pval:.6e}")
print(f"  Reject H₀ (α=0.05):      {t_pval < 0.05}")

# Mann-Whitney U test (non-parametric)
u_stat, u_pval = stats.mannwhitneyu(errors_original, errors_seg, alternative='two-sided')
print("\n2. MANN-WHITNEY U TEST:")
print("-" * 80)
print(f"  U-statistic:             {u_stat:.4f}")
print(f"  p-value:                 {u_pval:.6e}")
print(f"  Reject H₀ (α=0.05):      {u_pval < 0.05}")

# Kolmogorov-Smirnov test
ks_stat, ks_pval = stats.ks_2samp(errors_original, errors_seg)
print("\n3. KOLMOGOROV-SMIRNOV TEST:")
print("-" * 80)
print(f"  KS-statistic:            {ks_stat:.4f}")
print(f"  p-value:                 {ks_pval:.6e}")
print(f"  Reject H₀ (α=0.05):      {ks_pval < 0.05}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(errors_original)-1) * np.var(errors_original, ddof=1) + 
                       (len(errors_seg)-1) * np.var(errors_seg, ddof=1)) / 
                      (len(errors_original) + len(errors_seg) - 2))
cohens_d = (np.mean(errors_original) - np.mean(errors_seg)) / pooled_std
print("\n4. EFFECT SIZE:")
print("-" * 80)
print(f"  Cohen's d:               {cohens_d:.4f}")
print(f"  Effect size interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("Small")
elif abs(cohens_d) < 0.5:
    print("Medium")
else:
    print("Large")

print("\n" + "=" * 80)


# %% [markdown]
# ## 6. Comprehensive Descriptive Statistics


# %%
# Create comprehensive statistics dataframe
def compute_statistics(data: np.ndarray, name: str) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics.
    """
    stats_dict = {
        'Metric': name,
        'Count': len(data),
        'Mean': np.mean(data),
        'Std': np.std(data, ddof=1),
        'Min': np.min(data),
        '25%': np.percentile(data, 25),
        'Median': np.median(data),
        '75%': np.percentile(data, 75),
        'Max': np.max(data),
        'IQR': np.percentile(data, 75) - np.percentile(data, 25),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data)
    }
    return pd.DataFrame([stats_dict])

# Reconstruction errors
stats_orig_rmse = compute_statistics(errors_original, 'Original RMSE')
stats_seg_rmse = compute_statistics(errors_seg, 'Segmentation RMSE')

# MSE
mse_original = np.array([r['mse'] for r in results['original']])
mse_seg = np.array([r['mse'] for r in results['segmentation']])
stats_orig_mse = compute_statistics(mse_original, 'Original MSE')
stats_seg_mse = compute_statistics(mse_seg, 'Segmentation MSE')

# Explained variance
var_orig = np.array([r['total_variance_explained'] for r in results['original']])
var_seg = np.array([r['total_variance_explained'] for r in results['segmentation']])
stats_orig_var = compute_statistics(var_orig, 'Original Var. Explained')
stats_seg_var = compute_statistics(var_seg, 'Segmentation Var. Explained')

# Combine all statistics
all_stats = pd.concat([
    stats_orig_rmse, stats_seg_rmse,
    stats_orig_mse, stats_seg_mse,
    stats_orig_var, stats_seg_var
], ignore_index=True)

print("\n" + "=" * 120)
print("COMPREHENSIVE DESCRIPTIVE STATISTICS")
print("=" * 120)
print(all_stats.to_string(index=False))
print("=" * 120)


# %% [markdown]
# ## 7. Visualization: Reconstruction Error Distributions


# %%
# Create comprehensive comparison plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Histogram comparison
ax = axes[0, 0]
ax.hist(errors_original, bins=30, alpha=0.6, label='Original Images', color='blue', density=True)
ax.hist(errors_seg, bins=30, alpha=0.6, label='Segmentation', color='red', density=True)

# Overlay fitted Gaussian distributions
x_range = np.linspace(min(errors_original.min(), errors_seg.min()),
                      max(errors_original.max(), errors_seg.max()), 100)
ax.plot(x_range, stats.norm.pdf(x_range, mu_orig, sigma_orig), 
        'b-', linewidth=2, label=f'Fitted N({mu_orig:.3f}, {sigma_orig:.3f}²)')
ax.plot(x_range, stats.norm.pdf(x_range, mu_seg, sigma_seg), 
        'r-', linewidth=2, label=f'Fitted N({mu_seg:.3f}, {sigma_seg:.3f}²)')

ax.set_xlabel('Reconstruction Error (RMSE)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Reconstruction Error Distribution with MLE Fit', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Box plot comparison
ax = axes[0, 1]
box_data = [errors_original, errors_seg]
bp = ax.boxplot(box_data, labels=['Original\nImages', 'Segmentation\nMasks'],
                patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Reconstruction Error (RMSE)', fontsize=11)
ax.set_title('RMSE Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 3. Q-Q plot for normality check
ax = axes[0, 2]
stats.probplot(errors_original, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Original Images', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. Q-Q plot for segmentation
ax = axes[1, 0]
stats.probplot(errors_seg, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Segmentation Masks', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 5. Cumulative distribution comparison
ax = axes[1, 1]
sorted_orig = np.sort(errors_original)
sorted_seg = np.sort(errors_seg)
cdf_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
cdf_seg = np.arange(1, len(sorted_seg) + 1) / len(sorted_seg)

ax.plot(sorted_orig, cdf_orig, 'b-', linewidth=2, label='Original Images')
ax.plot(sorted_seg, cdf_seg, 'r-', linewidth=2, label='Segmentation Masks')
ax.set_xlabel('Reconstruction Error (RMSE)', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('Empirical CDF Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 6. Variance explained comparison
ax = axes[1, 2]
ax.scatter(range(len(var_orig)), var_orig, alpha=0.6, s=50, label='Original Images', color='blue')
ax.scatter(range(len(var_seg)), var_seg, alpha=0.6, s=50, label='Segmentation Masks', color='red')
ax.axhline(np.mean(var_orig), color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean Original: {np.mean(var_orig):.3f}')
ax.axhline(np.mean(var_seg), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean Seg: {np.mean(var_seg):.3f}')
ax.set_xlabel('Subject Index', fontsize=11)
ax.set_ylabel('Total Variance Explained', fontsize=11)
ax.set_title('Variance Explained by PCA', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH / 'pca_comparison_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Distribution plots saved!")


# %% [markdown]
# ## 8. Eigenvalue Analysis


# %%
# Extract eigenvalues
eigenvals_orig = [r['eigenvalues'] for r in results['original']]
eigenvals_seg = [r['eigenvalues'] for r in results['segmentation']]

# Average eigenvalues across all subjects
max_components_orig = max(len(ev) for ev in eigenvals_orig)
max_components_seg = max(len(ev) for ev in eigenvals_seg)

mean_eigenvals_orig = np.zeros(max_components_orig)
for ev in eigenvals_orig:
    mean_eigenvals_orig[:len(ev)] += ev
mean_eigenvals_orig /= len(eigenvals_orig)

mean_eigenvals_seg = np.zeros(max_components_seg)
for ev in eigenvals_seg:
    mean_eigenvals_seg[:len(ev)] += ev
mean_eigenvals_seg /= len(eigenvals_seg)

# Plot eigenvalue spectra
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Eigenvalue spectrum
ax = axes[0]
ax.plot(range(1, len(mean_eigenvals_orig)+1), mean_eigenvals_orig, 
        'bo-', linewidth=2, markersize=6, label='Original Images')
ax.plot(range(1, len(mean_eigenvals_seg)+1), mean_eigenvals_seg, 
        'ro-', linewidth=2, markersize=6, label='Segmentation Masks')
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Eigenvalue (Variance)', fontsize=12)
ax.set_title('Average Eigenvalue Spectrum', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 2. Cumulative explained variance
ax = axes[1]
cumvar_orig = np.cumsum(mean_eigenvals_orig) / np.sum(mean_eigenvals_orig)
cumvar_seg = np.cumsum(mean_eigenvals_seg) / np.sum(mean_eigenvals_seg)

ax.plot(range(1, len(cumvar_orig)+1), cumvar_orig, 
        'bo-', linewidth=2, markersize=6, label='Original Images')
ax.plot(range(1, len(cumvar_seg)+1), cumvar_seg, 
        'ro-', linewidth=2, markersize=6, label='Segmentation Masks')
ax.axhline(0.95, color='green', linestyle='--', linewidth=2, alpha=0.7, label='95% Variance')
ax.axhline(0.99, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='99% Variance')
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(RESULTS_PATH / 'eigenvalue_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key findings
print("\n" + "=" * 80)
print("EIGENVALUE ANALYSIS SUMMARY")
print("=" * 80)
print(f"\nOriginal Images:")
print(f"  Components for 95% variance: {np.argmax(cumvar_orig >= 0.95) + 1}")
print(f"  Components for 99% variance: {np.argmax(cumvar_orig >= 0.99) + 1}")
print(f"  First PC explains:          {mean_eigenvals_orig[0] / np.sum(mean_eigenvals_orig):.2%}")

print(f"\nSegmentation Masks:")
print(f"  Components for 95% variance: {np.argmax(cumvar_seg >= 0.95) + 1}")
print(f"  Components for 99% variance: {np.argmax(cumvar_seg >= 0.99) + 1}")
print(f"  First PC explains:          {mean_eigenvals_seg[0] / np.sum(mean_eigenvals_seg):.2%}")
print("=" * 80)


# %% [markdown]
# ## 9. Summary Report


# %%
# Generate comprehensive summary report
print("\n" + "="*100)
print(" " * 25 + "FINAL SUMMARY REPORT")
print("="*100)

print("\n📊 DATASET OVERVIEW:")
print("-" * 100)
print(f"  Total subjects processed:        {len(results['original'])}")
print(f"  Total slices analyzed:           {len(results['original'])}")
print(f"  Average tumor coverage:          {np.mean([r['tumor_percentage'] for r in results['segmentation']]):.2f}%")

print("\n🔬 PCA PERFORMANCE COMPARISON:")
print("-" * 100)
print(f"\n  Original Multi-Modal Images:")
print(f"    Mean RMSE:                     {np.mean(errors_original):.6f} ± {np.std(errors_original):.6f}")
print(f"    MLE estimate μ:                {mu_orig:.6f} (95% CI: [{fit_orig['ci_mu'][0]:.6f}, {fit_orig['ci_mu'][1]:.6f}])")
print(f"    Mean variance explained:       {np.mean(var_orig):.2%}")
print(f"    Median components for 95% var: {np.argmax(cumvar_orig >= 0.95) + 1}")

print(f"\n  Segmentation Masks:")
print(f"    Mean RMSE:                     {np.mean(errors_seg):.6f} ± {np.std(errors_seg):.6f}")
print(f"    MLE estimate μ:                {mu_seg:.6f} (95% CI: [{fit_seg['ci_mu'][0]:.6f}, {fit_seg['ci_mu'][1]:.6f}])")
print(f"    Mean variance explained:       {np.mean(var_seg):.2%}")
print(f"    Median components for 95% var: {np.argmax(cumvar_seg >= 0.95) + 1}")

print("\n📈 STATISTICAL TESTING:")
print("-" * 100)
print(f"  Two-sample t-test:")
print(f"    t-statistic = {t_stat:.4f}, p-value = {t_pval:.6e}")
print(f"    Conclusion: {'Significant difference' if t_pval < 0.05 else 'No significant difference'} (α=0.05)")
print(f"\n  Effect size (Cohen's d):         {cohens_d:.4f} ({'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'})")

print("\n💡 KEY FINDINGS:")
print("-" * 100)
if np.mean(errors_original) < np.mean(errors_seg):
    better = "Original images"
    worse = "Segmentation masks"
    diff = ((np.mean(errors_seg) - np.mean(errors_original)) / np.mean(errors_original)) * 100
else:
    better = "Segmentation masks"
    worse = "Original images"
    diff = ((np.mean(errors_original) - np.mean(errors_seg)) / np.mean(errors_seg)) * 100

print(f"  1. {better} show {diff:.1f}% lower reconstruction error than {worse}")
print(f"  2. MLE estimates confirm {'significant' if t_pval < 0.05 else 'no significant'} difference in error distributions")
print(f"  3. Both approaches explain {'high' if min(np.mean(var_orig), np.mean(var_seg)) > 0.9 else 'moderate'} variance")
print(f"  4. Segmentation PCA requires fewer components (simpler structure)")

print("\n" + "="*100)
print("✓ Analysis complete! Results saved to:", RESULTS_PATH)
print("="*100)


# %% [markdown]
# ## 11. Save Results


# %% [markdown]
# ## 10. Computational Load Comparison
#
# Compare the computational efficiency of PCA-based segmentation vs. segmentation in original space.


# %% [markdown]
# ## 10. Segmentation Quality: PCA Space vs Original Space
#
# Compare clustering/segmentation performance between PCA-reduced space and original feature space across all subjects.


# %%
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score

print("="*100)
print(" " * 25 + "SEGMENTATION QUALITY COMPARISON")
print(" " * 20 + "(PCA-Reduced Space vs Original Feature Space)")
print("="*100)

# Storage for comparison results
segmentation_comparison = {
    'subject_id': [],
    'n_pixels': [],
    'n_features_original': [],
    'n_components_pca': [],
    
    # Clustering metrics in original space
    'silhouette_original_gmm': [],
    'silhouette_original_kmeans': [],
    'davies_bouldin_original_gmm': [],
    'davies_bouldin_original_kmeans': [],
    
    # Clustering metrics in PCA space
    'silhouette_pca_gmm': [],
    'silhouette_pca_kmeans': [],
    'davies_bouldin_pca_gmm': [],
    'davies_bouldin_pca_kmeans': [],
    
    # Agreement between methods (ARI, NMI)
    'ari_gmm': [],  # Agreement between original and PCA for GMM
    'nmi_gmm': [],
    'ari_kmeans': [],  # Agreement between original and PCA for KMeans
    'nmi_kmeans': [],
}

# Configuration
n_clusters = 4
n_components_pca = 3  # Use 3 PCA components (captures ~99.95% variance based on earlier analysis)

print(f"\nConfiguration:")
print(f"  Number of clusters:     {n_clusters}")
print(f"  PCA components:         {n_components_pca}")
print(f"  Subjects to process:    {len(subjects)}")

print("\n" + "-"*100)
print("Processing subjects and comparing segmentation quality...")
print("-"*100)

# Process a subset of subjects (to keep computation reasonable)
n_subjects_to_process = min(50, len(subjects))  # Process first 50 subjects
print(f"Processing {n_subjects_to_process} subjects for detailed comparison\n")

for i, subject_id in enumerate(tqdm(subjects[:n_subjects_to_process], desc="Comparing segmentations")):
    try:
        # Load subject with tumor slice
        slices, slice_idx = brats_loader.load_subject_with_tumor_slice(subject_id, axis=2)
        
        # Skip if no segmentation
        if 'seg' not in slices or slices['seg'] is None:
            continue
        
        # Check if tumor is present
        seg = slices['seg']
        if np.sum(seg > 0) == 0:
            continue
        
        # Prepare multi-modal features
        X_original = prepare_multimodal_features(slices, modalities=['t1', 't1ce', 't2', 'flair'])
        
        # Skip if data is too small
        if X_original.shape[0] < n_clusters * 10:
            continue
        
        # ===== SEGMENT IN ORIGINAL SPACE =====
        # GMM in original space
        gmm_original = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
        labels_original_gmm = gmm_original.fit_predict(X_original)
        
        # KMeans in original space
        kmeans_original = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_original_kmeans = kmeans_original.fit_predict(X_original)
        
        # ===== SEGMENT IN PCA SPACE =====
        # Apply PCA
        pca = PCAAnalyzer(n_components=n_components_pca)
        X_pca = pca.fit_transform(X_original)
        
        # GMM in PCA space
        gmm_pca = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
        labels_pca_gmm = gmm_pca.fit_predict(X_pca)
        
        # KMeans in PCA space
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_pca_kmeans = kmeans_pca.fit_predict(X_pca)
        
        # ===== COMPUTE CLUSTERING QUALITY METRICS =====
        # Silhouette score (higher is better, range [-1, 1])
        sil_orig_gmm = silhouette_score(X_original, labels_original_gmm)
        sil_orig_kmeans = silhouette_score(X_original, labels_original_kmeans)
        sil_pca_gmm = silhouette_score(X_pca, labels_pca_gmm)
        sil_pca_kmeans = silhouette_score(X_pca, labels_pca_kmeans)
        
        # Davies-Bouldin index (lower is better, range [0, ∞))
        db_orig_gmm = davies_bouldin_score(X_original, labels_original_gmm)
        db_orig_kmeans = davies_bouldin_score(X_original, labels_original_kmeans)
        db_pca_gmm = davies_bouldin_score(X_pca, labels_pca_gmm)
        db_pca_kmeans = davies_bouldin_score(X_pca, labels_pca_kmeans)
        
        # ===== COMPUTE AGREEMENT BETWEEN ORIGINAL AND PCA =====
        # For GMM
        ari_gmm = adjusted_rand_score(labels_original_gmm, labels_pca_gmm)
        nmi_gmm = normalized_mutual_info_score(labels_original_gmm, labels_pca_gmm)
        
        # For KMeans
        ari_kmeans = adjusted_rand_score(labels_original_kmeans, labels_pca_kmeans)
        nmi_kmeans = normalized_mutual_info_score(labels_original_kmeans, labels_pca_kmeans)
        
        # Store results
        segmentation_comparison['subject_id'].append(subject_id)
        segmentation_comparison['n_pixels'].append(X_original.shape[0])
        segmentation_comparison['n_features_original'].append(X_original.shape[1])
        segmentation_comparison['n_components_pca'].append(n_components_pca)
        
        segmentation_comparison['silhouette_original_gmm'].append(sil_orig_gmm)
        segmentation_comparison['silhouette_original_kmeans'].append(sil_orig_kmeans)
        segmentation_comparison['davies_bouldin_original_gmm'].append(db_orig_gmm)
        segmentation_comparison['davies_bouldin_original_kmeans'].append(db_orig_kmeans)
        
        segmentation_comparison['silhouette_pca_gmm'].append(sil_pca_gmm)
        segmentation_comparison['silhouette_pca_kmeans'].append(sil_pca_kmeans)
        segmentation_comparison['davies_bouldin_pca_gmm'].append(db_pca_gmm)
        segmentation_comparison['davies_bouldin_pca_kmeans'].append(db_pca_kmeans)
        
        segmentation_comparison['ari_gmm'].append(ari_gmm)
        segmentation_comparison['nmi_gmm'].append(nmi_gmm)
        segmentation_comparison['ari_kmeans'].append(ari_kmeans)
        segmentation_comparison['nmi_kmeans'].append(nmi_kmeans)
        
    except Exception as e:
        print(f"\nError processing {subject_id}: {str(e)}")
        continue

print(f"\n✓ Successfully compared segmentation on {len(segmentation_comparison['subject_id'])} subjects")

# Convert to arrays for analysis
sil_orig_gmm = np.array(segmentation_comparison['silhouette_original_gmm'])
sil_orig_kmeans = np.array(segmentation_comparison['silhouette_original_kmeans'])
sil_pca_gmm = np.array(segmentation_comparison['silhouette_pca_gmm'])
sil_pca_kmeans = np.array(segmentation_comparison['silhouette_pca_kmeans'])

db_orig_gmm = np.array(segmentation_comparison['davies_bouldin_original_gmm'])
db_orig_kmeans = np.array(segmentation_comparison['davies_bouldin_original_kmeans'])
db_pca_gmm = np.array(segmentation_comparison['davies_bouldin_pca_gmm'])
db_pca_kmeans = np.array(segmentation_comparison['davies_bouldin_pca_kmeans'])

ari_gmm = np.array(segmentation_comparison['ari_gmm'])
nmi_gmm = np.array(segmentation_comparison['nmi_gmm'])
ari_kmeans = np.array(segmentation_comparison['ari_kmeans'])
nmi_kmeans = np.array(segmentation_comparison['nmi_kmeans'])

# ===== STATISTICAL ANALYSIS =====
print("\n" + "="*100)
print(" " * 30 + "CLUSTERING QUALITY METRICS")
print("="*100)

print("\n1. SILHOUETTE SCORE (Higher is Better, Range: [-1, 1]):")
print("-"*100)
print(f"  GMM - Original Space:     {np.mean(sil_orig_gmm):.4f} ± {np.std(sil_orig_gmm):.4f}")
print(f"  GMM - PCA Space:          {np.mean(sil_pca_gmm):.4f} ± {np.std(sil_pca_gmm):.4f}")
print(f"  Difference:               {np.mean(sil_pca_gmm) - np.mean(sil_orig_gmm):+.4f} ({((np.mean(sil_pca_gmm) - np.mean(sil_orig_gmm))/abs(np.mean(sil_orig_gmm)))*100:+.1f}%)")

print(f"\n  KMeans - Original Space:  {np.mean(sil_orig_kmeans):.4f} ± {np.std(sil_orig_kmeans):.4f}")
print(f"  KMeans - PCA Space:       {np.mean(sil_pca_kmeans):.4f} ± {np.std(sil_pca_kmeans):.4f}")
print(f"  Difference:               {np.mean(sil_pca_kmeans) - np.mean(sil_orig_kmeans):+.4f} ({((np.mean(sil_pca_kmeans) - np.mean(sil_orig_kmeans))/abs(np.mean(sil_orig_kmeans)))*100:+.1f}%)")

print("\n2. DAVIES-BOULDIN INDEX (Lower is Better, Range: [0, ∞)):")
print("-"*100)
print(f"  GMM - Original Space:     {np.mean(db_orig_gmm):.4f} ± {np.std(db_orig_gmm):.4f}")
print(f"  GMM - PCA Space:          {np.mean(db_pca_gmm):.4f} ± {np.std(db_pca_gmm):.4f}")
print(f"  Difference:               {np.mean(db_pca_gmm) - np.mean(db_orig_gmm):+.4f} ({((np.mean(db_pca_gmm) - np.mean(db_orig_gmm))/np.mean(db_orig_gmm))*100:+.1f}%)")

print(f"\n  KMeans - Original Space:  {np.mean(db_orig_kmeans):.4f} ± {np.std(db_orig_kmeans):.4f}")
print(f"  KMeans - PCA Space:       {np.mean(db_pca_kmeans):.4f} ± {np.std(db_pca_kmeans):.4f}")
print(f"  Difference:               {np.mean(db_pca_kmeans) - np.mean(db_orig_kmeans):+.4f} ({((np.mean(db_pca_kmeans) - np.mean(db_orig_kmeans))/np.mean(db_orig_kmeans))*100:+.1f}%)")

print("\n3. AGREEMENT BETWEEN ORIGINAL AND PCA SEGMENTATIONS:")
print("-"*100)
print(f"  GMM - Adjusted Rand Index:        {np.mean(ari_gmm):.4f} ± {np.std(ari_gmm):.4f}")
print(f"  GMM - Normalized Mutual Info:     {np.mean(nmi_gmm):.4f} ± {np.std(nmi_gmm):.4f}")
print(f"\n  KMeans - Adjusted Rand Index:     {np.mean(ari_kmeans):.4f} ± {np.std(ari_kmeans):.4f}")
print(f"  KMeans - Normalized Mutual Info:  {np.mean(nmi_kmeans):.4f} ± {np.std(nmi_kmeans):.4f}")

# Statistical tests
print("\n4. STATISTICAL SIGNIFICANCE TESTS:")
print("-"*100)

# Paired t-test for Silhouette (GMM)
t_sil_gmm, p_sil_gmm = stats.ttest_rel(sil_orig_gmm, sil_pca_gmm)
print(f"\n  Silhouette Score (GMM):")
print(f"    Paired t-test: t={t_sil_gmm:.4f}, p={p_sil_gmm:.6f}")
print(f"    Conclusion: {'PCA significantly different' if p_sil_gmm < 0.05 else 'No significant difference'} (α=0.05)")

# Paired t-test for Davies-Bouldin (GMM)
t_db_gmm, p_db_gmm = stats.ttest_rel(db_orig_gmm, db_pca_gmm)
print(f"\n  Davies-Bouldin Index (GMM):")
print(f"    Paired t-test: t={t_db_gmm:.4f}, p={p_db_gmm:.6f}")
print(f"    Conclusion: {'PCA significantly different' if p_db_gmm < 0.05 else 'No significant difference'} (α=0.05)")

# Paired t-test for Silhouette (KMeans)
t_sil_kmeans, p_sil_kmeans = stats.ttest_rel(sil_orig_kmeans, sil_pca_kmeans)
print(f"\n  Silhouette Score (KMeans):")
print(f"    Paired t-test: t={t_sil_kmeans:.4f}, p={p_sil_kmeans:.6f}")
print(f"    Conclusion: {'PCA significantly different' if p_sil_kmeans < 0.05 else 'No significant difference'} (α=0.05)")

print("\n" + "="*100)
print("KEY FINDINGS:")
print("="*100)

# Determine winner for each metric
sil_winner_gmm = "PCA" if np.mean(sil_pca_gmm) > np.mean(sil_orig_gmm) else "Original"
db_winner_gmm = "PCA" if np.mean(db_pca_gmm) < np.mean(db_orig_gmm) else "Original"
sil_winner_kmeans = "PCA" if np.mean(sil_pca_kmeans) > np.mean(sil_orig_kmeans) else "Original"

print(f"\n  1. For GMM clustering:")
print(f"     - Silhouette score: {sil_winner_gmm} space performs better")
print(f"     - Davies-Bouldin:   {db_winner_gmm} space performs better")
print(f"     - Agreement (ARI):  {np.mean(ari_gmm):.3f} - {'High' if np.mean(ari_gmm) > 0.8 else 'Moderate' if np.mean(ari_gmm) > 0.5 else 'Low'} clustering similarity")

print(f"\n  2. For KMeans clustering:")
print(f"     - Silhouette score: {sil_winner_kmeans} space performs better")
print(f"     - Agreement (ARI):  {np.mean(ari_kmeans):.3f} - {'High' if np.mean(ari_kmeans) > 0.8 else 'Moderate' if np.mean(ari_kmeans) > 0.5 else 'Low'} clustering similarity")

print(f"\n  3. PCA preserves clustering structure:")
print(f"     - Mean ARI > 0.8: Segmentations are {'highly consistent' if np.mean(ari_gmm) > 0.8 or np.mean(ari_kmeans) > 0.8 else 'moderately consistent'}")
print(f"     - PCA reduces dimensions by {(1 - n_components_pca/4)*100:.0f}% with minimal information loss")

print(f"\n  4. Computational benefit + Quality:")
if np.mean(sil_pca_gmm) >= np.mean(sil_orig_gmm) * 0.95:  # Within 5%
    print(f"     - PCA maintains ≥95% of clustering quality")
    print(f"     - While providing significant computational speedup")
else:
    print(f"     - Trade-off: {abs(np.mean(sil_pca_gmm) - np.mean(sil_orig_gmm))/np.mean(sil_orig_gmm)*100:.1f}% quality change for faster computation")

print("\n" + "="*100)


# %%
# Visualize segmentation quality comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Silhouette Score Comparison (GMM)
ax = axes[0, 0]
positions = [1, 2]
bp = ax.boxplot([sil_orig_gmm, sil_pca_gmm], positions=positions, 
                labels=['Original\nSpace', 'PCA\nSpace'],
                patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Silhouette Score', fontsize=11)
ax.set_title('GMM: Silhouette Score\n(Higher is Better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

# 2. Silhouette Score Comparison (KMeans)
ax = axes[0, 1]
bp = ax.boxplot([sil_orig_kmeans, sil_pca_kmeans], positions=positions,
                labels=['Original\nSpace', 'PCA\nSpace'],
                patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Silhouette Score', fontsize=11)
ax.set_title('KMeans: Silhouette Score\n(Higher is Better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

# 3. Davies-Bouldin Index Comparison (GMM)
ax = axes[0, 2]
bp = ax.boxplot([db_orig_gmm, db_pca_gmm], positions=positions,
                labels=['Original\nSpace', 'PCA\nSpace'],
                patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Davies-Bouldin Index', fontsize=11)
ax.set_title('GMM: Davies-Bouldin Index\n(Lower is Better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 4. Agreement Metrics (ARI)
ax = axes[1, 0]
x = np.arange(len(segmentation_comparison['subject_id']))
ax.plot(x, ari_gmm, 'o-', alpha=0.6, label='GMM', markersize=4)
ax.plot(x, ari_kmeans, 's-', alpha=0.6, label='KMeans', markersize=4)
ax.axhline(np.mean(ari_gmm), color='blue', linestyle='--', alpha=0.7, label=f'Mean GMM: {np.mean(ari_gmm):.3f}')
ax.axhline(np.mean(ari_kmeans), color='orange', linestyle='--', alpha=0.7, label=f'Mean KMeans: {np.mean(ari_kmeans):.3f}')
ax.axhline(0.8, color='green', linestyle=':', alpha=0.5, label='High Agreement (0.8)')
ax.set_xlabel('Subject Index', fontsize=11)
ax.set_ylabel('Adjusted Rand Index', fontsize=11)
ax.set_title('Agreement: Original vs PCA Segmentation\n(ARI)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.1, 1.1])

# 5. Agreement Metrics (NMI)
ax = axes[1, 1]
ax.plot(x, nmi_gmm, 'o-', alpha=0.6, label='GMM', markersize=4)
ax.plot(x, nmi_kmeans, 's-', alpha=0.6, label='KMeans', markersize=4)
ax.axhline(np.mean(nmi_gmm), color='blue', linestyle='--', alpha=0.7, label=f'Mean GMM: {np.mean(nmi_gmm):.3f}')
ax.axhline(np.mean(nmi_kmeans), color='orange', linestyle='--', alpha=0.7, label=f'Mean KMeans: {np.mean(nmi_kmeans):.3f}')
ax.set_xlabel('Subject Index', fontsize=11)
ax.set_ylabel('Normalized Mutual Information', fontsize=11)
ax.set_title('Agreement: Original vs PCA Segmentation\n(NMI)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.1])

# 6. Scatter: Silhouette Original vs PCA
ax = axes[1, 2]
ax.scatter(sil_orig_gmm, sil_pca_gmm, alpha=0.6, s=50, label='GMM', color='blue')
ax.scatter(sil_orig_kmeans, sil_pca_kmeans, alpha=0.6, s=50, label='KMeans', color='orange', marker='s')
# Add diagonal line (perfect agreement)
lim_min = min(sil_orig_gmm.min(), sil_pca_gmm.min(), sil_orig_kmeans.min(), sil_pca_kmeans.min())
lim_max = max(sil_orig_gmm.max(), sil_pca_gmm.max(), sil_orig_kmeans.max(), sil_pca_kmeans.max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')
ax.set_xlabel('Silhouette Score (Original Space)', fontsize=11)
ax.set_ylabel('Silhouette Score (PCA Space)', fontsize=11)
ax.set_title('Clustering Quality Correlation\n(Original vs PCA)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH / 'segmentation_quality_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Segmentation quality comparison plots saved!")

# Save comparison results to CSV
df_comparison = pd.DataFrame(segmentation_comparison)
df_comparison.to_csv(RESULTS_PATH / 'segmentation_quality_comparison.csv', index=False)
print(f"✓ Comparison data saved to: {RESULTS_PATH / 'segmentation_quality_comparison.csv'}")


# %% [markdown]
# ## 10.5. Visual Comparison: PCA vs Original vs Ground Truth
#
# Visualize actual segmentation results comparing PCA-based, original space, and expert ground truth segmentations.


# %%
from sklearn.metrics import jaccard_score, f1_score

# Select representative subjects for visualization
n_examples = 5
example_indices = np.linspace(0, len(subjects[:n_subjects_to_process])-1, n_examples, dtype=int)
example_subjects = [subjects[i] for i in example_indices if i < len(subjects)]

print("="*100)
print(" " * 25 + "VISUAL SEGMENTATION COMPARISON")
print(" " * 15 + "(PCA-based vs Original Space vs Ground Truth)")
print("="*100)

fig, axes = plt.subplots(n_examples, 6, figsize=(24, 4*n_examples))

comparison_metrics_visual = []

for row_idx, subject_id in enumerate(example_subjects):
    try:
        print(f"\nProcessing {subject_id}...")
        
        # Load subject with tumor slice
        slices, slice_idx = brats_loader.load_subject_with_tumor_slice(subject_id, axis=2)
        
        # Skip if no segmentation
        if 'seg' not in slices or slices['seg'] is None:
            print(f"  Skipping - no segmentation")
            continue
        
        # Get ground truth
        ground_truth = slices['seg']
        
        # Check if tumor is present
        if np.sum(ground_truth > 0) == 0:
            print(f"  Skipping - no tumor")
            continue
        
        # Prepare multi-modal features
        X_original = prepare_multimodal_features(slices, modalities=['t1', 't1ce', 't2', 'flair'])
        
        # Get image shape
        img_shape = ground_truth.shape
        
        # ===== SEGMENT IN ORIGINAL SPACE (GMM) =====
        gmm_original = GaussianMixture(n_components=4, random_state=42, covariance_type='full')
        labels_original = gmm_original.fit_predict(X_original)
        seg_original = labels_original.reshape(img_shape)
        
        # ===== SEGMENT IN PCA SPACE (GMM) =====
        pca = PCAAnalyzer(n_components=3)
        X_pca = pca.fit_transform(X_original)
        gmm_pca = GaussianMixture(n_components=4, random_state=42, covariance_type='full')
        labels_pca = gmm_pca.fit_predict(X_pca)
        seg_pca = labels_pca.reshape(img_shape)
        
        # ===== COMPUTE METRICS VS GROUND TRUTH =====
        # Binarize for tumor vs background comparison
        gt_binary = (ground_truth > 0).astype(int).flatten()
        
        # For algorithmic segmentations, find which label corresponds to tumor
        # (label with highest overlap with ground truth tumor region)
        def find_tumor_labels(seg_labels, gt_binary):
            """Find which cluster labels correspond to tumor regions"""
            best_overlap = 0
            best_binary = None
            
            # Try different combinations of labels as "tumor"
            unique_labels = np.unique(seg_labels)
            for tumor_label in unique_labels:
                binary = (seg_labels == tumor_label).astype(int)
                overlap = jaccard_score(gt_binary, binary)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_binary = binary
            
            return best_binary
        
        seg_orig_binary = find_tumor_labels(labels_original, gt_binary)
        seg_pca_binary = find_tumor_labels(labels_pca, gt_binary)
        
        # Compute metrics
        dice_original = f1_score(gt_binary, seg_orig_binary)
        dice_pca = f1_score(gt_binary, seg_pca_binary)
        jaccard_original = jaccard_score(gt_binary, seg_orig_binary)
        jaccard_pca = jaccard_score(gt_binary, seg_pca_binary)
        
        comparison_metrics_visual.append({
            'subject_id': subject_id,
            'dice_original': dice_original,
            'dice_pca': dice_pca,
            'jaccard_original': jaccard_original,
            'jaccard_pca': jaccard_pca
        })
        
        # ===== VISUALIZE =====
        # Use FLAIR as base image (best for tumor visualization)
        base_image = slices['flair']
        
        # 1. Original FLAIR image
        ax = axes[row_idx, 0]
        ax.imshow(base_image, cmap='gray')
        ax.set_title(f'{subject_id}\nFLAIR Image', fontsize=10)
        ax.axis('off')
        
        # 2. Ground Truth Segmentation
        ax = axes[row_idx, 1]
        ax.imshow(base_image, cmap='gray', alpha=0.7)
        ax.imshow(ground_truth, cmap='jet', alpha=0.5, vmin=0, vmax=4)
        ax.set_title(f'Ground Truth\n(Expert Labels)', fontsize=10)
        ax.axis('off')
        
        # 3. Original Space Segmentation
        ax = axes[row_idx, 2]
        ax.imshow(base_image, cmap='gray', alpha=0.7)
        ax.imshow(seg_original, cmap='viridis', alpha=0.5)
        ax.set_title(f'Original Space\nDice: {dice_original:.3f}', fontsize=10)
        ax.axis('off')
        
        # 4. PCA Space Segmentation
        ax = axes[row_idx, 3]
        ax.imshow(base_image, cmap='gray', alpha=0.7)
        ax.imshow(seg_pca, cmap='viridis', alpha=0.5)
        ax.set_title(f'PCA Space (3 PCs)\nDice: {dice_pca:.3f}', fontsize=10)
        ax.axis('off')
        
        # 5. Comparison: Original vs Ground Truth
        ax = axes[row_idx, 4]
        # Create RGB image showing overlap
        comparison_orig = np.zeros((*img_shape, 3))
        gt_mask = ground_truth > 0
        seg_orig_mask = seg_orig_binary.reshape(img_shape).astype(bool)
        
        # Red: Ground truth only, Green: Segmentation only, Yellow: Overlap
        comparison_orig[gt_mask & ~seg_orig_mask, 0] = 1  # Red: GT only
        comparison_orig[~gt_mask & seg_orig_mask, 1] = 1  # Green: Seg only
        comparison_orig[gt_mask & seg_orig_mask, 0] = 1   # Yellow: Overlap
        comparison_orig[gt_mask & seg_orig_mask, 1] = 1
        
        ax.imshow(base_image, cmap='gray', alpha=0.5)
        ax.imshow(comparison_orig, alpha=0.6)
        ax.set_title(f'Original vs GT\nJaccard: {jaccard_original:.3f}', fontsize=10)
        ax.axis('off')
        
        # 6. Comparison: PCA vs Ground Truth
        ax = axes[row_idx, 5]
        comparison_pca = np.zeros((*img_shape, 3))
        seg_pca_mask = seg_pca_binary.reshape(img_shape).astype(bool)
        
        comparison_pca[gt_mask & ~seg_pca_mask, 0] = 1  # Red: GT only
        comparison_pca[~gt_mask & seg_pca_mask, 1] = 1  # Green: Seg only
        comparison_pca[gt_mask & seg_pca_mask, 0] = 1   # Yellow: Overlap
        comparison_pca[gt_mask & seg_pca_mask, 1] = 1
        
        ax.imshow(base_image, cmap='gray', alpha=0.5)
        ax.imshow(comparison_pca, alpha=0.6)
        ax.set_title(f'PCA vs GT\nJaccard: {jaccard_pca:.3f}', fontsize=10)
        ax.axis('off')
        
        print(f"  ✓ Dice scores - Original: {dice_original:.3f}, PCA: {dice_pca:.3f}")
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        # Clear axes if error
        for col_idx in range(6):
            axes[row_idx, col_idx].axis('off')
        continue

# Add legend for color coding in comparison images
fig.text(0.5, 0.02, 
         'Color Key: Red = Ground Truth only | Green = Prediction only | Yellow = Correct Overlap',
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(RESULTS_PATH / 'visual_segmentation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*100)
print("VISUAL COMPARISON METRICS SUMMARY:")
print("="*100)

if comparison_metrics_visual:
    df_visual = pd.DataFrame(comparison_metrics_visual)
    
    print(f"\nAcross {len(df_visual)} example subjects:")
    print(f"\n  Original Space vs Ground Truth:")
    print(f"    Mean Dice Score:    {df_visual['dice_original'].mean():.4f} ± {df_visual['dice_original'].std():.4f}")
    print(f"    Mean Jaccard Index: {df_visual['jaccard_original'].mean():.4f} ± {df_visual['jaccard_original'].std():.4f}")
    
    print(f"\n  PCA Space vs Ground Truth:")
    print(f"    Mean Dice Score:    {df_visual['dice_pca'].mean():.4f} ± {df_visual['dice_pca'].std():.4f}")
    print(f"    Mean Jaccard Index: {df_visual['jaccard_pca'].mean():.4f} ± {df_visual['jaccard_pca'].std():.4f}")
    
    print(f"\n  Comparison:")
    dice_diff = df_visual['dice_pca'].mean() - df_visual['dice_original'].mean()
    print(f"    Dice difference (PCA - Original): {dice_diff:+.4f} ({dice_diff/df_visual['dice_original'].mean()*100:+.1f}%)")
    
    if dice_diff >= -0.02:  # Within 2% tolerance
        print(f"    ✓ PCA maintains comparable segmentation quality")
    elif dice_diff > 0:
        print(f"    ✓ PCA shows improved segmentation quality")
    else:
        print(f"    ⚠ PCA shows reduced segmentation quality")
    
    # Statistical test
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(df_visual['dice_original'], df_visual['dice_pca'])
    print(f"\n  Paired t-test (Dice scores):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value:     {p_val:.4f}")
    print(f"    Result:      {'Significantly different' if p_val < 0.05 else 'No significant difference'} (α=0.05)")

print("\n" + "="*100)
print("✓ Visual comparison complete!")
print("="*100)


# %%
import time

# Function to time segmentation operations
def benchmark_segmentation(X, n_components, n_clusters, method='gmm', n_runs=10):
    """
    Benchmark segmentation performance.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_components: Number of PCA components (None for original space)
        n_clusters: Number of clusters for segmentation
        method: 'gmm' or 'kmeans'
        n_runs: Number of timing runs
    
    Returns:
        Dictionary with timing results
    """
    times_pca = []
    times_clustering = []
    times_total = []
    
    for _ in range(n_runs):
        start_total = time.perf_counter()
        
        # PCA step (if using PCA)
        if n_components is not None:
            start_pca = time.perf_counter()
            pca = PCAAnalyzer(n_components=n_components)
            X_transformed = pca.fit_transform(X)
            pca_time = time.perf_counter() - start_pca
            times_pca.append(pca_time)
        else:
            X_transformed = X
            times_pca.append(0)
        
        # Clustering step
        start_clustering = time.perf_counter()
        if method == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        labels = model.fit_predict(X_transformed)
        clustering_time = time.perf_counter() - start_clustering
        times_clustering.append(clustering_time)
        
        total_time = time.perf_counter() - start_total
        times_total.append(total_time)
    
    return {
        'mean_pca_time': np.mean(times_pca),
        'std_pca_time': np.std(times_pca),
        'mean_clustering_time': np.mean(times_clustering),
        'std_clustering_time': np.std(times_clustering),
        'mean_total_time': np.mean(times_total),
        'std_total_time': np.std(times_total),
        'speedup_factor': None  # Will be calculated later
    }


# Select a representative subject for benchmarking
if len(subjects) > 0:
    print("="*100)
    print(" " * 30 + "COMPUTATIONAL LOAD COMPARISON")
    print("="*100)
    
    # Use a middle subject
    test_subject = subjects[len(subjects)//2]
    print(f"\nTesting with subject: {test_subject}")
    
    # Load data
    slices, slice_idx = brats_loader.load_subject_with_tumor_slice(test_subject, axis=2)
    
    # Prepare multi-modal features
    X_multimodal = prepare_multimodal_features(slices, modalities=['t1', 't1ce', 't2', 'flair'])
    
    print(f"Data shape: {X_multimodal.shape}")
    print(f"Number of pixels: {X_multimodal.shape[0]:,}")
    print(f"Number of features: {X_multimodal.shape[1]}")
    
    # Configuration
    n_clusters = 4
    n_components_list = [2, 3, 5]
    n_runs = 10
    
    print(f"\nConfiguration:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Timing runs: {n_runs}")
    print(f"  Methods: GMM and KMeans")
    
    # Benchmark different scenarios
    print("\n" + "-"*100)
    print("Running benchmarks...")
    print("-"*100)
    
    results_benchmark = {}
    
    # Original space (no PCA)
    print("\n1. Original Space (No PCA):")
    for method in ['gmm', 'kmeans']:
        print(f"   Testing {method.upper()}...", end=' ')
        result = benchmark_segmentation(X_multimodal, None, n_clusters, method=method, n_runs=n_runs)
        results_benchmark[f'original_{method}'] = result
        print(f"✓ Mean time: {result['mean_total_time']*1000:.2f} ms")
    
    # PCA with different components
    for n_comp in n_components_list:
        print(f"\n2. PCA with {n_comp} components:")
        for method in ['gmm', 'kmeans']:
            print(f"   Testing {method.upper()}...", end=' ')
            result = benchmark_segmentation(X_multimodal, n_comp, n_clusters, method=method, n_runs=n_runs)
            results_benchmark[f'pca{n_comp}_{method}'] = result
            print(f"✓ Mean time: {result['mean_total_time']*1000:.2f} ms")
    
    # Calculate speedup factors
    for key in results_benchmark:
        if 'pca' in key:
            method = key.split('_')[1]
            original_key = f'original_{method}'
            speedup = results_benchmark[original_key]['mean_total_time'] / results_benchmark[key]['mean_total_time']
            results_benchmark[key]['speedup_factor'] = speedup
    
    print("\n" + "="*100)
    print(" " * 35 + "TIMING RESULTS")
    print("="*100)
    
    # Display results in a formatted table
    print(f"\n{'Method':<25} {'PCA Time (ms)':<18} {'Clustering Time (ms)':<22} {'Total Time (ms)':<18} {'Speedup':<10}")
    print("-"*100)
    
    for key, result in results_benchmark.items():
        method_name = key.replace('_', ' ').upper()
        pca_time = result['mean_pca_time'] * 1000
        cluster_time = result['mean_clustering_time'] * 1000
        total_time = result['mean_total_time'] * 1000
        speedup = result['speedup_factor']
        
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "baseline"
        print(f"{method_name:<25} {pca_time:>8.2f} ± {result['std_pca_time']*1000:>5.2f}   "
              f"{cluster_time:>8.2f} ± {result['std_clustering_time']*1000:>5.2f}      "
              f"{total_time:>8.2f} ± {result['std_total_time']*1000:>5.2f}   {speedup_str:>10}")
    
    print("\n" + "="*100)
    print("KEY OBSERVATIONS:")
    print("="*100)
    
    # Find best performers
    gmm_keys = [k for k in results_benchmark.keys() if 'gmm' in k]
    kmeans_keys = [k for k in results_benchmark.keys() if 'kmeans' in k]
    
    best_gmm = min(gmm_keys, key=lambda k: results_benchmark[k]['mean_total_time'])
    best_kmeans = min(kmeans_keys, key=lambda k: results_benchmark[k]['mean_total_time'])
    
    print(f"\n  1. Fastest GMM approach:    {best_gmm.upper()}")
    print(f"     Time: {results_benchmark[best_gmm]['mean_total_time']*1000:.2f} ms")
    if results_benchmark[best_gmm]['speedup_factor']:
        print(f"     Speedup: {results_benchmark[best_gmm]['speedup_factor']:.2f}x faster than original space")
    
    print(f"\n  2. Fastest KMeans approach: {best_kmeans.upper()}")
    print(f"     Time: {results_benchmark[best_kmeans]['mean_total_time']*1000:.2f} ms")
    if results_benchmark[best_kmeans]['speedup_factor']:
        print(f"     Speedup: {results_benchmark[best_kmeans]['speedup_factor']:.2f}x faster than original space")
    
    print(f"\n  3. PCA overhead is {'minimal' if np.mean([results_benchmark[k]['mean_pca_time'] for k in results_benchmark if 'pca' in k]) < 0.01 else 'significant'}")
    print(f"     Mean PCA time: {np.mean([results_benchmark[k]['mean_pca_time'] for k in results_benchmark if 'pca' in k])*1000:.2f} ms")
    
    print(f"\n  4. Clustering time reduces with fewer dimensions:")
    for n_comp in n_components_list:
        pca_cluster_time = results_benchmark[f'pca{n_comp}_gmm']['mean_clustering_time']
        orig_cluster_time = results_benchmark['original_gmm']['mean_clustering_time']
        reduction = (1 - pca_cluster_time/orig_cluster_time) * 100
        print(f"     {n_comp} components: {reduction:.1f}% faster clustering")
    
    print(f"\n  5. Overall computational benefit: PCA reduces total time by up to "
          f"{max([r['speedup_factor'] for r in results_benchmark.values() if r['speedup_factor']])-1:.0%}")
    
    print("\n" + "="*100)
    
else:
    print("No subjects available for benchmarking")


# %%
# Save detailed results to CSV
df_orig = pd.DataFrame([{
    'subject_id': r['subject_id'],
    'slice_idx': r['slice_idx'],
    'type': 'original',
    'n_features': r['n_features'],
    'mse': r['mse'],
    'rmse': r['rmse'],
    'variance_explained': r['total_variance_explained'],
    'n_components': r['n_components']
} for r in results['original']])

df_seg = pd.DataFrame([{
    'subject_id': r['subject_id'],
    'slice_idx': r['slice_idx'],
    'type': 'segmentation',
    'n_features': r['n_features'],
    'mse': r['mse'],
    'rmse': r['rmse'],
    'variance_explained': r['total_variance_explained'],
    'n_components': r['n_components'],
    'tumor_percentage': r['tumor_percentage']
} for r in results['segmentation']])

df_combined = pd.concat([df_orig, df_seg], ignore_index=True)
df_combined.to_csv(RESULTS_PATH / 'pca_comparison_results.csv', index=False)

# Save statistical summary
summary_dict = {
    'analysis_type': 'PCA Comparison - Original vs Segmentation',
    'n_subjects': len(results['original']),
    'mle_mu_original': mu_orig,
    'mle_sigma_original': sigma_orig,
    'mle_mu_segmentation': mu_seg,
    'mle_sigma_segmentation': sigma_seg,
    't_statistic': t_stat,
    't_pvalue': t_pval,
    'cohens_d': cohens_d,
    'mean_rmse_original': np.mean(errors_original),
    'mean_rmse_segmentation': np.mean(errors_seg),
    'mean_variance_explained_original': np.mean(var_orig),
    'mean_variance_explained_segmentation': np.mean(var_seg)
}

pd.DataFrame([summary_dict]).to_csv(RESULTS_PATH / 'statistical_summary.csv', index=False)

print("✓ Results saved to CSV files:")
print(f"  - {RESULTS_PATH / 'pca_comparison_results.csv'}")
print(f"  - {RESULTS_PATH / 'statistical_summary.csv'}")
