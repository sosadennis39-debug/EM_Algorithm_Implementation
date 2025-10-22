"""
EM Algorithm Implementation for Old Faithful Geyser Data
CS5785 Homework 3 - Problem 2

This script implements a bimodal Gaussian Mixture Model using the EM algorithm
for the Old Faithful geyser dataset.

Author: AI Assistant
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class GaussianMixtureEM:
    """
    Gaussian Mixture Model with EM algorithm implementation
    
    This class implements the EM algorithm for fitting a Gaussian Mixture Model
    with diagonal covariance matrices (as suggested in the problem statement).
    """
    
    def __init__(self, n_components=2, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.converged_ = False
        self.n_iter_ = 0
        
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        n_samples, n_features = X.shape
        
        # Random initialization of means
        self.means_ = np.random.uniform(
            low=X.min(axis=0), 
            high=X.max(axis=0), 
            size=(self.n_components, n_features)
        )
        
        # Initialize covariances as diagonal matrices
        self.covariances_ = np.array([
            np.eye(n_features) * np.var(X, axis=0) 
            for _ in range(self.n_components)
        ])
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
    def _e_step(self, X):
        """
        E-step: Compute posterior probabilities P(z=k|x)
        
        Formula: P(z=k|x) = P(z=k,x) / P(x) = P(x|z=k) * P(z=k) / Σ_l P(x|z=l) * P(z=l)
        
        This is the posterior probability that data point x belongs to cluster k,
        given the current parameter estimates.
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Compute P(x|z=k) using multivariate normal
            responsibilities[:, k] = multivariate_normal.pdf(
                X, 
                mean=self.means_[k], 
                cov=self.covariances_[k]
            ) * self.weights_[k]
        
        # Normalize to get posterior probabilities
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities = responsibilities / responsibilities_sum
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        M-step: Update parameters to maximize the expected log-likelihood
        
        Formulas:
        μ_k = Σ_i r_ik * x_i / Σ_i r_ik
        Σ_k = Σ_i r_ik * (x_i - μ_k)(x_i - μ_k)^T / Σ_i r_ik
        φ_k = Σ_i r_ik / n
        
        Where r_ik = P(z_i = k | x_i) is the responsibility of cluster k for data point i.
        """
        n_samples, n_features = X.shape
        
        for k in range(self.n_components):
            # Compute effective number of points assigned to cluster k
            n_k = responsibilities[:, k].sum()
            
            if n_k > 0:
                # Update mean: μ_k = Σ_i r_ik * x_i / Σ_i r_ik
                self.means_[k] = np.sum(
                    responsibilities[:, k].reshape(-1, 1) * X, 
                    axis=0
                ) / n_k
                
                # Update covariance (diagonal assumption)
                # Σ_k = Σ_i r_ik * (x_i - μ_k)(x_i - μ_k)^T / Σ_i r_ik
                diff = X - self.means_[k]
                self.covariances_[k] = np.diag(
                    np.sum(
                        responsibilities[:, k].reshape(-1, 1) * (diff ** 2), 
                        axis=0
                    ) / n_k
                )
                
                # Update weight: φ_k = Σ_i r_ik / n
                self.weights_[k] = n_k / n_samples
            else:
                # Handle empty clusters by reinitializing
                self.means_[k] = np.random.uniform(
                    low=X.min(axis=0), 
                    high=X.max(axis=0)
                )
                self.covariances_[k] = np.eye(n_features) * np.var(X, axis=0)
                self.weights_[k] = 1.0 / self.n_components
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood of the data"""
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights_[k] * multivariate_normal.pdf(
                X, 
                mean=self.means_[k], 
                cov=self.covariances_[k]
            )
        return np.sum(np.log(log_likelihood + 1e-10))
    
    def fit(self, X):
        """Fit the Gaussian Mixture Model using EM algorithm"""
        self._initialize_parameters(X)
        
        log_likelihoods = []
        mean_trajectories = []
        
        for iteration in range(self.max_iter):
            # Store current means for trajectory plotting
            mean_trajectories.append(self.means_.copy())
            
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if iteration > 0:
                if abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                    self.converged_ = True
                    self.n_iter_ = iteration + 1
                    break
        
        if not self.converged_:
            self.n_iter_ = self.max_iter
            
        return log_likelihoods, mean_trajectories
    
    def predict(self, X):
        """Predict cluster assignments"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

def load_faithful_data():
    """Load Old Faithful geyser data from URL"""
    # Based on the web search results, the data format is:
    # eruptions waiting
    # 3.600 79
    # 1.800 54
    # etc.
    
    # Let's create the data directly from the web search results
    data = [
        [3.600, 79], [1.800, 54], [3.333, 74], [2.283, 62], [4.533, 85],
        [2.883, 55], [4.700, 88], [3.600, 85], [1.950, 51], [4.350, 85],
        [1.833, 54], [3.917, 84], [4.200, 78], [1.750, 47], [4.700, 83],
        [2.167, 52], [1.750, 62], [4.800, 84], [1.600, 52], [4.250, 79],
        [1.800, 51], [1.750, 47], [3.450, 78], [3.067, 69], [4.533, 74],
        [3.600, 83], [1.967, 55], [4.083, 76], [3.850, 78], [4.433, 79],
        [4.300, 73], [4.467, 77], [3.367, 66], [4.033, 80], [3.833, 74],
        [2.017, 52], [1.867, 48], [4.833, 80], [1.833, 59], [4.783, 90],
        [4.350, 80], [1.883, 58], [4.567, 84], [1.750, 58], [4.533, 73],
        [3.317, 83], [3.833, 64], [2.100, 53], [4.633, 82], [2.000, 59],
        [4.800, 75], [4.716, 90], [1.833, 54], [4.833, 80], [1.733, 54],
        [4.883, 83], [3.717, 71], [1.667, 64], [4.567, 77], [4.317, 81],
        [2.233, 59], [4.500, 84], [1.750, 48], [4.800, 82], [1.817, 60],
        [4.400, 92], [4.167, 78], [4.700, 78], [2.067, 65], [4.700, 73],
        [4.033, 82], [1.967, 56], [4.500, 79], [4.000, 71], [1.983, 62],
        [5.067, 76], [2.017, 60], [4.567, 78], [3.883, 76], [3.600, 83],
        [4.133, 75], [4.333, 82], [4.100, 70], [2.633, 65], [4.067, 73],
        [4.933, 88], [3.950, 76], [4.517, 80], [2.167, 48], [4.000, 86],
        [2.200, 60], [4.333, 90], [1.867, 50], [4.817, 78], [1.833, 63],
        [4.300, 72], [4.667, 84], [3.750, 75], [1.867, 51], [4.900, 82],
        [2.483, 62], [4.367, 88], [2.100, 49], [4.500, 83], [4.050, 81],
        [1.867, 47], [4.700, 84], [1.783, 52], [4.850, 86], [3.683, 81],
        [4.733, 75], [2.300, 59], [4.900, 89], [4.417, 79], [1.700, 59],
        [4.633, 81], [2.317, 50], [4.600, 85], [1.817, 59], [4.417, 87],
        [2.617, 53], [4.067, 69], [4.250, 77], [1.967, 56], [4.600, 88],
        [3.767, 81], [1.917, 45], [4.500, 82], [2.267, 55], [4.650, 90],
        [1.867, 45], [4.167, 83], [2.800, 56], [4.333, 89], [1.833, 46],
        [4.383, 82], [1.883, 51], [4.933, 86], [2.033, 53], [3.733, 79],
        [4.233, 81], [2.233, 60], [4.533, 82], [4.817, 77], [4.333, 76],
        [1.983, 59], [4.633, 80], [2.017, 49], [5.100, 96], [1.800, 53],
        [5.033, 77], [4.000, 77], [2.400, 65], [4.600, 81], [3.567, 71],
        [4.000, 70], [4.500, 81], [4.083, 93], [1.800, 53], [3.967, 89],
        [2.200, 45], [4.150, 86], [2.000, 58], [3.833, 78], [3.500, 66],
        [4.583, 76], [2.367, 63], [5.000, 88], [1.933, 52], [4.617, 93],
        [1.917, 49], [2.083, 57], [4.583, 77], [3.333, 68], [4.167, 81],
        [4.333, 81], [4.500, 73], [2.417, 50], [4.000, 85], [4.167, 74],
        [1.883, 55], [4.583, 77], [4.250, 83], [3.767, 83], [2.033, 51],
        [4.433, 78], [4.083, 84], [1.833, 46], [4.417, 83], [2.183, 55],
        [4.800, 81], [1.833, 57], [4.800, 76], [4.100, 84], [3.966, 77],
        [4.233, 81], [3.500, 87], [4.366, 77], [2.250, 51], [4.667, 78],
        [2.100, 60], [4.350, 82], [4.133, 91], [1.867, 53], [4.600, 78],
        [1.783, 46], [4.367, 77], [3.850, 84], [1.933, 49], [4.500, 83],
        [2.383, 71], [4.700, 80], [1.867, 49], [3.833, 75], [3.417, 64],
        [4.233, 76], [2.400, 53], [4.800, 94], [2.000, 55], [4.150, 76],
        [1.867, 50], [4.267, 82], [1.750, 54], [4.483, 75], [4.000, 78],
        [4.117, 79], [4.083, 78], [4.267, 78], [3.917, 70], [4.550, 79],
        [4.083, 70], [2.417, 54], [4.183, 86], [2.217, 50], [4.450, 90],
        [1.883, 54], [1.850, 54], [4.283, 77], [3.950, 79], [2.333, 64],
        [4.150, 75], [2.350, 47], [4.933, 86], [2.900, 63], [4.583, 85],
        [3.833, 82], [2.083, 57], [4.367, 82], [2.133, 67], [4.350, 74],
        [2.200, 54], [4.450, 83], [3.567, 73], [4.500, 73], [4.150, 88],
        [3.817, 80], [3.917, 71], [4.450, 83], [2.000, 56], [4.283, 79],
        [4.767, 78], [4.533, 84], [1.850, 58], [4.250, 83], [1.983, 43],
        [2.250, 60], [4.750, 75], [4.117, 81], [2.150, 46], [4.417, 90],
        [1.817, 46], [4.467, 74]
    ]
    
    return np.array(data)

def plot_data(X, title="Old Faithful Geyser Data"):
    """Plot the 2D data points"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=30)
    plt.xlabel('Eruption Duration (minutes)')
    plt.ylabel('Waiting Time (minutes)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_em_results(X, log_likelihoods, mean_trajectories, responsibilities):
    """Plot EM algorithm results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Data points colored by cluster assignment
    ax1 = axes[0, 0]
    cluster_assignments = np.argmax(responsibilities, axis=1)
    colors = ['red', 'blue']
    for k in range(2):
        mask = cluster_assignments == k
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, label=f'Cluster {k+1}')
    ax1.set_xlabel('Eruption Duration (minutes)')
    ax1.set_ylabel('Waiting Time (minutes)')
    ax1.set_title('EM Clustering Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood convergence
    ax2 = axes[0, 1]
    ax2.plot(log_likelihoods)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('EM Convergence')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean trajectories
    ax3 = axes[1, 0]
    mean_trajectories = np.array(mean_trajectories)
    ax3.plot(mean_trajectories[:, 0, 0], mean_trajectories[:, 0, 1], 'r-', marker='o', markersize=4, label='μ₁ trajectory')
    ax3.plot(mean_trajectories[:, 1, 0], mean_trajectories[:, 1, 1], 'b-', marker='s', markersize=4, label='μ₂ trajectory')
    ax3.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10, c='gray', label='Data points')
    ax3.set_xlabel('Eruption Duration (minutes)')
    ax3.set_ylabel('Waiting Time (minutes)')
    ax3.set_title('Mean Vector Trajectories')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cluster probabilities
    ax4 = axes[1, 1]
    scatter = ax4.scatter(X[:, 0], X[:, 1], c=responsibilities[:, 0], cmap='RdBu', alpha=0.7, s=30)
    ax4.set_xlabel('Eruption Duration (minutes)')
    ax4.set_ylabel('Waiting Time (minutes)')
    ax4.set_title('Cluster Assignment Probabilities')
    plt.colorbar(scatter, ax=ax4, label='P(Cluster 1)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def compare_with_kmeans(X):
    """Compare EM results with K-means clustering"""
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    plt.figure(figsize=(12, 5))
    
    # Plot K-means results
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue']
    for k in range(2):
        mask = kmeans_labels == k
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, label=f'Cluster {k+1}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='black', marker='x', s=200, linewidths=3, label='K-means centers')
    plt.xlabel('Eruption Duration (minutes)')
    plt.ylabel('Waiting Time (minutes)')
    plt.title('K-means Clustering Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot EM results for comparison
    plt.subplot(1, 2, 2)
    em_model = GaussianMixtureEM()
    log_likelihoods, mean_trajectories = em_model.fit(X)
    responsibilities = em_model._e_step(X)
    cluster_assignments = np.argmax(responsibilities, axis=1)
    
    for k in range(2):
        mask = cluster_assignments == k
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, label=f'Cluster {k+1}')
    plt.scatter(em_model.means_[:, 0], em_model.means_[:, 1], 
                c='black', marker='x', s=200, linewidths=3, label='EM centers')
    plt.xlabel('Eruption Duration (minutes)')
    plt.ylabel('Waiting Time (minutes)')
    plt.title('EM Clustering Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()

def print_theoretical_formulas():
    """Print the theoretical formulas for EM algorithm"""
    print("EM Algorithm Theoretical Formulas")
    print("=" * 50)
    print()
    print("E-step Formula:")
    print("P(z=k|x) = P(z=k,x) / P(x) = P(x|z=k) * P(z=k) / Σ_l P(x|z=l) * P(z=l)")
    print()
    print("M-step Formulas:")
    print("μ_k = Σ_i r_ik * x_i / Σ_i r_ik")
    print("Σ_k = Σ_i r_ik * (x_i - μ_k)(x_i - μ_k)^T / Σ_i r_ik")
    print("φ_k = Σ_i r_ik / n")
    print()
    print("Where:")
    print("- r_ik = P(z_i = k | x_i) is the responsibility of cluster k for data point i")
    print("- n is the total number of data points")
    print("- We assume diagonal covariance matrices: Σ_k = diag(σ²_1, σ²_2, ..., σ²_d)")
    print()

def main():
    """Main function to run the EM algorithm implementation"""
    print("EM Algorithm Implementation for Old Faithful Geyser Data")
    print("=" * 60)
    
    # Print theoretical formulas
    print_theoretical_formulas()
    
    # Load data
    print("Loading Old Faithful geyser data...")
    X = load_faithful_data()
    print(f"Loaded {X.shape[0]} data points with {X.shape[1]} features")
    print(f"Data shape: {X.shape}")
    print(f"Eruption duration range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}] minutes")
    print(f"Waiting time range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}] minutes")
    
    # Plot original data
    print("\nPlotting original data...")
    plot_data(X)
    plt.savefig('/Users/mkg/AML_HW3/faithful_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Run EM algorithm
    print("\nRunning EM algorithm...")
    em_model = GaussianMixtureEM(n_components=2, max_iter=100, tol=1e-6)
    log_likelihoods, mean_trajectories = em_model.fit(X)
    
    print(f"EM algorithm converged: {em_model.converged_}")
    print(f"Number of iterations: {em_model.n_iter_}")
    print(f"Final log-likelihood: {log_likelihoods[-1]:.4f}")
    
    # Get final responsibilities
    responsibilities = em_model._e_step(X)
    
    # Print final parameters
    print("\nFinal EM parameters:")
    for k in range(2):
        print(f"Cluster {k+1}:")
        print(f"  Mean: {em_model.means_[k]}")
        print(f"  Covariance diagonal: {np.diag(em_model.covariances_[k])}")
        print(f"  Weight: {em_model.weights_[k]:.4f}")
    
    # Plot results
    print("\nPlotting EM results...")
    plot_em_results(X, log_likelihoods, mean_trajectories, responsibilities)
    plt.savefig('/Users/mkg/AML_HW3/em_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compare with K-means
    print("\nComparing with K-means clustering...")
    compare_with_kmeans(X)
    plt.savefig('/Users/mkg/AML_HW3/em_vs_kmeans.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\nAnalysis:")
    print("-" * 40)
    print("EM Algorithm vs K-means:")
    print("1. EM provides probabilistic assignments (soft clustering)")
    print("2. K-means provides hard assignments")
    print("3. EM can model elliptical clusters with different covariances")
    print("4. K-means assumes spherical clusters with equal variance")
    print("5. For this dataset, both methods should give similar results")
    print("   since the clusters appear to be roughly spherical and well-separated")
    print()
    print("Termination Criterion:")
    print("- Used log-likelihood convergence with tolerance 1e-6")
    print("- Algorithm stops when |L(t) - L(t-1)| < tolerance")
    print("- This ensures the algorithm has converged to a local optimum")
    
    return em_model, X, log_likelihoods, mean_trajectories

if __name__ == "__main__":
    em_model, X, log_likelihoods, mean_trajectories = main()
