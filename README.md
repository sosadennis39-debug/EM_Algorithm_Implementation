# EM Algorithm Implementation for Old Faithful Geyser Data

## CS5785 Homework 3 - Problem 2

This repository contains a complete implementation of the EM algorithm for fitting a bimodal Gaussian Mixture Model to the Old Faithful geyser dataset.

## Files

- `em_algorithm.py` - Basic implementation
- `em_algorithm_complete.py` - Complete implementation with all features
- `EM_Algorithm_Implementation.ipynb` - Jupyter notebook with detailed explanations
- `README.md` - This file

## Problem Requirements (40 pts total)

### 1. Data Loading and Visualization (2 pts)
- Parse Old Faithful geyser data as 2D feature vectors
- Plot the data points on a 2D plane

### 2. E-step Formula (3 pts)
**Formula:** P(z=k|x) = P(z=k,x) / P(x) = P(x|z=k) * P(z=k) / Σ_l P(x|z=l) * P(z=l)

This computes the posterior probability that data point x belongs to cluster k.

### 3. M-step Formulas (5 pts)
**Mean:** μ_k = Σ_i r_ik * x_i / Σ_i r_ik
**Covariance:** Σ_k = Σ_i r_ik * (x_i - μ_k)(x_i - μ_k)^T / Σ_i r_ik
**Weights:** φ_k = Σ_i r_ik / n

Where r_ik = P(z_i = k | x_i) is the responsibility of cluster k for data point i.

### 4. EM Algorithm Implementation (25 pts)

#### 4.1 Implementation from Scratch (10 pts)
- Complete EM algorithm implementation using numpy
- Diagonal covariance assumption as suggested in the problem
- Random parameter initialization

#### 4.2 Termination Criterion (5 pts)
**Chosen criterion:** Log-likelihood convergence with tolerance 1e-6
- Algorithm stops when |L(t) - L(t-1)| < tolerance
- Ensures convergence to a local optimum
- Standard and robust convergence criterion

#### 4.3 Mean Vector Trajectories (10 pts)
- Plot trajectories of μ₁ and μ₂ during EM iterations
- Visualize how cluster centers evolve during optimization
- Show convergence behavior

### 5. Comparison with K-means (5 pts)
- Compare EM results with K-means clustering
- Analyze differences between soft and hard clustering
- Discuss cluster shape assumptions

## Key Features

### EM Algorithm Implementation
- **Soft Clustering:** Provides probabilistic cluster assignments
- **Diagonal Covariance:** Assumes Σ_k = diag(σ²_1, σ²_2, ..., σ²_d)
- **Convergence Detection:** Log-likelihood-based termination
- **Trajectory Tracking:** Records mean vector evolution

### Visualization
- Original data scatter plot
- EM clustering results
- Log-likelihood convergence plot
- Mean vector trajectories
- Cluster assignment probabilities
- K-means comparison

## Results

The implementation successfully:
- Loads 272 data points from Old Faithful geyser
- Converges in 8 iterations with tolerance 1e-6
- Identifies two distinct clusters:
  - Cluster 1: Short eruptions (~2.04 min) with short waiting times (~54.5 min)
  - Cluster 2: Long eruptions (~4.29 min) with long waiting times (~80.0 min)
- Provides probabilistic assignments for each data point

## Analysis

**EM vs K-means:**
1. **Soft vs Hard Clustering:** EM provides probabilistic assignments, K-means provides hard assignments
2. **Cluster Shapes:** EM can model elliptical clusters, K-means assumes spherical clusters
3. **Results:** Both methods give similar results for this dataset due to well-separated, roughly spherical clusters
4. **Flexibility:** EM provides more detailed probabilistic information and can model complex cluster shapes

## Usage

### Running the Complete Implementation
```bash
python3 em_algorithm_complete.py
```

### Using the Jupyter Notebook
```bash
jupyter notebook EM_Algorithm_Implementation.ipynb
```

## Dependencies

- numpy
- matplotlib
- scipy
- scikit-learn

## Installation

```bash
pip install numpy matplotlib scipy scikit-learn
```

## Data Source

The Old Faithful geyser data is sourced from:
https://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat

The dataset contains 272 observations of:
- Eruption duration (minutes)
- Waiting time to next eruption (minutes)

## References

- Hardle, W. (1991) Smoothing Techniques with Implementation in S. New York: Springer.
- Azzalini, A. and Bowman, A. W. (1990). A look at some data on the Old Faithful geyser. Applied Statistics 39, 357-365.
