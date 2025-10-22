# EM Algorithm Implementation for Old Faithful Geyser Data

## CS5785 Homework 3 - Problem 2

This repository contains a complete implementation of the EM algorithm for fitting a bimodal Gaussian Mixture Model to the Old Faithful geyser dataset.

## 📁 Project Structure

```
AML_HW3/
├── 📄 AML_HW_3_2025_FINAL.pdf          # Assignment PDF
├── 🐍 em_algorithm_complete.py         # Complete EM algorithm implementation
├── 📓 EM_Algorithm_Implementation.ipynb # Main Jupyter notebook
├── 📄 EM_Algorithm_Implementation.pdf   # Notebook export
├── 📁 HW3-2/                           # Additional notebook version
│   └── 📓 EM_Algorithm_Implementation.ipynb
├── 📁 images/                          # Generated visualizations
│   ├── 🖼️ em_results.png              # EM clustering results
│   ├── 🖼️ em_vs_kmeans.png           # EM vs K-means comparison
│   └── 🖼️ faithful_data.png          # Original data visualization
├── 🐍 test_em.py                      # Test script
└── 📄 README.md                       # This file
```

## 🚀 Quick Start

### Running the Complete Implementation
```bash
python3 em_algorithm_complete.py
```

### Using the Jupyter Notebook
```bash
jupyter notebook EM_Algorithm_Implementation.ipynb
```

### Running Tests
```bash
python3 test_em.py
```

## 📋 Problem Requirements (40 pts total)

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

## ✨ Key Features

### 🔬 EM Algorithm Implementation
- **🎯 Soft Clustering:** Provides probabilistic cluster assignments
- **📐 Diagonal Covariance:** Assumes Σ_k = diag(σ²_1, σ²_2, ..., σ²_d)
- **🔄 Convergence Detection:** Log-likelihood-based termination
- **📈 Trajectory Tracking:** Records mean vector evolution

### 📊 Visualization Features
- **📈 Original Data Plot:** Scatter plot of Old Faithful geyser data
- **🎨 EM Clustering Results:** Probabilistic cluster assignments
- **📉 Log-likelihood Convergence:** Iteration vs likelihood plot
- **🔄 Mean Vector Trajectories:** Evolution of cluster centers
- **🎯 Cluster Assignment Probabilities:** Soft clustering visualization
- **⚖️ EM vs K-means Comparison:** Side-by-side clustering comparison

## 📊 Results

The implementation successfully:

### 📈 Dataset Processing
- **📊 Data Points:** Loads 272 observations from Old Faithful geyser
- **⚡ Convergence:** Converges in 8 iterations with tolerance 1e-6
- **🎯 Clusters Identified:** Two distinct eruption patterns

### 🔍 Cluster Analysis
- **🌋 Cluster 1:** Short eruptions (~2.04 min) with short waiting times (~54.5 min)
- **🌋 Cluster 2:** Long eruptions (~4.29 min) with long waiting times (~80.0 min)
- **📊 Probabilistic Assignments:** Soft clustering provides probability distributions

### 📸 Generated Visualizations
All plots are saved in the `images/` folder:
- `faithful_data.png` - Original dataset visualization
- `em_results.png` - EM clustering results with probabilistic assignments
- `em_vs_kmeans.png` - Comparison between EM and K-means algorithms

## 🔍 Analysis

### ⚖️ EM vs K-means Comparison

| Aspect | EM Algorithm | K-means |
|--------|-------------|---------|
| **🎯 Assignment Type** | Soft (probabilistic) | Hard (deterministic) |
| **📐 Cluster Shape** | Elliptical (diagonal covariance) | Spherical |
| **📊 Information** | Detailed probability distributions | Binary assignments |
| **🔄 Flexibility** | Models complex cluster shapes | Assumes spherical clusters |

### 🎯 Key Findings
1. **🎲 Soft vs Hard Clustering:** EM provides probabilistic assignments, K-means provides hard assignments
2. **📐 Cluster Shapes:** EM can model elliptical clusters, K-means assumes spherical clusters  
3. **📊 Results:** Both methods give similar results for this dataset due to well-separated, roughly spherical clusters
4. **🔧 Flexibility:** EM provides more detailed probabilistic information and can model complex cluster shapes

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| **numpy** | Numerical computations and array operations |
| **matplotlib** | Data visualization and plotting |
| **scipy** | Scientific computing utilities |
| **scikit-learn** | Machine learning algorithms (K-means comparison) |

## 📦 Installation

```bash
pip install numpy matplotlib scipy scikit-learn
```

## 🎯 Usage Examples

### 📊 Running the Complete Implementation
```bash
python3 em_algorithm_complete.py
```
*Generates all visualizations and saves them to the `images/` folder*

### 📓 Interactive Jupyter Notebook
```bash
jupyter notebook EM_Algorithm_Implementation.ipynb
```
*Provides step-by-step analysis with interactive plots*

### 🧪 Running Tests
```bash
python3 test_em.py
```
*Validates the EM algorithm implementation*

## 📊 Data Source

The Old Faithful geyser data is sourced from:
**🔗 [CMU Statistics Data Archive](https://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat)**

### 📈 Dataset Characteristics
- **📊 Size:** 272 observations
- **📏 Features:** 2D feature vectors
  - **⏱️ Eruption Duration:** Time in minutes
  - **⏳ Waiting Time:** Time to next eruption in minutes
- **🎯 Purpose:** Demonstrate bimodal clustering patterns

## 📚 References

- **Hardle, W.** (1991) Smoothing Techniques with Implementation in S. New York: Springer.
- **Azzalini, A. and Bowman, A. W.** (1990). A look at some data on the Old Faithful geyser. Applied Statistics 39, 357-365.

---

## 📝 License

This project is part of CS5785 Applied Machine Learning coursework at Cornell University.

## 👨‍💻 Author

**Student:** sosadennis39-debug  
**Course:** CS5785 Applied Machine Learning  
**Assignment:** Homework 3 - Problem 2
