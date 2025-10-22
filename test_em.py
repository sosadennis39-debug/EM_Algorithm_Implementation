#!/usr/bin/env python3
"""
Test script for EM Algorithm Implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from em_algorithm_complete import GaussianMixtureEM, load_faithful_data
import numpy as np

def test_em_algorithm():
    """Test the EM algorithm implementation"""
    print("Testing EM Algorithm Implementation...")
    
    # Load data
    X = load_faithful_data()
    print(f"✓ Data loaded successfully: {X.shape}")
    
    # Test EM algorithm
    em_model = GaussianMixtureEM(n_components=2, max_iter=100, tol=1e-6)
    log_likelihoods, mean_trajectories = em_model.fit(X)
    
    # Check convergence
    assert em_model.converged_, "EM algorithm should converge"
    print(f"✓ EM algorithm converged in {em_model.n_iter_} iterations")
    
    # Check parameters
    assert em_model.means_ is not None, "Means should be initialized"
    assert em_model.covariances_ is not None, "Covariances should be initialized"
    assert em_model.weights_ is not None, "Weights should be initialized"
    print("✓ All parameters initialized correctly")
    
    # Check parameter shapes
    assert em_model.means_.shape == (2, 2), f"Means shape should be (2, 2), got {em_model.means_.shape}"
    assert em_model.covariances_.shape == (2, 2, 2), f"Covariances shape should be (2, 2, 2), got {em_model.covariances_.shape}"
    assert em_model.weights_.shape == (2,), f"Weights shape should be (2,), got {em_model.weights_.shape}"
    print("✓ Parameter shapes are correct")
    
    # Check weights sum to 1
    assert abs(em_model.weights_.sum() - 1.0) < 1e-10, "Weights should sum to 1"
    print("✓ Weights sum to 1")
    
    # Check log-likelihood increases (monotonic)
    for i in range(1, len(log_likelihoods)):
        assert log_likelihoods[i] >= log_likelihoods[i-1], "Log-likelihood should be non-decreasing"
    print("✓ Log-likelihood is non-decreasing")
    
    # Test prediction
    predictions = em_model.predict(X)
    assert len(predictions) == len(X), "Predictions should have same length as input"
    assert all(p in [0, 1] for p in predictions), "Predictions should be 0 or 1"
    print("✓ Predictions work correctly")
    
    print("\n🎉 All tests passed! EM algorithm implementation is working correctly.")
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Converged: {em_model.converged_}")
    print(f"Iterations: {em_model.n_iter_}")
    print(f"Final log-likelihood: {log_likelihoods[-1]:.4f}")
    print(f"Cluster 1 mean: {em_model.means_[0]}")
    print(f"Cluster 2 mean: {em_model.means_[1]}")
    print(f"Cluster 1 weight: {em_model.weights_[0]:.4f}")
    print(f"Cluster 2 weight: {em_model.weights_[1]:.4f}")

if __name__ == "__main__":
    test_em_algorithm()
