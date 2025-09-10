"""
Unit tests for JurisRank module
Author: Ignacio Adrián Lerer
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jurisrank.jurisrank import JurisRank

class TestJurisRank:
    """Test suite for JurisRank algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.jr = JurisRank(damping_factor=0.85, max_iterations=10, convergence_threshold=0.001)
        
        # Create sample data
        self.sample_cases = pd.DataFrame({
            'case_id': ['Case_A', 'Case_B', 'Case_C'],
            'date': ['2000-01-01', '2010-01-01', '2020-01-01'],
            'court_level': ['Supreme Court', 'Appeals Court', 'Lower Court']
        })
        
        # Simple citation matrix (B cites A, C cites B)
        self.citation_matrix = np.array([
            [0, 0, 0],  # Case A cites no one
            [1, 0, 0],  # Case B cites Case A
            [0, 1, 0]   # Case C cites Case B
        ])
    
    def test_initialization(self):
        """Test JurisRank initialization."""
        assert self.jr.damping_factor == 0.85
        assert self.jr.max_iterations == 10
        assert self.jr.convergence_threshold == 0.001
        
    def test_basic_calculation(self):
        """Test basic JurisRank calculation."""
        results = self.jr.calculate_jurisrank(self.citation_matrix, self.sample_cases)
        
        # Check that results are returned
        assert isinstance(results, dict)
        assert len(results) == 3
        
        # Check that all case IDs are present
        for case_id in self.sample_cases['case_id']:
            assert case_id in results
            
        # Check that scores sum to 1 (approximately)
        total_score = sum(results.values())
        assert abs(total_score - 1.0) < 0.01
        
    def test_temporal_weighting(self):
        """Test temporal decay functionality."""
        temporal_matrix = self.jr._apply_temporal_weights(self.citation_matrix, self.sample_cases)
        
        # Temporal matrix should have same shape
        assert temporal_matrix.shape == self.citation_matrix.shape
        
        # Values should be modified due to temporal decay
        # (newer cases citing older cases should have reduced weights)
        assert temporal_matrix[1, 0] < self.citation_matrix[1, 0]  # B->A should be reduced
        
    def test_hierarchical_weighting(self):
        """Test court hierarchy weighting."""
        hierarchical_matrix = self.jr._apply_hierarchical_weights(self.citation_matrix, self.sample_cases)
        
        # Matrix should have same shape
        assert hierarchical_matrix.shape == self.citation_matrix.shape
        
        # Supreme Court citations should have higher weight than Lower Court
        # Row 0 (Supreme Court) should have weight 1.0, Row 2 (Lower Court) should have weight 0.4
        if self.citation_matrix[0, :].sum() > 0:
            assert hierarchical_matrix[0, :].sum() >= hierarchical_matrix[2, :].sum()
    
    def test_matrix_normalization(self):
        """Test matrix normalization."""
        normalized = self.jr._normalize_matrix(self.citation_matrix.astype(float))
        
        # Each row should sum to 1 (or be handled for dangling nodes)
        for i in range(len(normalized)):
            row_sum = normalized[i, :].sum()
            assert abs(row_sum - 1.0) < 0.01 or row_sum == 0.0
            
    def test_empty_matrix(self):
        """Test handling of empty citation matrix."""
        empty_matrix = np.zeros((3, 3))
        
        # Should not crash and should return valid results
        results = self.jr.calculate_jurisrank(empty_matrix, self.sample_cases)
        
        assert isinstance(results, dict)
        assert len(results) == 3
        
        # With no citations, all cases should have equal fitness
        scores = list(results.values())
        for score in scores:
            assert abs(score - 1/3) < 0.01
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Non-square matrix
        with pytest.raises(ValueError):
            invalid_matrix = np.array([[1, 0], [0, 1], [1, 0]])
            self.jr.calculate_jurisrank(invalid_matrix, self.sample_cases)
        
        # Mismatched dimensions
        with pytest.raises(ValueError):
            wrong_cases = self.sample_cases.iloc[:2]  # Only 2 cases for 3x3 matrix
            self.jr.calculate_jurisrank(self.citation_matrix, wrong_cases)
    
    def test_fitness_evolution_tracking(self):
        """Test fitness evolution tracking."""
        # Calculate fitness and check history
        self.jr.calculate_jurisrank(self.citation_matrix, self.sample_cases)
        
        history = self.jr.get_fitness_evolution()
        assert isinstance(history, np.ndarray)
        assert history.shape[1] == 3  # 3 cases
        assert history.shape[0] > 0   # At least one iteration
    
    def test_doctrine_fitness_aggregation(self):
        """Test doctrine-level fitness calculation."""
        # Add doctrine column to sample data
        cases_with_doctrine = self.sample_cases.copy()
        cases_with_doctrine['primary_doctrine'] = ['Formalist', 'Emergency', 'Emergency']
        
        # Calculate individual fitness first
        fitness_scores = self.jr.calculate_jurisrank(self.citation_matrix, cases_with_doctrine)
        
        # Calculate doctrine-level fitness
        doctrine_fitness = self.jr.calculate_doctrine_fitness(
            fitness_scores, cases_with_doctrine, 'primary_doctrine'
        )
        
        assert isinstance(doctrine_fitness, dict)
        assert 'Formalist' in doctrine_fitness
        assert 'Emergency' in doctrine_fitness
        
    def test_fitness_leaders_identification(self):
        """Test identification of fitness leaders."""
        fitness_scores = self.jr.calculate_jurisrank(self.citation_matrix, self.sample_cases)
        
        leaders = self.jr.identify_fitness_leaders(fitness_scores, top_k=2)
        
        assert isinstance(leaders, list)
        assert len(leaders) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in leaders)
        
        # Should be sorted by fitness (descending)
        assert leaders[0][1] >= leaders[1][1]

if __name__ == "__main__":
    pytest.main([__file__])