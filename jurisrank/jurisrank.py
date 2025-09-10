"""
JurisRank: Measuring Legal Doctrine Fitness Through Citation Networks
Adapts PageRank algorithm with temporal and hierarchical constraints
Author: Ignacio Adrián Lerer
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JurisRank:
    """
    Calculates memetic fitness scores for legal doctrines through citation network analysis.
    
    The JurisRank algorithm extends PageRank to incorporate legal-specific factors:
    - Temporal decay: Recent citations weighted higher than older ones
    - Hierarchical weighting: Higher court citations carry more weight
    - Doctrinal coherence: Cases with similar doctrines amplify each other
    """
    
    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 100, 
                 convergence_threshold: float = 0.0001, temporal_decay: float = 0.05):
        """
        Initialize JurisRank calculator.
        
        Parameters:
        -----------
        damping_factor : float
            Probability of following citations vs random jump (default: 0.85)
        max_iterations : int
            Maximum iterations before forced convergence (default: 100)
        convergence_threshold : float
            Threshold for convergence detection (default: 0.0001)
        temporal_decay : float
            Annual decay rate for temporal weighting (default: 0.05)
        """
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.temporal_decay = temporal_decay
        self.fitness_history = []
        
    def calculate_jurisrank(self, citation_matrix: np.ndarray, 
                           case_metadata: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate fitness scores for legal doctrines.
        
        Parameters:
        -----------
        citation_matrix : np.ndarray
            N x N matrix where element [i,j] represents citation from case i to case j
        case_metadata : pd.DataFrame
            Metadata including case names, dates, and court levels
            Required columns: 'case_id', 'date', 'court_level'
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping case IDs to fitness scores
        """
        logger.info("Starting JurisRank calculation...")
        
        # Validate inputs
        self._validate_inputs(citation_matrix, case_metadata)
        
        n_cases = len(citation_matrix)
        
        # Initialize with equal fitness
        fitness_scores = np.ones(n_cases) / n_cases
        
        # Apply temporal normalization
        logger.info("Applying temporal weights...")
        temporal_matrix = self._apply_temporal_weights(citation_matrix, case_metadata)
        
        # Apply hierarchical weights
        logger.info("Applying hierarchical weights...")
        hierarchical_matrix = self._apply_hierarchical_weights(temporal_matrix, case_metadata)
        
        # Apply doctrinal clustering weights
        logger.info("Applying doctrinal clustering...")
        clustered_matrix = self._apply_doctrinal_clustering(hierarchical_matrix, case_metadata)
        
        # Normalize to create transition matrix
        transition_matrix = self._normalize_matrix(clustered_matrix)
        
        # Power iteration with convergence tracking
        logger.info("Running power iteration...")
        for iteration in range(self.max_iterations):
            previous_scores = fitness_scores.copy()
            
            # PageRank calculation with damping
            fitness_scores = (
                (1 - self.damping_factor) / n_cases + 
                self.damping_factor * transition_matrix.T.dot(fitness_scores)
            )
            
            # Normalize to maintain probability distribution
            fitness_scores /= fitness_scores.sum()
            
            # Store history
            self.fitness_history.append(fitness_scores.copy())
            
            # Check convergence
            convergence_diff = np.abs(fitness_scores - previous_scores).sum()
            if convergence_diff < self.convergence_threshold:
                logger.info(f"Converged after {iteration + 1} iterations (diff: {convergence_diff:.6f})")
                break
        else:
            logger.warning(f"Did not converge after {self.max_iterations} iterations")
            
        # Create result dictionary
        results = {}
        for idx, case_id in enumerate(case_metadata['case_id']):
            results[case_id] = float(fitness_scores[idx])
            
        logger.info("JurisRank calculation completed.")
        return results
    
    def _validate_inputs(self, citation_matrix: np.ndarray, metadata: pd.DataFrame):
        """Validate input data."""
        if citation_matrix.shape[0] != citation_matrix.shape[1]:
            raise ValueError("Citation matrix must be square")
        
        if len(metadata) != citation_matrix.shape[0]:
            raise ValueError("Metadata length must match citation matrix dimensions")
            
        required_columns = ['case_id', 'date', 'court_level']
        missing_columns = [col for col in required_columns if col not in metadata.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in metadata: {missing_columns}")
    
    def _apply_temporal_weights(self, matrix: np.ndarray, 
                                metadata: pd.DataFrame) -> np.ndarray:
        """
        Apply temporal decay to citations.
        
        More recent citations receive higher weights using exponential decay.
        """
        weighted_matrix = matrix.copy().astype(float)
        dates = pd.to_datetime(metadata['date'])
        
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i][j] > 0:
                    # Calculate years between cases
                    delta_years = (dates.iloc[i] - dates.iloc[j]).days / 365.25
                    
                    # Apply exponential decay (only for backward citations)
                    if delta_years > 0:  # Citing case is newer than cited case
                        temporal_weight = np.exp(-self.temporal_decay * delta_years)
                        weighted_matrix[i][j] *= temporal_weight
                    else:
                        # Future citations shouldn't exist, but if they do, heavily penalize
                        weighted_matrix[i][j] *= 0.01
                        
        return weighted_matrix
    
    def _apply_hierarchical_weights(self, matrix: np.ndarray, 
                                    metadata: pd.DataFrame) -> np.ndarray:
        """
        Apply court hierarchy weights.
        
        Citations from higher courts receive more weight.
        """
        hierarchy_weights = {
            'Supreme Court': 1.0,
            'Appeals Court': 0.7,
            'Federal Court': 0.6,
            'Provincial Supreme': 0.5,
            'Lower Court': 0.4,
            'Administrative': 0.3
        }
        
        weighted_matrix = matrix.copy()
        for i in range(len(matrix)):
            court_level = metadata.iloc[i]['court_level']
            weight = hierarchy_weights.get(court_level, 0.5)
            weighted_matrix[i, :] *= weight
            
        return weighted_matrix
    
    def _apply_doctrinal_clustering(self, matrix: np.ndarray,
                                   metadata: pd.DataFrame) -> np.ndarray:
        """
        Apply doctrinal clustering weights.
        
        Cases with similar doctrinal elements amplify each other's citations.
        """
        weighted_matrix = matrix.copy()
        
        # Check if doctrinal similarity data is available
        if 'doctrinal_elements' not in metadata.columns:
            logger.warning("No doctrinal elements found, skipping clustering weights")
            return weighted_matrix
        
        # Calculate doctrinal similarity matrix
        n_cases = len(metadata)
        similarity_matrix = np.zeros((n_cases, n_cases))
        
        for i in range(n_cases):
            for j in range(n_cases):
                if i != j:
                    elements_i = set(metadata.iloc[i].get('doctrinal_elements', []))
                    elements_j = set(metadata.iloc[j].get('doctrinal_elements', []))
                    
                    if elements_i and elements_j:
                        # Jaccard similarity
                        intersection = len(elements_i & elements_j)
                        union = len(elements_i | elements_j)
                        similarity_matrix[i][j] = intersection / union if union > 0 else 0
        
        # Apply clustering boost
        clustering_boost = 1.0 + 0.5 * similarity_matrix  # Up to 50% boost for identical doctrines
        weighted_matrix *= clustering_boost
        
        return weighted_matrix
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize matrix to create stochastic transition matrix.
        
        Each row sums to 1, representing transition probabilities.
        """
        row_sums = matrix.sum(axis=1)
        
        # Handle dangling nodes (cases with no outgoing citations)
        dangling_nodes = row_sums == 0
        if dangling_nodes.any():
            logger.info(f"Found {dangling_nodes.sum()} dangling nodes (cases with no citations)")
            # Distribute dangling node probability uniformly
            matrix[dangling_nodes, :] = 1.0 / matrix.shape[1]
            row_sums = matrix.sum(axis=1)
        
        # Normalize rows
        return matrix / row_sums[:, np.newaxis]
    
    def get_fitness_evolution(self) -> np.ndarray:
        """
        Get the evolution of fitness scores during iteration.
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_iterations, n_cases) showing fitness evolution
        """
        if not self.fitness_history:
            raise ValueError("No fitness history available. Run calculate_jurisrank first.")
        
        return np.array(self.fitness_history)
    
    def calculate_doctrine_fitness(self, fitness_scores: Dict[str, float],
                                  case_metadata: pd.DataFrame,
                                  doctrine_column: str = 'primary_doctrine') -> Dict[str, float]:
        """
        Calculate aggregate fitness scores for legal doctrines.
        
        Parameters:
        -----------
        fitness_scores : Dict[str, float]
            Individual case fitness scores from calculate_jurisrank
        case_metadata : pd.DataFrame
            Case metadata with doctrine classifications
        doctrine_column : str
            Column name containing doctrine classifications
            
        Returns:
        --------
        Dict[str, float]
            Aggregate fitness scores by doctrine
        """
        if doctrine_column not in case_metadata.columns:
            raise ValueError(f"Column '{doctrine_column}' not found in metadata")
        
        doctrine_fitness = {}
        doctrine_counts = {}
        
        for idx, row in case_metadata.iterrows():
            case_id = row['case_id']
            doctrine = row[doctrine_column]
            
            if case_id in fitness_scores and pd.notna(doctrine):
                if doctrine not in doctrine_fitness:
                    doctrine_fitness[doctrine] = 0.0
                    doctrine_counts[doctrine] = 0
                
                doctrine_fitness[doctrine] += fitness_scores[case_id]
                doctrine_counts[doctrine] += 1
        
        # Calculate average fitness per doctrine
        for doctrine in doctrine_fitness:
            doctrine_fitness[doctrine] /= doctrine_counts[doctrine]
        
        return doctrine_fitness
    
    def identify_fitness_leaders(self, fitness_scores: Dict[str, float],
                                top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Identify the top-k cases by fitness score.
        
        Parameters:
        -----------
        fitness_scores : Dict[str, float]
            Case fitness scores
        top_k : int
            Number of top cases to return
            
        Returns:
        --------
        List[Tuple[str, float]]
            List of (case_id, fitness_score) tuples, sorted by fitness descending
        """
        sorted_cases = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_cases[:top_k]
    
    def calculate_temporal_evolution(self, citation_matrix: np.ndarray,
                                   case_metadata: pd.DataFrame,
                                   time_windows: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate fitness evolution across different time periods.
        
        Parameters:
        -----------
        citation_matrix : np.ndarray
            Full citation matrix
        case_metadata : pd.DataFrame
            Case metadata with dates
        time_windows : List[Tuple[str, str]]
            List of (start_date, end_date) tuples defining time periods
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Nested dict: {period_name: {case_id: fitness_score}}
        """
        temporal_fitness = {}
        dates = pd.to_datetime(case_metadata['date'])
        
        for i, (start_date, end_date) in enumerate(time_windows):
            period_name = f"Period_{i+1}_{start_date}_{end_date}"
            
            # Filter cases within time window
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            mask = (dates >= start_dt) & (dates <= end_dt)
            period_indices = mask[mask].index.tolist()
            
            if not period_indices:
                continue
            
            # Extract submatrix for this period
            period_matrix = citation_matrix[np.ix_(period_indices, period_indices)]
            period_metadata = case_metadata.iloc[period_indices].reset_index(drop=True)
            
            # Calculate fitness for this period
            period_fitness = self.calculate_jurisrank(period_matrix, period_metadata)
            temporal_fitness[period_name] = period_fitness
            
        return temporal_fitness