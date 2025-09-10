"""
Legal-Memespace: Mapping Competitive Dynamics in Doctrinal Space
Models legal doctrine competition using Lotka-Volterra equations
Author: Ignacio Adrián Lerer
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.signal import find_peaks
import json

logger = logging.getLogger(__name__)

@dataclass
class PhaseTransition:
    """Represents a detected phase transition in doctrinal space."""
    date: str
    coordinates_before: List[float]
    coordinates_after: List[float]
    magnitude: float
    transition_type: str
    affected_dimensions: List[int]
    statistical_significance: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'date': self.date,
            'coordinates_before': self.coordinates_before,
            'coordinates_after': self.coordinates_after,
            'magnitude': self.magnitude,
            'transition_type': self.transition_type,
            'affected_dimensions': self.affected_dimensions,
            'statistical_significance': self.statistical_significance
        }

class LegalMemespace:
    """
    Maps legal doctrines in multi-dimensional space and models their competition.
    
    This class implements a comprehensive framework for analyzing legal doctrine
    evolution through dimensional reduction, competitive modeling, and phase
    transition detection.
    """
    
    def __init__(self, n_dimensions: int = 4, random_state: int = 42):
        """
        Initialize Legal-Memespace.
        
        Parameters:
        -----------
        n_dimensions : int
            Number of dimensions for doctrinal space (default: 4)
        random_state : int
            Random seed for reproducible results
        """
        self.n_dimensions = n_dimensions
        self.random_state = random_state
        self.pca = None
        self.scaler = StandardScaler()
        
        # Dimension interpretations for 4D space
        self.dimension_names = [
            'State_vs_Individual',      # Dimension 0: State power vs Individual rights
            'Emergency_vs_Normal',      # Dimension 1: Emergency powers vs Normal operations  
            'Formal_vs_Pragmatic',      # Dimension 2: Legal formalism vs Pragmatic interpretation
            'Temporary_vs_Permanent'    # Dimension 3: Temporary measures vs Permanent institutions
        ]
        
        # Extend dimension names if more dimensions requested
        while len(self.dimension_names) < n_dimensions:
            self.dimension_names.append(f'Dimension_{len(self.dimension_names)}')
            
        self.coordinates_cache = {}
        self.competition_parameters = {}
        
        np.random.seed(random_state)
        
    def map_doctrinal_space(self, cases: pd.DataFrame, 
                           feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Map legal doctrines to multi-dimensional space using PCA.
        
        Parameters:
        -----------
        cases : pd.DataFrame
            DataFrame with doctrinal features for each case
        feature_columns : List[str], optional
            Specific columns to use as features. If None, auto-detect.
            
        Returns:
        --------
        np.ndarray
            N x n_dimensions array of doctrinal coordinates
        """
        logger.info(f"Mapping {len(cases)} cases to {self.n_dimensions}D doctrinal space")
        
        # Extract or create features for PCA
        if feature_columns:
            if not all(col in cases.columns for col in feature_columns):
                missing = [col for col in feature_columns if col not in cases.columns]
                raise ValueError(f"Missing feature columns: {missing}")
            features = cases[feature_columns].values
        else:
            features = self._extract_or_create_features(cases)
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_dimensions, random_state=self.random_state)
        coordinates = self.pca.fit_transform(features_scaled)
        
        # Normalize coordinates to [0, 1] range for each dimension
        coordinates_normalized = np.zeros_like(coordinates)
        for i in range(self.n_dimensions):
            min_val = coordinates[:, i].min()
            max_val = coordinates[:, i].max()
            if max_val > min_val:
                coordinates_normalized[:, i] = (coordinates[:, i] - min_val) / (max_val - min_val)
            else:
                coordinates_normalized[:, i] = 0.5  # Default to center if no variation
        
        # Cache coordinates
        if 'case_id' in cases.columns:
            for idx, case_id in enumerate(cases['case_id']):
                self.coordinates_cache[case_id] = coordinates_normalized[idx]
        
        # Log PCA information
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        logger.info(f"PCA explained variance: {explained_variance}")
        logger.info(f"Cumulative variance explained: {cumulative_variance[-1]:.3f}")
        
        return coordinates_normalized
    
    def _extract_or_create_features(self, cases: pd.DataFrame) -> np.ndarray:
        """Extract or create features for PCA analysis."""
        features = []
        
        # Check for explicit feature columns
        feature_columns = [col for col in cases.columns if col.startswith('feature_')]
        
        if feature_columns:
            logger.info(f"Using {len(feature_columns)} explicit feature columns")
            return cases[feature_columns].values
        
        logger.info("Creating synthetic features from case attributes")
        
        # Create features from available case attributes
        for _, case in cases.iterrows():
            feature_vector = []
            
            # Core doctrinal dimensions
            feature_vector.extend([
                case.get('state_power', 0.5),           # State vs Individual
                case.get('emergency_level', 0.5),       # Emergency vs Normal
                case.get('formalism', 0.5),             # Formal vs Pragmatic  
                case.get('permanence', 0.5),            # Temporary vs Permanent
            ])
            
            # Additional doctrinal indicators
            feature_vector.extend([
                case.get('executive_power', 0.5),       # Executive authority
                case.get('legislative_deference', 0.5), # Legislative deference
                case.get('judicial_activism', 0.5),     # Judicial activism
                case.get('rights_protection', 0.5),     # Individual rights
                case.get('property_rights', 0.5),       # Property protection
                case.get('due_process', 0.5),           # Due process adherence
            ])
            
            # Court and temporal factors
            court_level_mapping = {
                'Supreme Court': 1.0,
                'Appeals Court': 0.8,
                'Federal Court': 0.6,
                'Provincial Supreme': 0.4,
                'Lower Court': 0.2
            }
            feature_vector.append(court_level_mapping.get(case.get('court_level'), 0.5))
            
            # Year normalized (assuming range 1900-2025)
            year = case.get('year', 1962)  # Default to middle of range
            normalized_year = (year - 1900) / 125.0
            feature_vector.append(normalized_year)
            
            # Economic context indicators
            feature_vector.extend([
                case.get('economic_crisis', 0.0),       # Economic crisis context
                case.get('inflation_context', 0.0),     # Inflation context
                case.get('fiscal_emergency', 0.0),      # Fiscal emergency
            ])
            
            # Doctrine-specific indicators
            feature_vector.extend([
                case.get('emergency_doctrine', 0.0),    # Emergency doctrine presence
                case.get('formalist_doctrine', 0.0),    # Formalist doctrine presence
                case.get('regulatory_state', 0.0),      # Regulatory state acceptance
                case.get('constitutional_strict', 0.0), # Strict constitutionalism
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in feature matrix."""
        # Replace NaN with column means
        features_clean = features.copy()
        
        for col in range(features_clean.shape[1]):
            col_data = features_clean[:, col]
            mask = ~np.isnan(col_data) & ~np.isinf(col_data)
            
            if mask.any():
                mean_val = np.mean(col_data[mask])
                features_clean[~mask, col] = mean_val
            else:
                features_clean[:, col] = 0.5  # Default neutral value
        
        return features_clean
    
    def simulate_competition(self, initial_populations: np.ndarray,
                           competition_matrix: np.ndarray,
                           time_points: np.ndarray,
                           growth_rates: Optional[np.ndarray] = None,
                           carrying_capacities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate competitive dynamics using Lotka-Volterra equations.
        
        Parameters:
        -----------
        initial_populations : np.ndarray
            Initial prevalence of each doctrine
        competition_matrix : np.ndarray
            Competition coefficients between doctrines (alpha_ij)
        time_points : np.ndarray
            Time points for simulation
        growth_rates : np.ndarray, optional
            Intrinsic growth rates for each doctrine
        carrying_capacities : np.ndarray, optional
            Carrying capacities for each doctrine
            
        Returns:
        --------
        np.ndarray
            Population trajectories over time (time_points x n_doctrines)
        """
        logger.info(f"Simulating competition for {len(initial_populations)} doctrines")
        
        n_doctrines = len(initial_populations)
        
        # Set default parameters if not provided
        if growth_rates is None:
            growth_rates = np.ones(n_doctrines) * 0.5
        if carrying_capacities is None:
            carrying_capacities = np.ones(n_doctrines) * 100
        
        # Store parameters for later analysis
        self.competition_parameters = {
            'growth_rates': growth_rates,
            'carrying_capacities': carrying_capacities,
            'competition_matrix': competition_matrix
        }
        
        def lotka_volterra(populations, t, r, K, alpha):
            """
            Generalized Lotka-Volterra competition equations.
            
            dN_i/dt = r_i * N_i * (1 - (sum(alpha_ij * N_j) / K_i))
            """
            n = len(populations)
            dpdt = np.zeros(n)
            
            for i in range(n):
                competition_effect = sum(alpha[i, j] * populations[j] for j in range(n))
                
                if K[i] > 0:  # Avoid division by zero
                    dpdt[i] = r[i] * populations[i] * (1 - competition_effect / K[i])
                else:
                    dpdt[i] = 0
                
                # Prevent negative populations
                if populations[i] <= 0:
                    dpdt[i] = max(0, dpdt[i])
            
            return dpdt
        
        # Simulate the system
        try:
            solution = odeint(
                lotka_volterra, 
                initial_populations,
                time_points,
                args=(growth_rates, carrying_capacities, competition_matrix)
            )
            
            logger.info("Competition simulation completed successfully")
            return solution
            
        except Exception as e:
            logger.error(f"Competition simulation failed: {e}")
            # Return simple exponential growth as fallback
            fallback_solution = np.zeros((len(time_points), n_doctrines))
            for i, t in enumerate(time_points):
                fallback_solution[i] = initial_populations * np.exp(growth_rates * t * 0.1)
            return fallback_solution
    
    def calculate_phase_transition(self, coordinates: np.ndarray,
                                  case_dates: pd.Series,
                                  window_size: int = 5,
                                  significance_threshold: float = 0.05) -> PhaseTransition:
        """
        Identify phase transitions in doctrinal space using statistical change-point detection.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Doctrinal coordinates over time
        case_dates : pd.Series
            Corresponding dates for each coordinate
        window_size : int
            Window size for local analysis
        significance_threshold : float
            Statistical significance threshold
            
        Returns:
        --------
        PhaseTransition
            Most significant phase transition detected
        """
        logger.info("Detecting phase transitions in doctrinal space")
        
        # Sort by date
        sorted_indices = case_dates.argsort()
        sorted_coords = coordinates[sorted_indices]
        sorted_dates = case_dates.iloc[sorted_indices]
        
        n_cases = len(sorted_coords)
        
        if n_cases < 2 * window_size:
            logger.warning("Insufficient data for phase transition detection")
            return self._create_null_transition(sorted_dates, sorted_coords)
        
        # Calculate moving statistics
        distances = []
        transition_indices = []
        
        for i in range(window_size, n_cases - window_size):
            # Calculate local change magnitude
            before_window = sorted_coords[i-window_size:i]
            after_window = sorted_coords[i:i+window_size]
            
            # Mean coordinates before and after
            mean_before = np.mean(before_window, axis=0)
            mean_after = np.mean(after_window, axis=0)
            
            # Euclidean distance between means
            distance = np.sqrt(np.sum((mean_after - mean_before)**2))
            distances.append(distance)
            transition_indices.append(i)
        
        if not distances:
            return self._create_null_transition(sorted_dates, sorted_coords)
        
        # Find peaks in distance profile
        distances_array = np.array(distances)
        peaks, _ = find_peaks(distances_array, height=np.percentile(distances_array, 75))
        
        if len(peaks) == 0:
            # Use maximum distance point if no peaks found
            max_idx = np.argmax(distances_array)
            peaks = [max_idx]
        
        # Select most significant transition
        max_distance_idx = peaks[np.argmax(distances_array[peaks])]
        transition_idx = transition_indices[max_distance_idx]
        
        # Calculate transition details
        before_coords = np.mean(sorted_coords[transition_idx-window_size:transition_idx], axis=0)
        after_coords = np.mean(sorted_coords[transition_idx:transition_idx+window_size], axis=0)
        magnitude = distances_array[max_distance_idx]
        
        # Determine affected dimensions
        coord_changes = np.abs(after_coords - before_coords)
        affected_dims = np.where(coord_changes > np.std(coord_changes))[0].tolist()
        
        # Statistical significance test
        p_value = self._calculate_transition_significance(
            sorted_coords[transition_idx-window_size:transition_idx],
            sorted_coords[transition_idx:transition_idx+window_size]
        )
        
        # Classify transition type
        transition_type = self._classify_transition_type(before_coords, after_coords, affected_dims)
        
        transition = PhaseTransition(
            date=str(sorted_dates.iloc[transition_idx]),
            coordinates_before=before_coords.tolist(),
            coordinates_after=after_coords.tolist(),
            magnitude=magnitude,
            transition_type=transition_type,
            affected_dimensions=affected_dims,
            statistical_significance=p_value
        )
        
        logger.info(f"Phase transition detected at {transition.date} with magnitude {magnitude:.3f}")
        
        return transition
    
    def _create_null_transition(self, dates: pd.Series, coords: np.ndarray) -> PhaseTransition:
        """Create a null transition when no significant change is detected."""
        mid_idx = len(dates) // 2
        return PhaseTransition(
            date=str(dates.iloc[mid_idx]),
            coordinates_before=[0.5] * self.n_dimensions,
            coordinates_after=[0.5] * self.n_dimensions,
            magnitude=0.0,
            transition_type="stable",
            affected_dimensions=[],
            statistical_significance=1.0
        )
    
    def _calculate_transition_significance(self, before_coords: np.ndarray,
                                         after_coords: np.ndarray) -> float:
        """Calculate statistical significance of transition using Hotelling's T² test."""
        try:
            from scipy.stats import f
            
            n1, n2 = len(before_coords), len(after_coords)
            p = before_coords.shape[1]  # number of dimensions
            
            # Calculate means
            mean1 = np.mean(before_coords, axis=0)
            mean2 = np.mean(after_coords, axis=0)
            
            # Calculate pooled covariance matrix
            cov1 = np.cov(before_coords.T)
            cov2 = np.cov(after_coords.T)
            pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
            
            # Hotelling's T² statistic
            mean_diff = mean1 - mean2
            
            # Add small regularization to handle singularity
            pooled_cov += np.eye(p) * 1e-6
            
            t_squared = (n1 * n2) / (n1 + n2) * mean_diff.T @ np.linalg.inv(pooled_cov) @ mean_diff
            
            # Convert to F-statistic
            f_stat = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * t_squared
            
            # Calculate p-value
            p_value = 1 - f.cdf(f_stat, p, n1 + n2 - p - 1)
            
            return p_value
            
        except Exception as e:
            logger.warning(f"Could not calculate transition significance: {e}")
            return 0.5  # Default moderate significance
    
    def _classify_transition_type(self, before_coords: np.ndarray, 
                                 after_coords: np.ndarray,
                                 affected_dims: List[int]) -> str:
        """Classify the type of phase transition."""
        coord_changes = after_coords - before_coords
        
        if len(affected_dims) == 0:
            return "stable"
        elif len(affected_dims) == 1:
            dim = affected_dims[0]
            if dim < len(self.dimension_names):
                dim_name = self.dimension_names[dim]
                direction = "increase" if coord_changes[dim] > 0 else "decrease"
                return f"unidimensional_{dim_name}_{direction}"
            else:
                return "unidimensional_shift"
        elif len(affected_dims) <= len(coord_changes) // 2:
            return "multidimensional_focused"
        else:
            return "systemic_transformation"
    
    def analyze_competitive_equilibrium(self, competition_matrix: np.ndarray,
                                      growth_rates: np.ndarray,
                                      carrying_capacities: np.ndarray) -> Dict[str, Union[np.ndarray, str]]:
        """
        Analyze competitive equilibrium and stability of the doctrinal system.
        
        Parameters:
        -----------
        competition_matrix : np.ndarray
            Competition coefficients
        growth_rates : np.ndarray
            Growth rates for each doctrine
        carrying_capacities : np.ndarray
            Carrying capacities
            
        Returns:
        --------
        Dict[str, Union[np.ndarray, str]]
            Analysis results including equilibrium points and stability
        """
        n_doctrines = len(growth_rates)
        
        # Find equilibrium points by solving the system
        # At equilibrium: r_i * (1 - (sum(alpha_ij * N_j) / K_i)) = 0
        
        # This gives us: sum(alpha_ij * N_j) = K_i for all i where N_i > 0
        
        results = {
            'n_doctrines': n_doctrines,
            'equilibrium_points': [],
            'stability_analysis': {},
            'dominant_doctrine': None,
            'coexistence_possible': False
        }
        
        try:
            # Check for competitive exclusion (one doctrine dominates)
            for i in range(n_doctrines):
                # Can doctrine i exclude all others?
                can_exclude_others = True
                equilibrium_pop_i = carrying_capacities[i] / competition_matrix[i, i]
                
                for j in range(n_doctrines):
                    if i != j:
                        # Check if doctrine j can invade when i is at equilibrium
                        invasion_growth = growth_rates[j] * (
                            1 - (competition_matrix[j, i] * equilibrium_pop_i) / carrying_capacities[j]
                        )
                        
                        if invasion_growth > 0:
                            can_exclude_others = False
                            break
                
                if can_exclude_others:
                    results['dominant_doctrine'] = i
                    results['equilibrium_points'].append({
                        'type': 'exclusion',
                        'populations': [equilibrium_pop_i if k == i else 0 for k in range(n_doctrines)],
                        'dominant': i
                    })
            
            # Check for coexistence equilibrium
            if results['dominant_doctrine'] is None:
                # Try to solve for coexistence equilibrium
                # This requires solving: alpha * N = K (where alpha is competition matrix)
                
                try:
                    equilibrium_pops = np.linalg.solve(competition_matrix, carrying_capacities)
                    
                    # Check if solution is feasible (all populations non-negative)
                    if np.all(equilibrium_pops >= 0):
                        results['coexistence_possible'] = True
                        results['equilibrium_points'].append({
                            'type': 'coexistence',
                            'populations': equilibrium_pops.tolist(),
                            'dominant': None
                        })
                        
                        # Analyze stability using eigenvalues of community matrix
                        community_matrix = self._calculate_community_matrix(
                            equilibrium_pops, competition_matrix, growth_rates, carrying_capacities
                        )
                        
                        eigenvalues = np.linalg.eigvals(community_matrix)
                        max_real_eigenvalue = np.max(np.real(eigenvalues))
                        
                        results['stability_analysis'] = {
                            'stable': max_real_eigenvalue < 0,
                            'max_eigenvalue': max_real_eigenvalue,
                            'eigenvalues': eigenvalues.tolist()
                        }
                        
                except np.linalg.LinAlgError:
                    logger.warning("Could not solve for coexistence equilibrium")
            
        except Exception as e:
            logger.error(f"Error in equilibrium analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_community_matrix(self, equilibrium_pops: np.ndarray,
                                   competition_matrix: np.ndarray,
                                   growth_rates: np.ndarray,
                                   carrying_capacities: np.ndarray) -> np.ndarray:
        """Calculate the community matrix for stability analysis."""
        n = len(equilibrium_pops)
        community_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements
                    community_matrix[i, i] = -growth_rates[i] * competition_matrix[i, i] * equilibrium_pops[i] / carrying_capacities[i]
                else:
                    # Off-diagonal elements
                    community_matrix[i, j] = -growth_rates[i] * competition_matrix[i, j] * equilibrium_pops[i] / carrying_capacities[i]
        
        return community_matrix
    
    def cluster_doctrinal_space(self, coordinates: np.ndarray,
                               n_clusters: Optional[int] = None,
                               max_clusters: int = 8) -> Tuple[np.ndarray, Dict]:
        """
        Cluster cases in doctrinal space to identify doctrinal families.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Doctrinal coordinates
        n_clusters : int, optional
            Number of clusters. If None, optimal number is determined.
        max_clusters : int
            Maximum number of clusters to consider
            
        Returns:
        --------
        Tuple[np.ndarray, Dict]
            Cluster labels and clustering information
        """
        if n_clusters is None:
            # Find optimal number of clusters using silhouette analysis
            silhouette_scores = []
            cluster_range = range(2, min(max_clusters + 1, len(coordinates)))
            
            for n in cluster_range:
                kmeans = KMeans(n_clusters=n, random_state=self.random_state)
                cluster_labels = kmeans.fit_predict(coordinates)
                silhouette_avg = silhouette_score(coordinates, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            if silhouette_scores:
                optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
            else:
                optimal_clusters = 3  # Default
        else:
            optimal_clusters = n_clusters
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Calculate cluster information
        cluster_info = {
            'n_clusters': optimal_clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_score(coordinates, cluster_labels) if len(set(cluster_labels)) > 1 else 0,
            'cluster_sizes': [np.sum(cluster_labels == i) for i in range(optimal_clusters)]
        }
        
        logger.info(f"Doctrinal space clustered into {optimal_clusters} families")
        
        return cluster_labels, cluster_info
    
    def export_analysis(self, coordinates: np.ndarray, case_metadata: pd.DataFrame,
                       output_path: str, include_plots: bool = True):
        """
        Export comprehensive memespace analysis to files.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Doctrinal coordinates
        case_metadata : pd.DataFrame
            Case metadata
        output_path : str
            Base path for output files
        include_plots : bool
            Whether to generate visualization plots
        """
        logger.info(f"Exporting memespace analysis to {output_path}")
        
        # Create analysis dictionary
        analysis = {
            'metadata': {
                'n_cases': len(coordinates),
                'n_dimensions': self.n_dimensions,
                'dimension_names': self.dimension_names,
                'pca_explained_variance': self.pca.explained_variance_ratio_.tolist() if self.pca else None
            },
            'coordinates': coordinates.tolist(),
            'case_ids': case_metadata['case_id'].tolist() if 'case_id' in case_metadata.columns else None
        }
        
        # Add phase transition analysis
        if 'date' in case_metadata.columns:
            phase_transition = self.calculate_phase_transition(coordinates, case_metadata['date'])
            analysis['phase_transition'] = phase_transition.to_dict()
        
        # Add clustering analysis
        cluster_labels, cluster_info = self.cluster_doctrinal_space(coordinates)
        analysis['clustering'] = cluster_info
        analysis['cluster_labels'] = cluster_labels.tolist()
        
        # Export to JSON
        json_path = f"{output_path}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Export to CSV
        csv_data = case_metadata.copy()
        for i in range(self.n_dimensions):
            csv_data[f'coord_{self.dimension_names[i]}'] = coordinates[:, i]
        csv_data['cluster_label'] = cluster_labels
        
        csv_path = f"{output_path}_coordinates.csv"
        csv_data.to_csv(csv_path, index=False)
        
        logger.info(f"Analysis exported to {json_path} and {csv_path}")
    
    def get_dimension_interpretation(self) -> Dict[str, str]:
        """
        Get interpretations for each dimension in the doctrinal space.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping dimension names to their interpretations
        """
        interpretations = {
            'State_vs_Individual': 
                'Measures the balance between state power and individual rights. '
                'Higher values indicate greater state authority, lower values favor individual liberty.',
            
            'Emergency_vs_Normal': 
                'Captures the degree of emergency powers and exceptional measures. '
                'Higher values represent emergency/crisis legal frameworks, lower values normal constitutional operation.',
            
            'Formal_vs_Pragmatic': 
                'Reflects the approach to legal interpretation and constitutional construction. '
                'Higher values indicate pragmatic/flexible interpretation, lower values strict formalist adherence.',
            
            'Temporary_vs_Permanent': 
                'Evaluates the temporal character of legal measures and institutions. '
                'Higher values represent permanent structural changes, lower values temporary/transitional measures.'
        }
        
        # Include any additional dimensions
        for i, dim_name in enumerate(self.dimension_names):
            if dim_name not in interpretations:
                interpretations[dim_name] = f"Dimension {i}: Extracted principal component capturing doctrinal variation."
        
        return interpretations