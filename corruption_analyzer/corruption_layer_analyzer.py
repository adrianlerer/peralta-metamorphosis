"""
CorruptionLayerAnalyzer: Quantifying Accumulative Evolution of Corruption
Analyzes how corruption accumulates in layers rather than evolving through substitution
Author: Ignacio Adrián Lerer

This module implements the biofilm model of corruption evolution, where multiple
corruption strategies coexist and mutually protect each other rather than
competing for replacement.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import logging
from scipy.stats import pearsonr
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)

class CorruptionLayerAnalyzer:
    """
    Analyzes the accumulation and persistence of corruption layers in legal systems.
    
    Based on biofilm model: multiple corruption species coexisting and protecting 
    each other, creating a resilient multilayer parasitic system.
    
    The four identified layers in Argentine corruption evolution:
    1. Electoral (1850-): Vote manipulation, clientelism, fraud
    2. Administrative (1912-): Bureaucratic capture, nepotism, ghost employees
    3. Entrepreneurial (1990-): Private sector capture, kickbacks, offshore structures
    4. Compliance Capture (2017-): Weaponization of anti-corruption tools
    """
    
    def __init__(self):
        """Initialize the analyzer with corruption layer definitions."""
        
        # Historical corruption layers with emergence dates and characteristics
        self.layers = {
            'electoral': {
                'origin_year': 1850,
                'peak_year': 1900,
                'description': 'Electoral manipulation and clientelistic networks',
                'indicators': [
                    'vote_buying', 'vote_chaining', 'ballot_theft',
                    'voter_transport', 'table_manipulation', 'fraudulent_counting',
                    'clientelistic_networks', 'patronage_systems'
                ],
                'base_persistence': 0.85,  # High historical persistence
                'decay_rate': 0.012,       # Slow decay due to structural embedding
                'resurgence_factor': 1.2   # Tendency to resurge in crisis periods
            },
            'administrative': {
                'origin_year': 1912,
                'peak_year': 1950,
                'description': 'Bureaucratic capture and administrative corruption',
                'indicators': [
                    'ghost_employees', 'directed_bidding', 'overpricing',
                    'nepotism', 'permit_selling', 'service_extortion',
                    'bureaucratic_capture', 'administrative_discretion_abuse'
                ],
                'base_persistence': 0.90,  # Very high persistence
                'decay_rate': 0.008,       # Very slow decay
                'resurgence_factor': 1.3   # Strong resurgence capability
            },
            'entrepreneurial': {
                'origin_year': 1990,
                'peak_year': 2010,
                'description': 'Private sector capture and sophisticated financial schemes',
                'indicators': [
                    'fake_consulting', 'offshore_structures', 'kickbacks',
                    'public_works_clubs', 'privatization_bribes', 'tax_evasion',
                    'revolving_doors', 'regulatory_capture'
                ],
                'base_persistence': 0.75,  # High but more volatile
                'decay_rate': 0.018,       # Faster adaptation to enforcement
                'resurgence_factor': 1.5   # High adaptation and mutation capability
            },
            'compliance_capture': {
                'origin_year': 2017,
                'peak_year': 2025,
                'description': 'Weaponization and capture of anti-corruption tools',
                'indicators': [
                    'cosmetic_programs', 'checkbox_compliance', 'defensive_use',
                    'consultant_capture', 'paper_training', 'fake_audits',
                    'ethics_washing', 'compliance_theater'
                ],
                'base_persistence': 0.95,  # Extremely high initial persistence
                'decay_rate': 0.005,       # Very slow decay (too new for significant pressure)
                'resurgence_factor': 2.0   # Maximum mutation and adaptation capability
            }
        }
        
        # Environmental factors affecting corruption layer evolution
        self.environmental_factors = {
            'democratic_consolidation': {
                'years': [1983, 2001],  # Democratic periods with varying strength
                'impact_on_layers': {
                    'electoral': -0.3,      # Reduces electoral corruption
                    'administrative': 0.1,   # May increase bureaucratic corruption
                    'entrepreneurial': 0.2,  # Increases private sector opportunities
                    'compliance_capture': 0.0
                }
            },
            'economic_crisis': {
                'years': [1989, 2001, 2018],
                'impact_on_layers': {
                    'electoral': 0.4,        # Crisis increases clientelism
                    'administrative': 0.5,   # Desperation increases admin corruption
                    'entrepreneurial': -0.2, # Reduces private resources
                    'compliance_capture': 0.1
                }
            },
            'international_pressure': {
                'years': [1999, 2016, 2018],  # FATF, OECD pressures
                'impact_on_layers': {
                    'electoral': -0.1,
                    'administrative': -0.2,
                    'entrepreneurial': -0.4,  # Strong pressure on financial schemes
                    'compliance_capture': 0.6  # Paradoxically increases compliance theater
                }
            }
        }
        
        # Cache for performance optimization
        self._persistence_cache = {}
        self._interaction_cache = {}
        
    def measure_layer_persistence(self, corruption_cases: pd.DataFrame, 
                                 year: int, use_cache: bool = True) -> Dict[str, float]:
        """
        Measure what percentage of each corruption layer persists at a given year.
        
        Uses a sophisticated model incorporating:
        - Base persistence rates
        - Temporal decay with resistance
        - Environmental factors
        - Actual case evidence
        - Cross-layer protection effects
        
        Parameters:
        -----------
        corruption_cases : pd.DataFrame
            Dataset with corruption cases including year, layer, and outcome
        year : int
            Year to measure persistence (1850-2025)
        use_cache : bool
            Whether to use cached results for performance
            
        Returns:
        --------
        Dict[str, float]
            Persistence score for each layer (0-1), where 1 means fully active
        """
        # Check cache first
        cache_key = f"{year}_{len(corruption_cases)}"
        if use_cache and cache_key in self._persistence_cache:
            return self._persistence_cache[cache_key]
        
        logger.debug(f"Calculating layer persistence for year {year}")
        
        persistence_scores = {}
        
        for layer_name, layer_info in self.layers.items():
            # Base calculation based on layer lifecycle
            years_since_origin = max(0, year - layer_info['origin_year'])
            years_since_peak = year - layer_info['peak_year']
            
            # Calculate base persistence
            if year < layer_info['origin_year']:
                # Layer hasn't emerged yet
                base_persistence = 0.0
            elif year <= layer_info['peak_year']:
                # Growing phase: S-curve growth to peak
                growth_factor = years_since_origin / (layer_info['peak_year'] - layer_info['origin_year'])
                base_persistence = layer_info['base_persistence'] * (1 / (1 + np.exp(-5 * (growth_factor - 0.5))))
            else:
                # Decay phase with resistance
                decay_years = years_since_peak
                decay_rate = layer_info['decay_rate']
                min_persistence = 0.1  # Minimum survival level
                
                base_persistence = max(
                    min_persistence,
                    layer_info['base_persistence'] * np.exp(-decay_rate * decay_years)
                )
            
            # Apply environmental factors
            environmental_adjustment = self._calculate_environmental_impact(layer_name, year)
            adjusted_persistence = base_persistence * (1 + environmental_adjustment)
            
            # Adjust based on actual evidence in dataset
            evidence_adjustment = self._calculate_evidence_adjustment(
                corruption_cases, layer_name, year
            )
            final_persistence = adjusted_persistence * evidence_adjustment
            
            # Apply cross-layer protection (biofilm effect)
            if len(persistence_scores) > 0:  # If other layers already calculated
                protection_bonus = self._calculate_protection_bonus(
                    layer_name, persistence_scores, year
                )
                final_persistence *= (1 + protection_bonus)
            
            # Ensure bounds [0, 1]
            persistence_scores[layer_name] = max(0.0, min(1.0, final_persistence))
            
        # Cache the result
        if use_cache:
            self._persistence_cache[cache_key] = persistence_scores
            
        return persistence_scores
    
    def _calculate_environmental_impact(self, layer_name: str, year: int) -> float:
        """Calculate environmental factor impacts on layer persistence."""
        
        total_impact = 0.0
        
        for factor_name, factor_info in self.environmental_factors.items():
            # Check if year falls within influence period of this factor
            for factor_year in factor_info['years']:
                years_diff = abs(year - factor_year)
                
                # Exponential decay of factor influence over time
                if years_diff <= 10:  # Factors have 10-year influence window
                    influence_strength = np.exp(-0.1 * years_diff)
                    layer_impact = factor_info['impact_on_layers'].get(layer_name, 0)
                    total_impact += influence_strength * layer_impact
        
        return total_impact
    
    def _calculate_evidence_adjustment(self, cases: pd.DataFrame, 
                                     layer_name: str, year: int) -> float:
        """Adjust persistence based on actual case evidence."""
        
        if cases.empty:
            return 1.0  # No adjustment if no data
        
        # Look at cases within ±5 years of target year
        relevant_cases = cases[
            (cases['layer'] == layer_name) &
            (abs(cases['year'] - year) <= 5)
        ]
        
        if len(relevant_cases) == 0:
            # No recent evidence - apply mild reduction
            return 0.8
        
        # Analyze outcomes to determine activity level
        positive_outcomes = ['unchallenged', 'normalized', 'cosmetic', 'defensive', 'emerging']
        negative_outcomes = ['prosecuted', 'reform_attempt', 'effective', 'challenged']
        
        positive_cases = len(relevant_cases[relevant_cases['outcome'].isin(positive_outcomes)])
        negative_cases = len(relevant_cases[relevant_cases['outcome'].isin(negative_outcomes)])
        
        if positive_cases + negative_cases == 0:
            return 1.0
        
        # Calculate activity ratio
        success_ratio = positive_cases / (positive_cases + negative_cases)
        
        # Convert to adjustment factor (0.5 to 1.5 range)
        adjustment = 0.5 + success_ratio
        
        # Apply case frequency bonus
        case_frequency = len(relevant_cases) / 5  # Cases per year in window
        frequency_bonus = min(0.3, case_frequency * 0.1)
        
        return adjustment + frequency_bonus
    
    def _calculate_protection_bonus(self, layer_name: str, 
                                   other_layers: Dict[str, float], 
                                   year: int) -> float:
        """Calculate protection bonus from other active layers (biofilm effect)."""
        
        # Protection matrix: how much each layer protects others
        protection_matrix = {
            'electoral': {
                'administrative': 0.3,      # Electoral networks protect admin corruption
                'entrepreneurial': 0.2,     # Provides political cover
                'compliance_capture': 0.1
            },
            'administrative': {
                'electoral': 0.4,           # Admin resources fund electoral operations
                'entrepreneurial': 0.3,     # Creates bureaucratic shields
                'compliance_capture': 0.2
            },
            'entrepreneurial': {
                'electoral': 0.3,           # Provides funding for electoral corruption
                'administrative': 0.4,      # Creates revolving door opportunities
                'compliance_capture': 0.5   # Captures compliance systems
            },
            'compliance_capture': {
                'electoral': 0.2,           # Provides legal cover
                'administrative': 0.3,      # Creates procedural shields
                'entrepreneurial': 0.6      # Strongest protection for business corruption
            }
        }
        
        protection_bonus = 0.0
        
        for protector_layer, persistence in other_layers.items():
            if protector_layer != layer_name and persistence > 0.1:
                # Only active layers (>10% persistence) provide protection
                protection_coefficient = protection_matrix.get(
                    protector_layer, {}
                ).get(layer_name, 0)
                
                protection_bonus += persistence * protection_coefficient
        
        return min(0.5, protection_bonus)  # Cap protection bonus at 50%
    
    def calculate_accumulation_index(self, historical_data: pd.DataFrame, 
                                   start_year: int = 1880, 
                                   end_year: int = 2025) -> float:
        """
        Calculate the Accumulation Index measuring whether corruption layers
        accumulate (coexist) or substitute (replace) each other.
        
        Range: 0 = pure substitution, 1 = pure accumulation
        
        The algorithm measures:
        1. Cross-temporal correlation between layers
        2. Simultaneous activity patterns
        3. Persistence despite new layer emergence
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical corruption cases dataset
        start_year : int
            Starting year for analysis (default: 1880)
        end_year : int
            Ending year for analysis (default: 2025)
            
        Returns:
        --------
        float
            Accumulation index (0-1), higher values indicate more accumulation
        """
        logger.info("Calculating Accumulation Index for corruption layer evolution")
        
        # Generate persistence data for analysis period
        years = list(range(start_year, end_year + 1, 5))  # Every 5 years
        layer_trajectories = {layer: [] for layer in self.layers.keys()}
        
        for year in years:
            persistence = self.measure_layer_persistence(historical_data, year)
            for layer, score in persistence.items():
                layer_trajectories[layer].append(score)
        
        # Component 1: Cross-layer correlations (40% weight)
        correlation_component = self._calculate_correlation_component(layer_trajectories)
        
        # Component 2: Simultaneous activity patterns (35% weight)
        simultaneity_component = self._calculate_simultaneity_component(layer_trajectories, years)
        
        # Component 3: Persistence despite succession (25% weight)
        persistence_component = self._calculate_persistence_component(layer_trajectories, years)
        
        # Weighted combination
        accumulation_index = (
            correlation_component * 0.40 +
            simultaneity_component * 0.35 +
            persistence_component * 0.25
        )
        
        logger.info(f"Accumulation Index components: correlation={correlation_component:.3f}, "
                   f"simultaneity={simultaneity_component:.3f}, persistence={persistence_component:.3f}")
        
        return min(1.0, max(0.0, accumulation_index))
    
    def _calculate_correlation_component(self, trajectories: Dict[str, List[float]]) -> float:
        """Calculate cross-layer correlation component of accumulation index."""
        
        correlations = []
        layers_list = list(trajectories.keys())
        
        for i in range(len(layers_list)):
            for j in range(i+1, len(layers_list)):
                traj1 = trajectories[layers_list[i]]
                traj2 = trajectories[layers_list[j]]
                
                if len(traj1) > 2 and len(traj2) > 2:  # Need minimum data points
                    try:
                        corr, p_value = pearsonr(traj1, traj2)
                        if not np.isnan(corr) and p_value < 0.1:  # Significant correlation
                            correlations.append(abs(corr))
                    except Exception:
                        continue
        
        # High positive correlation indicates accumulation
        return np.mean(correlations) if correlations else 0.5
    
    def _calculate_simultaneity_component(self, trajectories: Dict[str, List[float]], 
                                        years: List[int]) -> float:
        """Calculate simultaneity component based on concurrent layer activity."""
        
        simultaneity_scores = []
        
        for i, year in enumerate(years):
            if year >= 1950:  # Only analyze modern period with multiple layers
                active_layers = 0
                total_activity = 0
                
                for layer_name, trajectory in trajectories.items():
                    if i < len(trajectory):
                        persistence = trajectory[i]
                        if persistence > 0.1:  # Consider active if >10%
                            active_layers += 1
                            total_activity += persistence
                
                # Simultaneity score: number of active layers weighted by total activity
                if active_layers > 0:
                    simultaneity_score = (active_layers - 1) / 3 * (total_activity / active_layers)
                    simultaneity_scores.append(simultaneity_score)
        
        return np.mean(simultaneity_scores) if simultaneity_scores else 0.0
    
    def _calculate_persistence_component(self, trajectories: Dict[str, List[float]], 
                                       years: List[int]) -> float:
        """Calculate persistence component measuring old layer survival."""
        
        persistence_scores = []
        
        for i, year in enumerate(years):
            if year >= 1990:  # Modern period analysis
                # Check how well old layers persist despite new ones emerging
                electoral_persistence = trajectories['electoral'][i] if i < len(trajectories['electoral']) else 0
                admin_persistence = trajectories['administrative'][i] if i < len(trajectories['administrative']) else 0
                
                # Old layers should decline in substitution model, persist in accumulation
                old_layer_survival = (electoral_persistence + admin_persistence) / 2
                persistence_scores.append(old_layer_survival)
        
        return np.mean(persistence_scores) if persistence_scores else 0.0
    
    def analyze_layer_interaction(self, cases: pd.DataFrame, 
                                 year_window: int = 10) -> Dict:
        """
        Analyze how different corruption layers interact and protect each other.
        
        Examines:
        - Co-occurrence patterns
        - Temporal clustering
        - Mutual reinforcement
        - Protection coefficients
        
        Parameters:
        -----------
        cases : pd.DataFrame
            Corruption cases dataset
        year_window : int
            Time window for analyzing interactions (default: 10 years)
            
        Returns:
        --------
        Dict
            Comprehensive interaction analysis including matrices and coefficients
        """
        logger.info("Analyzing corruption layer interactions and mutual protection")
        
        layers_list = list(self.layers.keys())
        n_layers = len(layers_list)
        
        # Initialize interaction matrices
        co_occurrence_matrix = np.zeros((n_layers, n_layers))
        protection_matrix = np.zeros((n_layers, n_layers))
        temporal_clustering_matrix = np.zeros((n_layers, n_layers))
        
        # Analyze co-occurrence and temporal patterns
        for i, layer1 in enumerate(layers_list):
            for j, layer2 in enumerate(layers_list):
                if i != j:
                    interaction_score = self._calculate_pairwise_interaction(
                        cases, layer1, layer2, year_window
                    )
                    co_occurrence_matrix[i, j] = interaction_score['co_occurrence']
                    protection_matrix[i, j] = interaction_score['protection']
                    temporal_clustering_matrix[i, j] = interaction_score['temporal_clustering']
        
        # Calculate aggregate protection coefficients
        protection_coefficients = {}
        for i, protected_layer in enumerate(layers_list):
            total_protection = 0
            for j, protecting_layer in enumerate(layers_list):
                if i != j:
                    total_protection += protection_matrix[j, i]  # Protection received
            
            protection_coefficients[protected_layer] = total_protection / (n_layers - 1)
        
        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(
            co_occurrence_matrix, layers_list
        )
        
        return {
            'co_occurrence_matrix': co_occurrence_matrix.tolist(),
            'protection_matrix': protection_matrix.tolist(),
            'temporal_clustering_matrix': temporal_clustering_matrix.tolist(),
            'protection_coefficients': protection_coefficients,
            'network_metrics': network_metrics,
            'layers': layers_list,
            'analysis_params': {
                'year_window': year_window,
                'cases_analyzed': len(cases)
            }
        }
    
    def _calculate_pairwise_interaction(self, cases: pd.DataFrame, 
                                      layer1: str, layer2: str, 
                                      year_window: int) -> Dict[str, float]:
        """Calculate interaction metrics between two specific layers."""
        
        layer1_cases = cases[cases['layer'] == layer1]
        layer2_cases = cases[cases['layer'] == layer2]
        
        # Co-occurrence: cases of both layers in same time periods
        co_occurrence_score = 0
        protection_score = 0
        temporal_clustering_score = 0
        
        if len(layer1_cases) > 0 and len(layer2_cases) > 0:
            # Analyze temporal windows
            for _, case1 in layer1_cases.iterrows():
                case1_year = case1['year']
                
                # Find layer2 cases in temporal window
                nearby_cases = layer2_cases[
                    abs(layer2_cases['year'] - case1_year) <= year_window
                ]
                
                if len(nearby_cases) > 0:
                    co_occurrence_score += 1
                    
                    # Protection analysis: successful layer1 cases near layer2 activity
                    if case1.get('outcome', '') in ['unchallenged', 'cosmetic', 'normalized']:
                        protection_score += len(nearby_cases)
                    
                    # Temporal clustering: cases very close in time
                    very_close = nearby_cases[abs(nearby_cases['year'] - case1_year) <= 2]
                    temporal_clustering_score += len(very_close)
        
        # Normalize scores
        max_cases = max(len(layer1_cases), len(layer2_cases), 1)
        
        return {
            'co_occurrence': min(1.0, co_occurrence_score / max_cases),
            'protection': min(1.0, protection_score / (max_cases * 5)),
            'temporal_clustering': min(1.0, temporal_clustering_score / max_cases)
        }
    
    def _calculate_network_metrics(self, matrix: np.ndarray, 
                                  layers: List[str]) -> Dict[str, float]:
        """Calculate network-level metrics for layer interaction system."""
        
        # Network density: proportion of possible connections that exist
        n = len(layers)
        total_possible = n * (n - 1)
        actual_connections = np.sum(matrix > 0.1)  # Threshold for meaningful connection
        density = actual_connections / total_possible if total_possible > 0 else 0
        
        # Average clustering coefficient
        clustering_coeffs = []
        for i in range(n):
            neighbors = [j for j in range(n) if matrix[i, j] > 0.1 and i != j]
            if len(neighbors) > 1:
                # Calculate clustering among neighbors
                neighbor_connections = 0
                possible_neighbor_connections = len(neighbors) * (len(neighbors) - 1)
                
                for j in neighbors:
                    for k in neighbors:
                        if j != k and matrix[j, k] > 0.1:
                            neighbor_connections += 1
                
                clustering = neighbor_connections / possible_neighbor_connections
                clustering_coeffs.append(clustering)
        
        avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0
        
        # System resilience: how well connected the system remains if strongest layer removed
        layer_strengths = np.sum(matrix, axis=1)
        if len(layer_strengths) > 0:
            strongest_idx = np.argmax(layer_strengths)
            reduced_matrix = np.delete(np.delete(matrix, strongest_idx, axis=0), strongest_idx, axis=1)
            
            if reduced_matrix.size > 0:
                reduced_density = np.sum(reduced_matrix > 0.1) / (reduced_matrix.size - np.trace(reduced_matrix != 0))
                resilience = reduced_density / density if density > 0 else 0
            else:
                resilience = 0
        else:
            resilience = 0
        
        return {
            'network_density': density,
            'average_clustering': avg_clustering,
            'system_resilience': resilience,
            'total_connections': int(actual_connections),
            'strongest_layer': layers[np.argmax(layer_strengths)] if len(layer_strengths) > 0 else None
        }
    
    def predict_next_mutation(self, current_state: pd.DataFrame, 
                             prediction_horizon: int = 5) -> Dict:
        """
        Predict the next likely mutation or adaptation in corruption patterns.
        
        Uses pressure analysis, historical patterns, and environmental factors
        to forecast emerging corruption strategies.
        
        Parameters:
        -----------
        current_state : pd.DataFrame
            Recent corruption cases dataset (last 5-10 years)
        prediction_horizon : int
            Years into future to predict (default: 5)
            
        Returns:
        --------
        Dict
            Predictions with probabilities, timeline, and detailed analysis
        """
        logger.info("Analyzing corruption evolution pressures and predicting mutations")
        
        current_year = datetime.now().year
        recent_cutoff = current_year - 5
        
        # Filter to recent cases
        recent_cases = current_state[current_state['year'] >= recent_cutoff]
        
        # Analyze enforcement pressure by layer
        enforcement_pressure = self._analyze_enforcement_pressure(recent_cases)
        
        # Analyze technological and regulatory environment
        environmental_pressures = self._analyze_environmental_pressures(current_year)
        
        # Generate mutation predictions
        mutation_predictions = self._generate_mutation_predictions(
            enforcement_pressure, environmental_pressures, prediction_horizon
        )
        
        # Calculate overall mutation pressure
        overall_pressure = np.mean(list(enforcement_pressure.values()))
        
        return {
            'enforcement_pressure': enforcement_pressure,
            'environmental_pressures': environmental_pressures,
            'predictions': sorted(mutation_predictions, key=lambda x: x['probability'], reverse=True),
            'overall_mutation_pressure': overall_pressure,
            'analysis_date': current_year,
            'prediction_horizon': prediction_horizon,
            'methodology': 'Biofilm adaptation model with enforcement pressure analysis'
        }
    
    def _analyze_enforcement_pressure(self, recent_cases: pd.DataFrame) -> Dict[str, float]:
        """Analyze enforcement pressure on each corruption layer."""
        
        enforcement_pressure = {}
        
        for layer in self.layers.keys():
            layer_cases = recent_cases[recent_cases['layer'] == layer]
            
            if len(layer_cases) == 0:
                enforcement_pressure[layer] = 0.0
                continue
            
            # Classify outcomes by enforcement severity
            high_pressure_outcomes = ['prosecuted', 'mass_prosecution', 'reform_attempt']
            medium_pressure_outcomes = ['challenged', 'scandal', 'partial_prosecution']
            low_pressure_outcomes = ['documented', 'emerging']
            no_pressure_outcomes = ['unchallenged', 'normalized', 'cosmetic', 'defensive']
            
            high_pressure = len(layer_cases[layer_cases['outcome'].isin(high_pressure_outcomes)])
            medium_pressure = len(layer_cases[layer_cases['outcome'].isin(medium_pressure_outcomes)])
            total_cases = len(layer_cases)
            
            # Calculate pressure score (0-1)
            pressure_score = (high_pressure * 1.0 + medium_pressure * 0.5) / total_cases
            enforcement_pressure[layer] = pressure_score
        
        return enforcement_pressure
    
    def _analyze_environmental_pressures(self, current_year: int) -> Dict[str, float]:
        """Analyze environmental pressures that could drive mutations."""
        
        pressures = {
            'technological_advancement': 0.8,    # AI, blockchain, digital surveillance
            'international_oversight': 0.6,     # FATF, OECD, transparency initiatives
            'civil_society_monitoring': 0.7,    # NGO watchdogs, investigative journalism
            'economic_crisis': 0.5,             # Economic stress driving innovation
            'regulatory_sophistication': 0.9,   # Increasingly complex regulations
            'digital_transformation': 0.85      # Shift to digital governance and services
        }
        
        # Adjust based on current events and trends (simplified model)
        # In real implementation, this would incorporate actual indicators
        
        return pressures
    
    def _generate_mutation_predictions(self, enforcement_pressure: Dict[str, float],
                                     environmental_pressures: Dict[str, float],
                                     horizon: int) -> List[Dict]:
        """Generate specific mutation predictions based on pressure analysis."""
        
        predictions = []
        
        # High compliance pressure → AI-enhanced compliance theater
        if enforcement_pressure.get('compliance_capture', 0) > 0.3:
            predictions.append({
                'mutation': 'AI-generated compliance reports and training',
                'probability': 0.75,
                'timeline': '12-18 months',
                'description': 'Automated generation of convincing but hollow compliance documentation using large language models',
                'threat_level': 'High',
                'countermeasures': ['Human verification systems', 'Substance-over-form auditing'],
                'driving_factors': ['High compliance enforcement', 'AI technology availability']
            })
        
        # High electoral pressure + digital transformation → digital clientelism
        if (enforcement_pressure.get('electoral', 0) > 0.2 and 
            environmental_pressures.get('digital_transformation', 0) > 0.7):
            predictions.append({
                'mutation': 'Algorithmic clientelism through mobile apps',
                'probability': 0.68,
                'timeline': '6-12 months',
                'description': 'Using mobile apps and social media algorithms to coordinate vote buying and benefits distribution with greater precision and deniability',
                'threat_level': 'Very High',
                'countermeasures': ['Digital platform regulation', 'Algorithmic auditing'],
                'driving_factors': ['Electoral pressure', 'Smartphone penetration', 'Social media ubiquity']
            })
        
        # High entrepreneurial pressure → crypto-corruption
        if enforcement_pressure.get('entrepreneurial', 0) > 0.4:
            predictions.append({
                'mutation': 'Cryptocurrency-based corruption networks',
                'probability': 0.55,
                'timeline': '18-36 months',
                'description': 'Integration of privacy coins and DeFi protocols for untraceable bribery and money laundering',
                'threat_level': 'Medium',
                'countermeasures': ['Crypto transaction monitoring', 'DeFi protocol regulation'],
                'driving_factors': ['Financial surveillance pressure', 'Crypto adoption', 'DeFi innovation']
            })
        
        # Low overall pressure → expansion and consolidation
        if np.mean(list(enforcement_pressure.values())) < 0.15:
            predictions.append({
                'mutation': 'Cross-layer integration and systematization',
                'probability': 0.45,
                'timeline': '24-48 months',
                'description': 'Formal integration of corruption layers into systematic, resilient networks with shared protection mechanisms',
                'threat_level': 'Extreme',
                'countermeasures': ['System-wide integrity reforms', 'Cross-sector coordination'],
                'driving_factors': ['Low enforcement pressure', 'Institutional capture']
            })
        
        # High regulatory sophistication → regulatory arbitrage
        if environmental_pressures.get('regulatory_sophistication', 0) > 0.8:
            predictions.append({
                'mutation': 'Regulatory arbitrage and forum shopping',
                'probability': 0.62,
                'timeline': '6-18 months',
                'description': 'Exploiting differences between regulatory frameworks and jurisdictions to minimize oversight',
                'threat_level': 'High',
                'countermeasures': ['Regulatory harmonization', 'International cooperation'],
                'driving_factors': ['Complex regulations', 'Jurisdictional differences']
            })
        
        # Emerging: ESG capture
        predictions.append({
            'mutation': 'ESG and sustainability compliance capture',
            'probability': 0.58,
            'timeline': '12-24 months',
            'description': 'Weaponization of Environmental, Social, and Governance frameworks for competitive advantage and regulatory shield',
            'threat_level': 'Medium',
            'countermeasures': ['Independent ESG verification', 'Stakeholder involvement'],
            'driving_factors': ['ESG regulation growth', 'Greenwashing precedents']
        })
        
        return predictions
    
    def generate_biofilm_score(self, cases: pd.DataFrame, year: int, 
                              detailed: bool = False) -> float:
        """
        Calculate the overall 'biofilm' score showing system-wide corruption protection.
        
        The biofilm score measures how well corruption layers protect each other,
        creating a resilient, interconnected system similar to bacterial biofilms.
        
        Components:
        1. Layer diversity (number of active layers)
        2. Average persistence (strength of each layer)
        3. Mutual protection (cross-layer defense)
        4. System redundancy (backup mechanisms)
        
        Parameters:
        -----------
        cases : pd.DataFrame
            Corruption cases dataset
        year : int
            Year to calculate biofilm score
        detailed : bool
            Whether to return detailed component breakdown
            
        Returns:
        --------
        float or Dict
            Biofilm score (0-1) or detailed breakdown if requested
        """
        # Calculate layer persistence
        persistence = self.measure_layer_persistence(cases, year)
        
        # Get interaction analysis
        interactions = self.analyze_layer_interaction(cases)
        
        # Component 1: Layer diversity (30% weight)
        active_layers = sum(1 for score in persistence.values() if score > 0.1)
        max_possible_layers = len(self.layers)
        diversity_score = active_layers / max_possible_layers
        
        # Component 2: Average persistence strength (25% weight)
        avg_persistence = np.mean(list(persistence.values()))
        
        # Component 3: Mutual protection strength (35% weight)
        protection_scores = list(interactions['protection_coefficients'].values())
        avg_protection = np.mean(protection_scores) if protection_scores else 0
        
        # Component 4: System redundancy (10% weight)
        # Measure how well system survives if strongest layer is removed
        network_metrics = interactions['network_metrics']
        redundancy_score = network_metrics.get('system_resilience', 0)
        
        # Calculate weighted biofilm score
        biofilm_score = (
            diversity_score * 0.30 +
            avg_persistence * 0.25 +
            avg_protection * 0.35 +
            redundancy_score * 0.10
        )
        
        if detailed:
            return {
                'biofilm_score': min(1.0, biofilm_score),
                'components': {
                    'diversity_score': diversity_score,
                    'avg_persistence': avg_persistence,
                    'avg_protection': avg_protection,
                    'redundancy_score': redundancy_score
                },
                'layer_details': {
                    'active_layers': active_layers,
                    'total_layers': max_possible_layers,
                    'persistence_by_layer': persistence,
                    'protection_by_layer': interactions['protection_coefficients']
                },
                'interpretation': self._interpret_biofilm_score(biofilm_score)
            }
        
        return min(1.0, biofilm_score)
    
    def _interpret_biofilm_score(self, score: float) -> str:
        """Provide human-readable interpretation of biofilm score."""
        
        if score >= 0.8:
            return "Extremely resilient corruption system with strong mutual protection"
        elif score >= 0.6:
            return "High corruption resilience with significant cross-layer protection"
        elif score >= 0.4:
            return "Moderate corruption system with some protective mechanisms"
        elif score >= 0.2:
            return "Weak corruption system with limited mutual protection"
        else:
            return "Fragmented corruption with minimal protective mechanisms"
    
    def export_analysis(self, cases: pd.DataFrame, output_path: str, 
                       years_to_analyze: List[int] = None):
        """
        Export comprehensive corruption layer analysis to JSON file.
        
        Parameters:
        -----------
        cases : pd.DataFrame
            Corruption cases dataset
        output_path : str
            Path for output JSON file
        years_to_analyze : List[int]
            Specific years to analyze (default: key historical years)
        """
        if years_to_analyze is None:
            years_to_analyze = [1880, 1912, 1950, 1990, 2000, 2010, 2017, 2020, 2025]
        
        analysis = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_cases': len(cases),
                'year_range': [cases['year'].min(), cases['year'].max()] if not cases.empty else [None, None],
                'layers_analyzed': list(self.layers.keys())
            },
            'layer_definitions': self.layers,
            'temporal_analysis': {},
            'accumulation_index': self.calculate_accumulation_index(cases),
            'layer_interactions': self.analyze_layer_interaction(cases),
            'predictions': self.predict_next_mutation(cases),
            'biofilm_evolution': {}
        }
        
        # Temporal analysis for each year
        for year in years_to_analyze:
            persistence = self.measure_layer_persistence(cases, year)
            biofilm = self.generate_biofilm_score(cases, year, detailed=True)
            
            analysis['temporal_analysis'][year] = {
                'layer_persistence': persistence,
                'biofilm_analysis': biofilm
            }
        
        # Biofilm evolution over time
        for year in range(1900, 2026, 25):
            score = self.generate_biofilm_score(cases, year)
            analysis['biofilm_evolution'][year] = score
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Comprehensive analysis exported to {output_path}")
        
        return analysis