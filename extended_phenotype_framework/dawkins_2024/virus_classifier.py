"""
Virus Classifier - Implementation of Dawkins 2024 Verticovirus/Horizontovirus Concept

Classifies legal norms as verticoviruses (intergenerational, aligned with future)
or horizontoviruses (lateral, short-term oriented) based on their "shared output"
to the future and transmission patterns.

Based on Richard Dawkins' 2024 concepts applied to legal evolutionary theory
by Ignacio Adrián Lerer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from datetime import datetime, timedelta
import numpy as np
import uuid
import json
from collections import defaultdict


class VirusType(Enum):
    """Types of legal viruses based on transmission and temporal alignment"""
    VERTICOVIRUS = "verticovirus"      # Intergenerational, future-aligned
    HORIZONTOVIRUS = "horizontovirus"  # Lateral, present-focused
    HYBRID = "hybrid"                  # Mixed characteristics
    DORMANT = "dormant"               # Inactive but present
    EXTINCT = "extinct"               # No longer active


class TransmissionPattern(Enum):
    """Patterns of legal norm transmission"""
    INTERGENERATIONAL = "intergenerational"  # Parent to child institutions
    LATERAL_PEER = "lateral_peer"            # Between peer institutions
    HIERARCHICAL_DOWN = "hierarchical_down" # Top-down authority
    HIERARCHICAL_UP = "hierarchical_up"     # Bottom-up pressure
    NETWORK_VIRAL = "network_viral"         # Viral spread through networks
    CULTURAL_DIFFUSION = "cultural_diffusion" # Cultural/professional spread


class TemporalAlignment(Enum):
    """Temporal alignment of norm benefits"""
    FUTURE_FOCUSED = "future_focused"    # Benefits accrue in future
    PRESENT_FOCUSED = "present_focused"  # Benefits accrue immediately
    PAST_ANCHORED = "past_anchored"     # Maintains past benefits
    BALANCED_TEMPORAL = "balanced_temporal" # Balanced across time


@dataclass
class TransmissionData:
    """Data about how a legal norm transmits"""
    transmission_pattern: TransmissionPattern
    transmission_speed: float         # Rate of transmission (0-1)
    transmission_fidelity: float      # Accuracy of transmission (0-1)
    network_reach: int               # Number of nodes reached
    adoption_resistance: float       # Resistance to adoption (0-1)
    mutation_rate: float            # Rate of mutation during transmission (0-1)
    
    # Temporal characteristics
    transmission_timeline: List[Tuple[datetime, str]]  # (date, adopting_entity)
    generation_gaps: List[float]     # Time between generational transmissions
    lateral_spread_rate: float       # Rate of lateral spreading
    
    # Network analysis
    transmission_network: Dict[str, List[str]]  # Network of transmission relationships
    influence_hubs: List[str]        # Key nodes in transmission network
    transmission_barriers: List[str] # Barriers encountered


@dataclass
class FutureAlignmentAnalysis:
    """Analysis of norm's alignment with future shared outcomes"""
    score: float                     # Overall future alignment score (0-1)
    intergenerational_benefit: float # Benefit to future generations (0-1)
    sustainability_index: float      # Environmental/resource sustainability (0-1)
    adaptability_score: float       # Ability to adapt to future conditions (0-1)
    
    # Specific alignment factors
    climate_alignment: float         # Alignment with climate goals
    technological_alignment: float   # Alignment with tech advancement
    social_justice_alignment: float  # Alignment with social progress
    economic_sustainability: float   # Economic sustainability
    
    # Beneficiary analysis
    current_beneficiaries: Dict[str, float]  # Who benefits now
    future_beneficiaries: Dict[str, float]   # Who will benefit in future
    cost_bearers: Dict[str, float]          # Who bears costs
    
    # Temporal distribution
    benefit_timeline: List[Tuple[int, float]]  # (years_from_now, benefit_level)
    cost_timeline: List[Tuple[int, float]]     # (years_from_now, cost_level)


@dataclass
class ViralClassification:
    """Complete viral classification of a legal norm"""
    virus_type: VirusType
    confidence_score: float          # Confidence in classification (0-1)
    
    # Core analysis components
    future_alignment_score: float
    transmission_classification: TransmissionData
    temporal_beneficiary_analysis: Dict[str, Any]
    
    # Survival predictions
    survival_probability: float      # Probability of surviving next decade (0-1)
    expected_lifespan: int          # Expected lifespan in years
    critical_vulnerabilities: List[str] # Main threats to survival
    
    # Competitive analysis
    competitive_advantage: float     # Advantage over competing norms (0-1)
    niche_specialization: float     # Degree of specialization (0-1)
    ecosystem_fit: float           # Fit with current legal ecosystem (0-1)
    
    # Evolution potential
    mutation_potential: float       # Potential for beneficial mutation (0-1)
    adaptation_capacity: float      # Capacity to adapt to pressure (0-1)
    
    # Detailed breakdowns
    classification_factors: Dict[str, float]
    uncertainty_factors: List[str]


@dataclass
class ViralCompetitionModel:
    """Model of competition between legal viruses"""
    classifications: List[ViralClassification]
    compatibility_analysis: np.ndarray        # Compatibility matrix
    coexistence_prediction: 'CoexistenceSimulation'
    dominant_virus_prediction: Optional[ViralClassification]
    ecosystem_stability: float               # Overall ecosystem stability (0-1)
    
    # Competition dynamics
    competitive_exclusion_pairs: List[Tuple[str, str]]  # Mutually exclusive pairs
    symbiotic_relationships: List[Tuple[str, str]]      # Mutually beneficial pairs
    neutral_interactions: List[Tuple[str, str]]         # No significant interaction
    
    # Evolution predictions
    convergent_evolution_potential: List[str]  # Norms likely to converge
    divergent_evolution_potential: List[str]   # Norms likely to diverge
    hybrid_formation_potential: List[Tuple[str, str]]  # Potential hybrids


class ViralClassificationEngine:
    """
    Main engine for classifying legal norms as viruses
    
    Implements Dawkins 2024 concepts of verticovirus vs horizontovirus
    based on transmission patterns and shared output to future.
    """
    
    def __init__(self):
        self.classification_history: List[ViralClassification] = []
        self.transmission_networks: Dict[str, TransmissionData] = {}
        self.future_alignment_cache: Dict[str, FutureAlignmentAnalysis] = {}
        
        # Classification parameters
        self.verticovirus_threshold = 0.7    # Future alignment threshold
        self.horizontovirus_threshold = 0.3  # Below this is horizontal
        self.temporal_window = 20           # Years for future projection
        
        # Analysis weights
        self.alignment_weights = {
            'intergenerational_benefit': 0.3,
            'sustainability_index': 0.25,
            'adaptability_score': 0.2,
            'climate_alignment': 0.1,
            'technological_alignment': 0.1,
            'social_justice_alignment': 0.05
        }
    
    def classify_legal_norm(
        self, 
        norm: 'LegalNorm',
        transmission_data: TransmissionData
    ) -> ViralClassification:
        """
        Classify legal norm as verticovirus/horizontovirus
        
        Args:
            norm: Legal norm to classify
            transmission_data: Data about norm transmission
            
        Returns:
            Complete viral classification
        """
        # Analyze future alignment
        future_alignment = self._analyze_future_alignment(norm)
        
        # Classify transmission pattern
        transmission_classification = self._classify_transmission_pattern(transmission_data)
        
        # Analyze temporal beneficiaries
        beneficiary_analysis = self._analyze_temporal_beneficiaries(norm)
        
        # Make primary classification
        virus_type, confidence = self._classify_virus_type(
            future_alignment, 
            transmission_classification,
            beneficiary_analysis
        )
        
        # Predict survival and competition
        survival_analysis = self._predict_viral_survival(
            norm, virus_type, future_alignment, transmission_classification
        )
        
        # Calculate competitive factors
        competitive_analysis = self._analyze_competitive_factors(
            norm, virus_type, future_alignment
        )
        
        classification = ViralClassification(
            virus_type=virus_type,
            confidence_score=confidence,
            future_alignment_score=future_alignment.score,
            transmission_classification=transmission_data,
            temporal_beneficiary_analysis=beneficiary_analysis,
            survival_probability=survival_analysis['probability'],
            expected_lifespan=survival_analysis['lifespan'],
            critical_vulnerabilities=survival_analysis['vulnerabilities'],
            competitive_advantage=competitive_analysis['advantage'],
            niche_specialization=competitive_analysis['specialization'],
            ecosystem_fit=competitive_analysis['ecosystem_fit'],
            mutation_potential=self._calculate_mutation_potential(norm, virus_type),
            adaptation_capacity=self._calculate_adaptation_capacity(norm, transmission_data),
            classification_factors=self._extract_classification_factors(
                future_alignment, transmission_classification, beneficiary_analysis
            ),
            uncertainty_factors=self._identify_uncertainty_factors(norm, transmission_data)
        )
        
        # Store for future analysis
        self.classification_history.append(classification)
        
        return classification
    
    def classify_legal_norm_comprehensive(
        self, 
        norm: 'LegalNorm',
        transmission_history: 'TransmissionHistory',
        stakeholder_analysis: 'StakeholderAnalysis'
    ) -> ViralClassification:
        """
        Comprehensive classification with full context analysis
        
        Args:
            norm: Legal norm to classify
            transmission_history: Historical transmission data
            stakeholder_analysis: Analysis of affected stakeholders
            
        Returns:
            Comprehensive viral classification
        """
        # Enhanced transmission data from history
        transmission_data = self._build_transmission_data_from_history(transmission_history)
        
        # Enhanced future alignment incorporating stakeholder analysis
        future_alignment = self._analyze_comprehensive_future_alignment(
            norm, stakeholder_analysis
        )
        
        # Enhanced beneficiary analysis
        beneficiary_analysis = self._analyze_comprehensive_beneficiaries(
            norm, stakeholder_analysis, transmission_history
        )
        
        return self.classify_legal_norm(norm, transmission_data)
    
    def model_viral_competition(
        self, 
        competing_norms: List['LegalNorm'],
        legal_ecosystem: 'LegalEcosystem'
    ) -> ViralCompetitionModel:
        """
        Model competition between multiple legal viruses
        
        Args:
            competing_norms: List of competing legal norms
            legal_ecosystem: Legal ecosystem context
            
        Returns:
            Complete competition model
        """
        # Classify all norms
        viral_classifications = []
        for norm in competing_norms:
            transmission_data = self._extract_transmission_data(norm, legal_ecosystem)
            classification = self.classify_legal_norm(norm, transmission_data)
            viral_classifications.append(classification)
        
        # Build compatibility matrix
        compatibility_matrix = self._build_compatibility_matrix(viral_classifications)
        
        # Simulate coexistence
        coexistence_simulation = self._simulate_coexistence(
            viral_classifications,
            compatibility_matrix,
            legal_ecosystem
        )
        
        # Predict dominant virus
        dominant_virus = self._predict_dominant_virus(
            viral_classifications, 
            coexistence_simulation
        )
        
        # Assess ecosystem stability
        ecosystem_stability = self._assess_ecosystem_stability(coexistence_simulation)
        
        # Analyze competition dynamics
        competition_dynamics = self._analyze_competition_dynamics(
            viral_classifications, 
            compatibility_matrix
        )
        
        return ViralCompetitionModel(
            classifications=viral_classifications,
            compatibility_analysis=compatibility_matrix,
            coexistence_prediction=coexistence_simulation,
            dominant_virus_prediction=dominant_virus,
            ecosystem_stability=ecosystem_stability,
            competitive_exclusion_pairs=competition_dynamics['exclusion_pairs'],
            symbiotic_relationships=competition_dynamics['symbiotic_pairs'],
            neutral_interactions=competition_dynamics['neutral_pairs'],
            convergent_evolution_potential=self._predict_convergent_evolution(viral_classifications),
            divergent_evolution_potential=self._predict_divergent_evolution(viral_classifications),
            hybrid_formation_potential=self._predict_hybrid_formation(viral_classifications)
        )
    
    def predict_viral_survival(
        self,
        norm: 'LegalNorm',
        virus_type: VirusType,
        future_alignment: FutureAlignmentAnalysis,
        transmission_pattern: TransmissionData
    ) -> Dict[str, Any]:
        """
        Predict survival characteristics of viral norm
        
        Args:
            norm: Legal norm
            virus_type: Classified virus type
            future_alignment: Future alignment analysis
            transmission_pattern: Transmission pattern data
            
        Returns:
            Survival prediction data
        """
        # Base survival probability from virus type
        type_survival_factors = {
            VirusType.VERTICOVIRUS: 0.8,      # High survival
            VirusType.HORIZONTOVIRUS: 0.3,    # Low survival
            VirusType.HYBRID: 0.6,            # Medium survival
            VirusType.DORMANT: 0.4,           # Low-medium survival
            VirusType.EXTINCT: 0.0            # No survival
        }
        
        base_survival = type_survival_factors[virus_type]
        
        # Adjust based on future alignment
        alignment_factor = future_alignment.score ** 0.5  # Diminishing returns
        
        # Adjust based on transmission characteristics
        transmission_factor = (
            transmission_pattern.transmission_fidelity * 0.4 +
            (1 - transmission_pattern.adoption_resistance) * 0.3 +
            transmission_pattern.transmission_speed * 0.3
        )
        
        # Calculate final survival probability
        survival_probability = base_survival * alignment_factor * transmission_factor
        
        # Estimate lifespan
        if virus_type == VirusType.VERTICOVIRUS:
            base_lifespan = 50  # Long-lived
        elif virus_type == VirusType.HORIZONTOVIRUS:
            base_lifespan = 10  # Short-lived
        else:
            base_lifespan = 25  # Medium-lived
        
        expected_lifespan = int(base_lifespan * alignment_factor * transmission_factor)
        
        # Identify vulnerabilities
        vulnerabilities = self._identify_viral_vulnerabilities(
            norm, virus_type, future_alignment, transmission_pattern
        )
        
        return {
            'probability': min(1.0, max(0.0, survival_probability)),
            'lifespan': max(1, expected_lifespan),
            'vulnerabilities': vulnerabilities
        }
    
    # Private implementation methods
    
    def _analyze_future_alignment(self, norm: 'LegalNorm') -> FutureAlignmentAnalysis:
        """Analyze norm's alignment with future shared outcomes"""
        # Intergenerational benefit analysis
        intergenerational_benefit = self._calculate_intergenerational_benefit(norm)
        
        # Sustainability analysis
        sustainability_index = self._calculate_sustainability_index(norm)
        
        # Adaptability analysis
        adaptability_score = self._calculate_adaptability_score(norm)
        
        # Specific alignment factors
        climate_alignment = self._calculate_climate_alignment(norm)
        technological_alignment = self._calculate_technological_alignment(norm)
        social_justice_alignment = self._calculate_social_justice_alignment(norm)
        economic_sustainability = self._calculate_economic_sustainability(norm)
        
        # Calculate overall score
        overall_score = (
            intergenerational_benefit * self.alignment_weights['intergenerational_benefit'] +
            sustainability_index * self.alignment_weights['sustainability_index'] +
            adaptability_score * self.alignment_weights['adaptability_score'] +
            climate_alignment * self.alignment_weights['climate_alignment'] +
            technological_alignment * self.alignment_weights['technological_alignment'] +
            social_justice_alignment * self.alignment_weights['social_justice_alignment']
        )
        
        # Analyze beneficiaries
        current_beneficiaries = self._analyze_current_beneficiaries(norm)
        future_beneficiaries = self._analyze_future_beneficiaries(norm)
        cost_bearers = self._analyze_cost_bearers(norm)
        
        # Create benefit/cost timelines
        benefit_timeline = self._project_benefit_timeline(norm, self.temporal_window)
        cost_timeline = self._project_cost_timeline(norm, self.temporal_window)
        
        return FutureAlignmentAnalysis(
            score=overall_score,
            intergenerational_benefit=intergenerational_benefit,
            sustainability_index=sustainability_index,
            adaptability_score=adaptability_score,
            climate_alignment=climate_alignment,
            technological_alignment=technological_alignment,
            social_justice_alignment=social_justice_alignment,
            economic_sustainability=economic_sustainability,
            current_beneficiaries=current_beneficiaries,
            future_beneficiaries=future_beneficiaries,
            cost_bearers=cost_bearers,
            benefit_timeline=benefit_timeline,
            cost_timeline=cost_timeline
        )
    
    def _classify_transmission_pattern(self, transmission_data: TransmissionData) -> TransmissionPattern:
        """Classify the primary transmission pattern"""
        # Analyze generational gaps
        if transmission_data.generation_gaps and np.mean(transmission_data.generation_gaps) > 10:
            return TransmissionPattern.INTERGENERATIONAL
        
        # Analyze network structure
        if transmission_data.lateral_spread_rate > 0.7:
            return TransmissionPattern.NETWORK_VIRAL
        
        # Analyze timeline patterns
        if len(transmission_data.transmission_timeline) > 1:
            time_diffs = [
                (t2[0] - t1[0]).days for t1, t2 in 
                zip(transmission_data.transmission_timeline[:-1], transmission_data.transmission_timeline[1:])
            ]
            avg_time_diff = np.mean(time_diffs)
            
            if avg_time_diff < 30:  # Less than a month
                return TransmissionPattern.LATERAL_PEER
            elif avg_time_diff > 365 * 5:  # More than 5 years
                return TransmissionPattern.INTERGENERATIONAL
        
        # Default classification
        return transmission_data.transmission_pattern
    
    def _analyze_temporal_beneficiaries(self, norm: 'LegalNorm') -> Dict[str, Any]:
        """Analyze who benefits when from the norm"""
        analysis = {
            'immediate_beneficiaries': self._identify_immediate_beneficiaries(norm),
            'long_term_beneficiaries': self._identify_long_term_beneficiaries(norm),
            'cost_distribution': self._analyze_cost_distribution(norm),
            'temporal_pattern': self._classify_temporal_benefit_pattern(norm),
            'includes_future_generations': self._includes_future_generations(norm),
            'immediate_benefits_only': self._immediate_benefits_only(norm)
        }
        return analysis
    
    def _classify_virus_type(
        self,
        future_alignment: FutureAlignmentAnalysis,
        transmission_classification: TransmissionPattern,
        beneficiary_analysis: Dict[str, Any]
    ) -> Tuple[VirusType, float]:
        """Main classification logic for virus type"""
        
        # Check for verticovirus characteristics
        verticovirus_score = 0.0
        
        # Future alignment factor (40% weight)
        if future_alignment.score > self.verticovirus_threshold:
            verticovirus_score += 0.4
        
        # Transmission pattern factor (30% weight)
        if transmission_classification == TransmissionPattern.INTERGENERATIONAL:
            verticovirus_score += 0.3
        elif transmission_classification in [TransmissionPattern.HIERARCHICAL_DOWN, 
                                          TransmissionPattern.CULTURAL_DIFFUSION]:
            verticovirus_score += 0.15
        
        # Beneficiary analysis factor (30% weight)
        if beneficiary_analysis['includes_future_generations']:
            verticovirus_score += 0.2
        if not beneficiary_analysis['immediate_benefits_only']:
            verticovirus_score += 0.1
        
        # Classification with confidence
        if verticovirus_score >= 0.7:
            return VirusType.VERTICOVIRUS, verticovirus_score
        elif verticovirus_score <= 0.3:
            return VirusType.HORIZONTOVIRUS, 1 - verticovirus_score
        else:
            return VirusType.HYBRID, 1 - abs(verticovirus_score - 0.5) * 2
    
    def _build_compatibility_matrix(self, classifications: List[ViralClassification]) -> np.ndarray:
        """Build compatibility matrix between viral classifications"""
        n = len(classifications)
        matrix = np.zeros((n, n))
        
        for i, class_i in enumerate(classifications):
            for j, class_j in enumerate(classifications):
                if i != j:
                    compatibility = self._calculate_viral_compatibility(class_i, class_j)
                    matrix[i, j] = compatibility
                else:
                    matrix[i, j] = 1.0  # Perfect self-compatibility
        
        return matrix
    
    def _calculate_viral_compatibility(
        self, 
        virus1: ViralClassification, 
        virus2: ViralClassification
    ) -> float:
        """Calculate compatibility between two viral norms"""
        # Type compatibility
        type_compatibility = {
            (VirusType.VERTICOVIRUS, VirusType.VERTICOVIRUS): 0.8,
            (VirusType.VERTICOVIRUS, VirusType.HORIZONTOVIRUS): 0.3,
            (VirusType.VERTICOVIRUS, VirusType.HYBRID): 0.6,
            (VirusType.HORIZONTOVIRUS, VirusType.HORIZONTOVIRUS): 0.5,
            (VirusType.HORIZONTOVIRUS, VirusType.HYBRID): 0.4,
            (VirusType.HYBRID, VirusType.HYBRID): 0.7
        }
        
        base_compatibility = type_compatibility.get(
            (virus1.virus_type, virus2.virus_type), 0.5
        )
        
        # Future alignment compatibility
        alignment_diff = abs(virus1.future_alignment_score - virus2.future_alignment_score)
        alignment_compatibility = 1.0 - alignment_diff
        
        # Ecosystem fit compatibility
        ecosystem_compatibility = (virus1.ecosystem_fit + virus2.ecosystem_fit) / 2
        
        # Combined compatibility
        return (base_compatibility * 0.5 + 
                alignment_compatibility * 0.3 + 
                ecosystem_compatibility * 0.2)
    
    # Utility methods for specific calculations
    
    def _calculate_intergenerational_benefit(self, norm: 'LegalNorm') -> float:
        """Calculate benefit to future generations"""
        # Simplified calculation - would implement domain-specific logic
        if hasattr(norm, 'environmental_impact') and norm.environmental_impact > 0:
            return 0.8
        elif hasattr(norm, 'education_impact') and norm.education_impact > 0:
            return 0.9
        elif hasattr(norm, 'infrastructure_impact') and norm.infrastructure_impact > 0:
            return 0.7
        else:
            return 0.3
    
    def _calculate_sustainability_index(self, norm: 'LegalNorm') -> float:
        """Calculate sustainability index"""
        # Simplified - would implement comprehensive sustainability analysis
        sustainability_factors = []
        
        if hasattr(norm, 'resource_consumption'):
            sustainability_factors.append(1.0 - norm.resource_consumption)
        if hasattr(norm, 'renewable_energy_support'):
            sustainability_factors.append(norm.renewable_energy_support)
        if hasattr(norm, 'circular_economy_support'):
            sustainability_factors.append(norm.circular_economy_support)
        
        return np.mean(sustainability_factors) if sustainability_factors else 0.5
    
    def _calculate_adaptability_score(self, norm: 'LegalNorm') -> float:
        """Calculate adaptability to future conditions"""
        # Simplified - would implement detailed adaptability analysis
        if hasattr(norm, 'flexibility_mechanisms'):
            return norm.flexibility_mechanisms
        else:
            return 0.5
    
    def _includes_future_generations(self, norm: 'LegalNorm') -> bool:
        """Check if norm explicitly considers future generations"""
        # Simplified check - would implement text analysis
        return hasattr(norm, 'intergenerational_clause') and norm.intergenerational_clause
    
    def _immediate_benefits_only(self, norm: 'LegalNorm') -> bool:
        """Check if norm only provides immediate benefits"""
        # Simplified check - would implement temporal benefit analysis
        return hasattr(norm, 'immediate_only') and norm.immediate_only
    
    def __repr__(self) -> str:
        return f"ViralClassificationEngine(classifications={len(self.classification_history)})"