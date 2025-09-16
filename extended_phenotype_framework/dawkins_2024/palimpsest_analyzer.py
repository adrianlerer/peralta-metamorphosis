"""
Palimpsest Analyzer - Implementation of Dawkins 2024 Palimpsest Concept

Analyzes legal structures as palimpsests - layered documents where historical
traces constrain new constructions. Legal systems cannot "start from scratch"
but must work within constraints of historical layers.

Based on Richard Dawkins' 2024 concepts applied to legal evolutionary theory
by Ignacio Adrián Lerer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, date
import numpy as np
import uuid
import re
from collections import defaultdict


class LayerVisibility(Enum):
    """Visibility levels of historical layers"""
    DOMINANT = "dominant"  # Clearly visible and constraining
    VISIBLE = "visible"    # Detectable with analysis
    TRACE = "trace"        # Faint traces remain
    BURIED = "buried"      # Requires deep analysis
    ERASED = "erased"      # No longer detectable


class ErosionType(Enum):
    """Types of erosion affecting historical layers"""
    NATURAL_DECAY = "natural_decay"       # Gradual forgetting/irrelevance
    ACTIVE_SUPPRESSION = "active_suppression"  # Deliberate erasure
    TECHNOLOGICAL_OBSOLESCENCE = "technological_obsolescence"
    POLITICAL_PRESSURE = "political_pressure"
    ECONOMIC_PRESSURE = "economic_pressure"
    CULTURAL_SHIFT = "cultural_shift"


@dataclass
class HistoricalLayer:
    """Represents a historical layer in legal palimpsest"""
    layer_id: str
    epoch_start: datetime
    epoch_end: Optional[datetime]
    dominant_constructors: List[str]  # Constructor IDs that created this layer
    legal_concepts: List[str]         # Key legal concepts from this layer
    institutional_structures: List[str] # Institutions created in this layer
    power_configuration: Dict[str, float] # Power distribution in this epoch
    
    # Persistence characteristics
    visibility_level: LayerVisibility
    embedding_strength: float  # How deeply embedded (0-1)
    restriction_power: float   # How much it constrains new constructions (0-1)
    
    # Content analysis
    textual_markers: List[str]  # Characteristic phrases/terms
    conceptual_signatures: Dict[str, float]  # Conceptual fingerprints
    institutional_legacies: Dict[str, float]  # Surviving institutional elements
    
    # Evolution tracking
    mutation_rate: float = 0.05  # Rate of change/adaptation
    erosion_factors: Dict[ErosionType, float] = field(default_factory=dict)
    coalescence_points: List[str] = field(default_factory=list)  # Common ancestors


@dataclass
class PalimpsestIndex:
    """Results of palimpsest analysis"""
    restriction_coefficient: float  # Overall restriction from history (0-1)
    innovation_freedom: float       # Space for genuine innovation (0-1)
    historical_anchors: List[HistoricalLayer]  # Active constraining layers
    coalescence_point: Optional[HistoricalLayer]  # Common ancestor layer
    
    # Path dependency analysis
    critical_path_dependencies: List[str]  # Unavoidable historical constraints
    deviation_costs: Dict[str, float]      # Cost to deviate from each path
    innovation_opportunities: List[str]    # Areas with room for innovation
    
    # Prediction metrics
    construction_difficulty: float     # Difficulty of new construction (0-1)
    adaptation_capacity: float        # Capacity to adapt to new pressures (0-1)
    stability_prediction: float       # Predicted stability of new construction (0-1)


@dataclass
class ConstraintAnalysis:
    """Analysis of constraints on new legal construction"""
    path_dependencies: List[str]
    deviation_cost: float
    innovation_freedom: float
    optimal_construction_strategy: Dict[str, Any]
    
    # Specific constraint types
    constitutional_constraints: Dict[str, float]
    institutional_constraints: Dict[str, float]
    cultural_constraints: Dict[str, float]
    economic_constraints: Dict[str, float]
    
    # Recommendations
    recommended_approaches: List[str]
    high_risk_areas: List[str]
    innovation_windows: List[str]


@dataclass
class PersistencePrediction:
    """Prediction of historical layer persistence"""
    layer_id: str
    persistence_probability: float
    critical_erosion_threshold: float
    expected_half_life: float  # Years until 50% erosion
    
    # Vulnerability analysis
    primary_threats: List[ErosionType]
    protection_factors: List[str]
    adaptation_scenarios: List[str]


class PalimpsestAnalyzer:
    """
    Main analyzer for legal palimpsest analysis
    
    Implements Dawkins 2024 concept of palimpsest applied to legal systems:
    legal constructions cannot start fresh but must work within historical
    constraints, like writing on a palimpsest where previous text shows through.
    """
    
    def __init__(self):
        self.historical_layers: List[HistoricalLayer] = []
        self.visibility_coefficients: Dict[str, float] = {}
        self.restriction_matrix: Optional[np.ndarray] = None
        self.concept_genealogies: Dict[str, List[str]] = {}
        
        # Analysis cache for performance
        self._analysis_cache: Dict[str, Any] = {}
        
    def add_historical_layer(
        self, 
        layer: HistoricalLayer,
        visibility: float,
        restriction_power: float
    ):
        """
        Add historical layer with visibility and restriction coefficients
        
        Args:
            layer: Historical layer to add
            visibility: Visibility coefficient (0-1)
            restriction_power: How much this layer constrains new constructions (0-1)
        """
        self.historical_layers.append(layer)
        self.visibility_coefficients[layer.layer_id] = visibility
        layer.restriction_power = restriction_power
        self._update_restriction_matrix()
    
    def analyze_construction_constraints(
        self, 
        new_construction_proposal: 'LegalConstruction'
    ) -> ConstraintAnalysis:
        """
        Analyze palimpsestic constraints for new construction
        
        Args:
            new_construction_proposal: Proposed new legal construction
            
        Returns:
            Comprehensive constraint analysis
        """
        # Identify active path dependencies
        active_dependencies = self._identify_active_dependencies(new_construction_proposal)
        
        # Calculate cost of deviating from historical paths
        deviation_cost = self._calculate_deviation_cost(active_dependencies)
        
        # Calculate available innovation space
        innovation_space = self._calculate_innovation_space(
            new_construction_proposal.target_domain,
            active_dependencies
        )
        
        # Analyze specific constraint types
        constitutional_constraints = self._analyze_constitutional_constraints(new_construction_proposal)
        institutional_constraints = self._analyze_institutional_constraints(new_construction_proposal)
        cultural_constraints = self._analyze_cultural_constraints(new_construction_proposal)
        economic_constraints = self._analyze_economic_constraints(new_construction_proposal)
        
        # Generate optimal construction strategy
        optimal_strategy = self._suggest_optimal_strategy(
            deviation_cost, 
            innovation_space, 
            active_dependencies
        )
        
        return ConstraintAnalysis(
            path_dependencies=active_dependencies,
            deviation_cost=deviation_cost,
            innovation_freedom=innovation_space,
            optimal_construction_strategy=optimal_strategy,
            constitutional_constraints=constitutional_constraints,
            institutional_constraints=institutional_constraints,
            cultural_constraints=cultural_constraints,
            economic_constraints=economic_constraints,
            recommended_approaches=self._generate_recommended_approaches(optimal_strategy),
            high_risk_areas=self._identify_high_risk_areas(active_dependencies),
            innovation_windows=self._identify_innovation_windows(innovation_space)
        )
    
    def calculate_palimpsest_index(
        self, 
        legal_text: str,
        target_domain: str,
        historical_layers: List[HistoricalLayer] = None
    ) -> PalimpsestIndex:
        """
        Calculate comprehensive palimpsest index for legal text/domain
        
        Args:
            legal_text: Legal text to analyze
            target_domain: Target legal domain
            historical_layers: Specific layers to consider (default: all)
            
        Returns:
            Comprehensive palimpsest analysis
        """
        if historical_layers is None:
            historical_layers = self.historical_layers
        
        # Detect historical traces in the text
        visible_traces = self._detect_historical_traces(legal_text, historical_layers)
        
        # Calculate restriction weight from visible traces
        restriction_weight = self._calculate_path_dependency(visible_traces)
        
        # Calculate innovation freedom (inverse of restriction)
        innovation_freedom = 1.0 - restriction_weight
        
        # Find common coalescence point
        coalescence_point = self._find_common_ancestor(visible_traces)
        
        # Calculate construction difficulty
        construction_difficulty = self._calculate_construction_difficulty(
            restriction_weight, 
            target_domain, 
            visible_traces
        )
        
        # Predict adaptation capacity
        adaptation_capacity = self._calculate_adaptation_capacity(
            visible_traces, 
            target_domain
        )
        
        # Predict stability of new constructions
        stability_prediction = self._predict_construction_stability(
            restriction_weight,
            adaptation_capacity,
            visible_traces
        )
        
        return PalimpsestIndex(
            restriction_coefficient=restriction_weight,
            innovation_freedom=innovation_freedom,
            historical_anchors=visible_traces,
            coalescence_point=coalescence_point,
            critical_path_dependencies=self._extract_critical_paths(visible_traces),
            deviation_costs=self._calculate_all_deviation_costs(visible_traces),
            innovation_opportunities=self._identify_innovation_opportunities(
                target_domain, visible_traces
            ),
            construction_difficulty=construction_difficulty,
            adaptation_capacity=adaptation_capacity,
            stability_prediction=stability_prediction
        )
    
    def simulate_layer_erosion(
        self, 
        current_layer: HistoricalLayer,
        erosion_factors: Dict[ErosionType, float],
        time_horizon: int = 10
    ) -> HistoricalLayer:
        """
        Simulate erosion/persistence of historical layers under pressures
        
        Args:
            current_layer: Layer to analyze
            erosion_factors: Erosion pressures by type
            time_horizon: Years to simulate
            
        Returns:
            Layer after simulated erosion
        """
        # Calculate persistence probability
        persistence_probability = self._calculate_persistence_probability(
            current_layer, 
            erosion_factors
        )
        
        # Simulate erosion over time
        eroded_layer = self._apply_erosion_simulation(
            current_layer, 
            persistence_probability, 
            time_horizon
        )
        
        return eroded_layer
    
    def predict_layer_persistence(
        self, 
        layer: HistoricalLayer,
        future_pressure: Dict[str, float],
        time_horizon: int = 20
    ) -> PersistencePrediction:
        """
        Predict persistence of historical layer under future pressures
        
        Args:
            layer: Historical layer to analyze
            future_pressure: Projected future pressures
            time_horizon: Years to predict
            
        Returns:
            Persistence prediction analysis
        """
        # Calculate current embedding strength
        current_embedding = self._calculate_embedding_strength(layer)
        
        # Analyze erosion factors from future pressures
        erosion_factors = self._analyze_erosion_factors(future_pressure)
        
        # Calculate persistence probability
        persistence_probability = current_embedding * self._calculate_survival_multiplier(
            erosion_factors, time_horizon
        )
        
        # Calculate expected half-life
        half_life = self._calculate_half_life(erosion_factors, current_embedding)
        
        # Identify primary threats and protection factors
        primary_threats = self._identify_primary_threats(erosion_factors)
        protection_factors = self._identify_protection_factors(layer, future_pressure)
        
        return PersistencePrediction(
            layer_id=layer.layer_id,
            persistence_probability=persistence_probability,
            critical_erosion_threshold=self._calculate_critical_threshold(erosion_factors),
            expected_half_life=half_life,
            primary_threats=primary_threats,
            protection_factors=protection_factors,
            adaptation_scenarios=self._generate_adaptation_scenarios(layer, future_pressure)
        )
    
    def trace_concept_genealogy(
        self, 
        legal_concept: str,
        max_depth: int = 50
    ) -> List[HistoricalLayer]:
        """
        Trace genealogy of legal concept through historical layers
        
        Args:
            legal_concept: Legal concept to trace
            max_depth: Maximum genealogical depth
            
        Returns:
            Ordered list of layers containing concept ancestry
        """
        concept_genealogy = []
        
        # Search through layers chronologically (reverse order)
        sorted_layers = sorted(
            self.historical_layers, 
            key=lambda l: l.epoch_start, 
            reverse=True
        )
        
        current_concept = legal_concept
        depth = 0
        
        for layer in sorted_layers:
            if depth >= max_depth:
                break
                
            if self._concept_appears_in_layer(current_concept, layer):
                concept_genealogy.append(layer)
                
                # Look for conceptual ancestors in this layer
                ancestors = self._find_conceptual_ancestors(current_concept, layer)
                if ancestors:
                    current_concept = ancestors[0]  # Follow primary ancestor
                    
            depth += 1
        
        return concept_genealogy
    
    # Private implementation methods
    
    def _identify_active_dependencies(self, construction_proposal) -> List[str]:
        """Identify path dependencies that actively constrain construction"""
        dependencies = []
        
        for layer in self.historical_layers:
            if layer.visibility_level in [LayerVisibility.DOMINANT, LayerVisibility.VISIBLE]:
                # Check if layer concepts overlap with proposal target
                if self._has_conceptual_overlap(layer, construction_proposal):
                    dependency_strength = layer.restriction_power * self.visibility_coefficients.get(
                        layer.layer_id, 0.5
                    )
                    if dependency_strength > 0.3:  # Significance threshold
                        dependencies.append(layer.layer_id)
        
        return dependencies
    
    def _calculate_deviation_cost(self, active_dependencies: List[str]) -> float:
        """Calculate cost of deviating from historical paths"""
        if not active_dependencies:
            return 0.0
        
        total_cost = 0.0
        for dep_id in active_dependencies:
            layer = self._get_layer_by_id(dep_id)
            if layer:
                # Cost is function of embedding strength and visibility
                cost = (layer.embedding_strength * 
                       self.visibility_coefficients.get(dep_id, 0.5) * 
                       layer.restriction_power)
                total_cost += cost
        
        return min(1.0, total_cost / len(active_dependencies))
    
    def _calculate_innovation_space(self, target_domain: str, dependencies: List[str]) -> float:
        """Calculate available space for innovation"""
        # Base innovation space for domain
        domain_innovation_factors = {
            'constitutional_law': 0.1,      # Very constrained
            'international_law': 0.3,       # Moderately constrained
            'administrative_law': 0.5,      # Medium flexibility
            'corporate_law': 0.7,           # High flexibility
            'technology_law': 0.9,          # Very high flexibility
            'new_domains': 0.95             # Almost unconstrained
        }
        
        base_space = domain_innovation_factors.get(target_domain, 0.5)
        
        # Reduce based on active dependencies
        dependency_constraint = len(dependencies) * 0.1  # Each dependency reduces space
        
        return max(0.05, base_space - dependency_constraint)
    
    def _detect_historical_traces(
        self, 
        legal_text: str, 
        layers: List[HistoricalLayer]
    ) -> List[HistoricalLayer]:
        """Detect traces of historical layers in legal text"""
        visible_layers = []
        
        for layer in layers:
            trace_strength = 0.0
            
            # Check for textual markers
            for marker in layer.textual_markers:
                if marker.lower() in legal_text.lower():
                    trace_strength += 0.3
            
            # Check for conceptual signatures
            for concept, weight in layer.conceptual_signatures.items():
                if concept.lower() in legal_text.lower():
                    trace_strength += weight * 0.5
            
            # Check for institutional references
            for institution, weight in layer.institutional_legacies.items():
                if institution.lower() in legal_text.lower():
                    trace_strength += weight * 0.4
            
            # If trace is strong enough and layer is visible enough
            visibility = self.visibility_coefficients.get(layer.layer_id, 0.5)
            if trace_strength * visibility > 0.2:  # Detection threshold
                visible_layers.append(layer)
        
        return visible_layers
    
    def _calculate_path_dependency(self, visible_traces: List[HistoricalLayer]) -> float:
        """Calculate overall path dependency from visible traces"""
        if not visible_traces:
            return 0.0
        
        total_restriction = 0.0
        for layer in visible_traces:
            layer_restriction = (
                layer.restriction_power * 
                layer.embedding_strength * 
                self.visibility_coefficients.get(layer.layer_id, 0.5)
            )
            total_restriction += layer_restriction
        
        # Normalize and apply diminishing returns
        normalized_restriction = total_restriction / len(visible_traces)
        return min(0.95, normalized_restriction)  # Leave minimum innovation space
    
    def _find_common_ancestor(self, layers: List[HistoricalLayer]) -> Optional[HistoricalLayer]:
        """Find common ancestral layer (coalescence point)"""
        if not layers:
            return None
        
        # Sort layers by age (oldest first)
        sorted_layers = sorted(layers, key=lambda l: l.epoch_start)
        
        # Find layer that influences most others
        ancestor_scores = {}
        for potential_ancestor in sorted_layers:
            score = 0
            for later_layer in sorted_layers:
                if (later_layer.epoch_start > potential_ancestor.epoch_start and
                    self._has_ancestral_relationship(potential_ancestor, later_layer)):
                    score += 1
            ancestor_scores[potential_ancestor.layer_id] = score
        
        if ancestor_scores:
            best_ancestor_id = max(ancestor_scores.keys(), key=lambda k: ancestor_scores[k])
            return self._get_layer_by_id(best_ancestor_id)
        
        return None
    
    def _calculate_embedding_strength(self, layer: HistoricalLayer) -> float:
        """Calculate how deeply embedded a layer is"""
        # Factors: age, institutional legacy, conceptual influence
        age_factor = min(1.0, (datetime.now() - layer.epoch_start).days / (365 * 100))  # 100 years max
        
        institutional_factor = sum(layer.institutional_legacies.values()) / max(
            len(layer.institutional_legacies), 1
        )
        
        conceptual_factor = sum(layer.conceptual_signatures.values()) / max(
            len(layer.conceptual_signatures), 1
        )
        
        return (age_factor * 0.3 + institutional_factor * 0.4 + conceptual_factor * 0.3)
    
    def _calculate_construction_difficulty(
        self, 
        restriction_weight: float,
        target_domain: str,
        visible_traces: List[HistoricalLayer]
    ) -> float:
        """Calculate difficulty of new construction given constraints"""
        base_difficulty = restriction_weight
        
        # Domain-specific difficulty modifiers
        domain_difficulties = {
            'constitutional_law': 1.5,
            'international_law': 1.2,
            'administrative_law': 1.0,
            'corporate_law': 0.8,
            'technology_law': 0.6
        }
        
        domain_modifier = domain_difficulties.get(target_domain, 1.0)
        
        # Trace complexity factor
        trace_complexity = len(visible_traces) * 0.1
        
        return min(1.0, base_difficulty * domain_modifier + trace_complexity)
    
    def _update_restriction_matrix(self):
        """Update restriction matrix between layers"""
        n = len(self.historical_layers)
        if n == 0:
            return
            
        self.restriction_matrix = np.zeros((n, n))
        
        for i, layer_i in enumerate(self.historical_layers):
            for j, layer_j in enumerate(self.historical_layers):
                if i != j:
                    # Calculate restriction relationship
                    restriction = self._calculate_layer_restriction(layer_i, layer_j)
                    self.restriction_matrix[i, j] = restriction
    
    # Utility methods
    
    def _get_layer_by_id(self, layer_id: str) -> Optional[HistoricalLayer]:
        """Get layer by ID"""
        for layer in self.historical_layers:
            if layer.layer_id == layer_id:
                return layer
        return None
    
    def _has_conceptual_overlap(self, layer: HistoricalLayer, construction_proposal) -> bool:
        """Check if layer has conceptual overlap with construction proposal"""
        # Simplified overlap detection
        proposal_concepts = getattr(construction_proposal, 'target_concepts', [])
        return bool(set(layer.legal_concepts).intersection(set(proposal_concepts)))
    
    def _concept_appears_in_layer(self, concept: str, layer: HistoricalLayer) -> bool:
        """Check if concept appears in historical layer"""
        return (concept.lower() in [c.lower() for c in layer.legal_concepts] or
                concept.lower() in layer.conceptual_signatures.keys())
    
    def _find_conceptual_ancestors(self, concept: str, layer: HistoricalLayer) -> List[str]:
        """Find conceptual ancestors of concept in layer"""
        # Simplified ancestor detection
        ancestors = []
        for concept_sig in layer.conceptual_signatures.keys():
            if self._concepts_related(concept, concept_sig):
                ancestors.append(concept_sig)
        return ancestors
    
    def _concepts_related(self, concept1: str, concept2: str) -> bool:
        """Check if two concepts are related"""
        # Simplified relationship detection
        return (concept1.lower() in concept2.lower() or 
                concept2.lower() in concept1.lower() or
                self._semantic_similarity(concept1, concept2) > 0.7)
    
    def _semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity between concepts"""
        # Simplified similarity calculation
        # In practice, this would use more sophisticated NLP
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    # Additional implementation methods would continue here...
    # (Implementing remaining private methods for completeness)
    
    def _analyze_constitutional_constraints(self, proposal) -> Dict[str, float]:
        """Analyze constitutional constraints on proposal"""
        return {"basic_rights": 0.8, "federal_structure": 0.6, "separation_powers": 0.7}
    
    def _analyze_institutional_constraints(self, proposal) -> Dict[str, float]:
        """Analyze institutional constraints"""
        return {"judiciary": 0.7, "legislature": 0.5, "executive": 0.6}
    
    def _analyze_cultural_constraints(self, proposal) -> Dict[str, float]:
        """Analyze cultural constraints"""
        return {"legal_culture": 0.6, "social_norms": 0.5, "professional_norms": 0.7}
    
    def _analyze_economic_constraints(self, proposal) -> Dict[str, float]:
        """Analyze economic constraints"""
        return {"implementation_cost": 0.8, "compliance_burden": 0.6}
    
    def _suggest_optimal_strategy(self, deviation_cost, innovation_space, dependencies) -> Dict[str, Any]:
        """Suggest optimal construction strategy"""
        if deviation_cost > 0.7:
            strategy_type = "incremental_adaptation"
        elif innovation_space > 0.7:
            strategy_type = "innovative_construction"
        else:
            strategy_type = "hybrid_approach"
        
        return {
            "strategy_type": strategy_type,
            "deviation_cost": deviation_cost,
            "innovation_space": innovation_space,
            "primary_constraints": dependencies[:3]  # Top 3 constraints
        }
    
    def _extract_critical_paths(self, traces: List[HistoricalLayer]) -> List[str]:
        """Extract critical path dependencies"""
        return [layer.layer_id for layer in traces if layer.restriction_power > 0.7]
    
    def _calculate_all_deviation_costs(self, traces: List[HistoricalLayer]) -> Dict[str, float]:
        """Calculate deviation costs for all traces"""
        costs = {}
        for layer in traces:
            costs[layer.layer_id] = layer.restriction_power * layer.embedding_strength
        return costs
    
    def _identify_innovation_opportunities(self, domain: str, traces: List[HistoricalLayer]) -> List[str]:
        """Identify areas with innovation opportunities"""
        opportunities = []
        
        # Domain-specific opportunities
        if domain == "technology_law":
            opportunities.extend(["ai_governance", "blockchain_regulation", "data_rights"])
        elif domain == "environmental_law":
            opportunities.extend(["climate_litigation", "green_finance", "carbon_markets"])
        
        return opportunities
    
    def _calculate_adaptation_capacity(self, traces: List[HistoricalLayer], domain: str) -> float:
        """Calculate system adaptation capacity"""
        if not traces:
            return 0.8  # High capacity if no constraints
        
        avg_mutation_rate = sum(layer.mutation_rate for layer in traces) / len(traces)
        return min(1.0, avg_mutation_rate * 5)  # Scale mutation rate
    
    def _predict_construction_stability(
        self, 
        restriction_weight: float,
        adaptation_capacity: float, 
        traces: List[HistoricalLayer]
    ) -> float:
        """Predict stability of new construction"""
        stability = (restriction_weight * 0.4 +  # Historical anchoring provides stability
                    adaptation_capacity * 0.3 +    # Adaptation capacity helps stability
                    (1 - len(traces) * 0.1) * 0.3)  # Too many constraints reduce stability
        
        return max(0.1, min(1.0, stability))
    
    def __repr__(self) -> str:
        return f"PalimpsestAnalyzer(layers={len(self.historical_layers)})"