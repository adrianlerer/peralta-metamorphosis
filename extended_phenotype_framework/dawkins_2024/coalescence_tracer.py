"""
Coalescence Tracer - Implementation of Dawkins 2024 Coalescence Theory

Traces legal concepts back to their "coalescence point" - the common ancestor
constructor or historical moment where diverse lineages converge.

Implements gene coalescence theory applied to legal memetics: all legal concepts
can be traced back to common ancestral constructors and power configurations.

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
import networkx as nx
from collections import defaultdict, deque
import heapq


class LineageType(Enum):
    """Types of legal concept lineages"""
    CONSTITUTIONAL = "constitutional"      # Constitutional law lineages
    STATUTORY = "statutory"               # Legislative lineages
    JUDICIAL = "judicial"                 # Case law lineages
    REGULATORY = "regulatory"             # Administrative law lineages
    CONTRACTUAL = "contractual"           # Private law lineages
    CUSTOMARY = "customary"              # Customary law lineages
    INTERNATIONAL = "international"       # International law lineages
    HYBRID = "hybrid"                    # Mixed lineages


class CoalescenceType(Enum):
    """Types of coalescence points"""
    CONSTRUCTOR_ORIGIN = "constructor_origin"      # Original constructor
    HISTORICAL_EVENT = "historical_event"         # Major historical event
    INSTITUTIONAL_MOMENT = "institutional_moment" # Institution creation
    CRISIS_CONVERGENCE = "crisis_convergence"     # Crisis-driven convergence
    CULTURAL_SHIFT = "cultural_shift"             # Cultural transformation
    TECHNOLOGICAL_CHANGE = "technological_change" # Tech-driven change
    POWER_TRANSITION = "power_transition"         # Change in power structure


class MutationEvent(Enum):
    """Types of mutations in legal lineages"""
    ADAPTATION = "adaptation"             # Adaptive change
    DRIFT = "drift"                      # Random drift
    SELECTION_PRESSURE = "selection_pressure"  # Environmental pressure
    HORIZONTAL_TRANSFER = "horizontal_transfer" # Cross-lineage transfer
    FOUNDER_EFFECT = "founder_effect"    # New jurisdiction founding
    BOTTLENECK = "bottleneck"           # Severe constraint event


@dataclass
class LegalLineage:
    """Represents a lineage of legal concepts"""
    lineage_id: str
    lineage_type: LineageType
    origin_date: datetime
    termination_date: Optional[datetime]
    
    # Lineage characteristics
    founding_constructor: Optional[str]    # Constructor that started lineage
    concept_sequence: List[str]           # Sequence of concepts in lineage
    mutation_history: List['MutationRecord']  # Record of mutations
    
    # Genealogical relationships
    parent_lineages: List[str]            # Parent lineage IDs
    child_lineages: List[str]             # Child lineage IDs
    sister_lineages: List[str]            # Sister lineage IDs (same coalescence)
    
    # Survival characteristics
    fitness_history: List[Tuple[datetime, float]]  # Fitness over time
    environmental_adaptations: List[str]   # Environmental pressures survived
    extinction_cause: Optional[str]        # Cause of extinction if terminated
    
    # Persistence metrics
    concept_fidelity: float               # How faithful concepts remain (0-1)
    transmission_rate: float              # Rate of concept transmission
    geographic_spread: List[str]          # Geographic jurisdictions
    institutional_embedding: float        # How embedded in institutions (0-1)


@dataclass 
class CoalescencePoint:
    """Represents a point where lineages coalesce to common ancestor"""
    coalescence_id: str
    coalescence_type: CoalescenceType
    coalescence_date: datetime
    
    # Coalescence characteristics
    ancestral_constructor: Optional[str]   # Constructor at coalescence point
    ancestral_concept: str                # Core concept at coalescence
    coalescence_probability: float        # Probability this is true coalescence (0-1)
    
    # Lineage convergence
    converging_lineages: List[str]        # Lineages that converge here
    divergence_after_coalescence: List[str]  # How lineages diverged after
    
    # Historical context
    historical_context: Dict[str, Any]    # Historical circumstances
    power_configuration: Dict[str, float] # Power distribution at time
    environmental_pressures: List[str]    # Environmental forces active
    
    # Evidence strength
    textual_evidence_strength: float      # Strength of textual evidence (0-1)
    institutional_evidence_strength: float # Strength of institutional evidence (0-1)
    genealogical_evidence_strength: float # Strength of genealogical evidence (0-1)


@dataclass
class CoalescenceTrace:
    """Complete trace from concept to coalescence point"""
    concept: str
    coalescence_point: CoalescencePoint
    original_constructor: Optional[str]
    
    # Trace characteristics
    genealogy_depth: int                  # How far back trace goes
    mutation_history: List['MutationRecord']  # Mutations along the way
    survival_lineages: List[LegalLineage] # Lineages that survived to present
    extinct_lineages: List[LegalLineage]  # Lineages that went extinct
    
    # Path analysis
    dominant_path: List[str]              # Most influential evolutionary path
    alternative_paths: List[List[str]]    # Alternative evolutionary paths
    path_probabilities: Dict[str, float]  # Probability of each path
    
    # Confidence metrics
    trace_confidence: float               # Confidence in trace accuracy (0-1)
    coalescence_confidence: float         # Confidence in coalescence point (0-1)
    uncertainty_factors: List[str]        # Sources of uncertainty


@dataclass
class ConstructorGenealogyMap:
    """Map of constructor genealogies in legal domain"""
    domain: str
    concept_traces: List[CoalescenceTrace]
    constructor_hierarchy: Dict[str, List[str]]  # Constructor relationships
    construction_patterns: List[str]      # Recurring construction patterns
    
    # Dominance analysis
    dominant_lineages: List[str]          # Currently dominant lineages
    emerging_lineages: List[str]          # Recently emerged lineages
    declining_lineages: List[str]         # Currently declining lineages
    
    # Construction efficiency
    construction_efficiency: Dict[str, float]  # Constructor efficiency scores
    successful_constructions: Dict[str, List[str]]  # Successful constructions per constructor
    failed_constructions: Dict[str, List[str]]     # Failed constructions per constructor
    
    # Evolution patterns
    convergent_evolution: List[Tuple[str, str]]  # Concepts evolving similarly
    divergent_evolution: List[Tuple[str, str]]   # Concepts evolving differently
    parallel_evolution: List[Tuple[str, str]]    # Independent similar evolution


@dataclass
class MutationRecord:
    """Record of mutation event in legal lineage"""
    mutation_id: str
    mutation_date: datetime
    mutation_type: MutationEvent
    
    # Mutation details
    original_concept: str
    mutated_concept: str
    triggering_pressure: Optional[str]
    mutation_magnitude: float             # How significant the change (0-1)
    
    # Context
    environmental_context: Dict[str, Any]
    constructor_involvement: Optional[str] # Constructor that influenced mutation
    
    # Outcomes
    fitness_change: float                 # Change in fitness from mutation
    survival_impact: float               # Impact on lineage survival
    propagation_success: float           # How successfully mutation spread


class CoalescenceTracker:
    """
    Main tracker for legal concept coalescence analysis
    
    Implements Dawkins' coalescence theory applied to legal memetics:
    traces legal concepts back to common ancestral constructors and
    identifies patterns of legal evolution and construction.
    """
    
    def __init__(self):
        self.lineage_database: Dict[str, LegalLineage] = {}
        self.coalescence_points: Dict[str, CoalescencePoint] = {}
        self.genealogy_networks: Dict[str, nx.DiGraph] = {}
        
        # Analysis parameters
        self.max_genealogy_depth = 100    # Maximum depth for genealogical trace
        self.coalescence_threshold = 0.7  # Minimum probability for coalescence
        self.mutation_significance_threshold = 0.3  # Minimum significance for mutations
        
        # Caching for performance
        self._trace_cache: Dict[str, CoalescenceTrace] = {}
        self._genealogy_cache: Dict[str, nx.DiGraph] = {}
        
    def trace_concept_coalescence(
        self, 
        legal_concept: str,
        genealogical_depth: int = None
    ) -> CoalescenceTrace:
        """
        Trace legal concept back to its coalescence point
        
        Args:
            legal_concept: Legal concept to trace
            genealogical_depth: Maximum depth to trace (default: self.max_genealogy_depth)
            
        Returns:
            Complete coalescence trace
        """
        depth = genealogical_depth or self.max_genealogy_depth
        
        # Check cache first
        cache_key = f"{legal_concept}_{depth}"
        if cache_key in self._trace_cache:
            return self._trace_cache[cache_key]
        
        # Build genealogy tree for concept
        genealogy_tree = self._build_concept_genealogy(legal_concept, depth)
        
        # Find coalescence point (most recent common ancestor)
        coalescence_point = self._find_coalescence_point(genealogy_tree)
        
        # Identify original constructor
        original_constructor = self._identify_original_constructor(coalescence_point)
        
        # Trace mutations and derivations
        mutation_history = self._trace_mutations(genealogy_tree, coalescence_point)
        
        # Identify surviving and extinct lineages
        surviving_lineages = self._identify_surviving_lineages(genealogy_tree)
        extinct_lineages = self._identify_extinct_lineages(genealogy_tree)
        
        # Analyze evolutionary paths
        path_analysis = self._analyze_evolutionary_paths(genealogy_tree, coalescence_point)
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_trace_confidence(
            genealogy_tree, coalescence_point, mutation_history
        )
        
        trace = CoalescenceTrace(
            concept=legal_concept,
            coalescence_point=coalescence_point,
            original_constructor=original_constructor,
            genealogy_depth=len(genealogy_tree.nodes),
            mutation_history=mutation_history,
            survival_lineages=surviving_lineages,
            extinct_lineages=extinct_lineages,
            dominant_path=path_analysis['dominant_path'],
            alternative_paths=path_analysis['alternative_paths'],
            path_probabilities=path_analysis['path_probabilities'],
            trace_confidence=confidence_metrics['trace_confidence'],
            coalescence_confidence=confidence_metrics['coalescence_confidence'],
            uncertainty_factors=confidence_metrics['uncertainty_factors']
        )
        
        # Cache result
        self._trace_cache[cache_key] = trace
        
        return trace
    
    def map_constructor_genealogy(
        self, 
        legal_domain: str
    ) -> ConstructorGenealogyMap:
        """
        Map complete genealogy of constructors in legal domain
        
        Args:
            legal_domain: Legal domain to analyze
            
        Returns:
            Complete constructor genealogy map
        """
        # Identify all concepts in domain
        domain_concepts = self._extract_domain_concepts(legal_domain)
        
        # Trace coalescence for each concept
        coalescence_traces = []
        for concept in domain_concepts:
            trace = self.trace_concept_coalescence(concept)
            coalescence_traces.append(trace)
        
        # Build constructor hierarchy map
        constructor_map = self._build_constructor_hierarchy_map(coalescence_traces)
        
        # Identify construction patterns
        construction_patterns = self._analyze_construction_patterns(coalescence_traces)
        
        # Analyze lineage dynamics
        lineage_analysis = self._analyze_lineage_dynamics(coalescence_traces)
        
        # Calculate construction efficiency
        construction_efficiency = self._calculate_constructor_efficiency(coalescence_traces)
        
        # Identify evolution patterns
        evolution_patterns = self._identify_evolution_patterns(coalescence_traces)
        
        return ConstructorGenealogyMap(
            domain=legal_domain,
            concept_traces=coalescence_traces,
            constructor_hierarchy=constructor_map,
            construction_patterns=construction_patterns,
            dominant_lineages=lineage_analysis['dominant'],
            emerging_lineages=lineage_analysis['emerging'],
            declining_lineages=lineage_analysis['declining'],
            construction_efficiency=construction_efficiency,
            successful_constructions=self._map_successful_constructions(coalescence_traces),
            failed_constructions=self._map_failed_constructions(coalescence_traces),
            convergent_evolution=evolution_patterns['convergent'],
            divergent_evolution=evolution_patterns['divergent'],
            parallel_evolution=evolution_patterns['parallel']
        )
    
    def find_common_ancestor(
        self, 
        concepts: List[str]
    ) -> Optional[CoalescencePoint]:
        """
        Find most recent common ancestor of multiple concepts
        
        Args:
            concepts: List of concepts to find common ancestor for
            
        Returns:
            Common coalescence point if found
        """
        if not concepts:
            return None
        
        # Get genealogy trees for all concepts
        genealogy_trees = [
            self._build_concept_genealogy(concept, self.max_genealogy_depth)
            for concept in concepts
        ]
        
        # Find intersection of all ancestral paths
        common_ancestors = self._find_common_ancestors(genealogy_trees)
        
        if not common_ancestors:
            return None
        
        # Return most recent common ancestor
        return self._select_most_recent_ancestor(common_ancestors)
    
    def analyze_lineage_competition(
        self,
        lineages: List[LegalLineage],
        time_period: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Analyze competition between legal lineages
        
        Args:
            lineages: List of lineages to analyze
            time_period: Time period for analysis
            
        Returns:
            Competition analysis results
        """
        # Filter lineages active during time period
        active_lineages = [
            lineage for lineage in lineages 
            if self._lineage_active_during_period(lineage, time_period)
        ]
        
        # Build competition network
        competition_network = self._build_competition_network(active_lineages)
        
        # Analyze competitive interactions
        competitive_analysis = self._analyze_competitive_interactions(
            active_lineages, 
            competition_network
        )
        
        # Predict winners and losers
        survival_predictions = self._predict_lineage_survival(
            active_lineages,
            competition_network,
            time_period
        )
        
        return {
            'active_lineages': active_lineages,
            'competition_network': competition_network,
            'competitive_interactions': competitive_analysis,
            'survival_predictions': survival_predictions,
            'dominant_lineage': self._identify_dominant_lineage(active_lineages),
            'extinction_risks': self._assess_extinction_risks(active_lineages),
            'niche_overlaps': self._calculate_niche_overlaps(active_lineages)
        }
    
    def predict_concept_evolution(
        self,
        concept: str,
        future_pressures: Dict[str, float],
        time_horizon: int = 20
    ) -> Dict[str, Any]:
        """
        Predict future evolution of legal concept
        
        Args:
            concept: Legal concept to analyze
            future_pressures: Expected future environmental pressures
            time_horizon: Years to predict into future
            
        Returns:
            Evolution prediction analysis
        """
        # Get current lineage for concept
        current_lineage = self._get_concept_current_lineage(concept)
        
        if not current_lineage:
            return {'error': f'No active lineage found for concept: {concept}'}
        
        # Analyze historical mutation patterns
        mutation_patterns = self._analyze_historical_mutations(current_lineage)
        
        # Predict mutation probability under future pressures
        mutation_probability = self._predict_mutation_probability(
            current_lineage,
            future_pressures,
            mutation_patterns
        )
        
        # Predict potential mutations
        potential_mutations = self._predict_potential_mutations(
            current_lineage,
            future_pressures,
            time_horizon
        )
        
        # Assess survival probability
        survival_probability = self._assess_concept_survival_probability(
            current_lineage,
            future_pressures,
            time_horizon
        )
        
        # Predict fitness trajectory
        fitness_trajectory = self._predict_fitness_trajectory(
            current_lineage,
            future_pressures,
            time_horizon
        )
        
        return {
            'concept': concept,
            'current_lineage': current_lineage,
            'mutation_probability': mutation_probability,
            'potential_mutations': potential_mutations,
            'survival_probability': survival_probability,
            'fitness_trajectory': fitness_trajectory,
            'recommended_adaptations': self._recommend_adaptations(
                current_lineage, future_pressures
            ),
            'extinction_scenarios': self._generate_extinction_scenarios(
                current_lineage, future_pressures
            )
        }
    
    # Private implementation methods
    
    def _build_concept_genealogy(
        self, 
        concept: str, 
        max_depth: int
    ) -> nx.DiGraph:
        """Build genealogy tree for concept"""
        # Check cache
        cache_key = f"{concept}_{max_depth}"
        if cache_key in self._genealogy_cache:
            return self._genealogy_cache[cache_key]
        
        genealogy = nx.DiGraph()
        genealogy.add_node(concept, depth=0, type='current')
        
        # Use BFS to build genealogy tree
        queue = deque([(concept, 0)])
        visited = {concept}
        
        while queue and len(queue) > 0:
            current_concept, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Find parent concepts (predecessors)
            parent_concepts = self._find_parent_concepts(current_concept)
            
            for parent in parent_concepts:
                if parent not in visited:
                    genealogy.add_node(parent, depth=depth+1, type='ancestor')
                    genealogy.add_edge(parent, current_concept, 
                                     relationship='precedes',
                                     confidence=self._calculate_relationship_confidence(parent, current_concept))
                    queue.append((parent, depth+1))
                    visited.add(parent)
                elif parent in genealogy:
                    # Add edge if not already present
                    if not genealogy.has_edge(parent, current_concept):
                        genealogy.add_edge(parent, current_concept,
                                         relationship='precedes',
                                         confidence=self._calculate_relationship_confidence(parent, current_concept))
        
        # Cache result
        self._genealogy_cache[cache_key] = genealogy
        
        return genealogy
    
    def _find_coalescence_point(self, genealogy_tree: nx.DiGraph) -> CoalescencePoint:
        """Find coalescence point in genealogy tree"""
        # Find root nodes (nodes with no predecessors)
        root_nodes = [node for node in genealogy_tree.nodes() 
                     if genealogy_tree.in_degree(node) == 0]
        
        if not root_nodes:
            # If no clear root, find node with maximum ancestor count
            ancestor_counts = {
                node: len(list(nx.ancestors(genealogy_tree, node)))
                for node in genealogy_tree.nodes()
            }
            root_concept = max(ancestor_counts.keys(), key=lambda k: ancestor_counts[k])
        else:
            # If multiple roots, find the one with most descendants
            descendant_counts = {
                root: len(list(nx.descendants(genealogy_tree, root)))
                for root in root_nodes
            }
            root_concept = max(descendant_counts.keys(), key=lambda k: descendant_counts[k])
        
        # Create coalescence point
        coalescence_date = self._estimate_concept_origin_date(root_concept)
        coalescence_type = self._classify_coalescence_type(root_concept, genealogy_tree)
        
        return CoalescencePoint(
            coalescence_id=str(uuid.uuid4()),
            coalescence_type=coalescence_type,
            coalescence_date=coalescence_date,
            ancestral_constructor=self._identify_concept_constructor(root_concept),
            ancestral_concept=root_concept,
            coalescence_probability=self._calculate_coalescence_probability(root_concept, genealogy_tree),
            converging_lineages=self._identify_converging_lineages(root_concept, genealogy_tree),
            divergence_after_coalescence=self._identify_post_coalescence_divergence(root_concept, genealogy_tree),
            historical_context=self._extract_historical_context(root_concept, coalescence_date),
            power_configuration=self._extract_power_configuration(root_concept, coalescence_date),
            environmental_pressures=self._extract_environmental_pressures(root_concept, coalescence_date),
            textual_evidence_strength=self._calculate_textual_evidence_strength(root_concept),
            institutional_evidence_strength=self._calculate_institutional_evidence_strength(root_concept),
            genealogical_evidence_strength=self._calculate_genealogical_evidence_strength(root_concept, genealogy_tree)
        )
    
    def _find_parent_concepts(self, concept: str) -> List[str]:
        """Find parent concepts that influenced this concept"""
        # This would implement sophisticated concept relationship detection
        # For now, simplified heuristic approach
        
        parent_concepts = []
        
        # Look in concept database for relationships
        if concept in self.lineage_database:
            lineage = self.lineage_database[concept]
            # Find concepts that appear before this one in sequence
            concept_index = lineage.concept_sequence.index(concept) if concept in lineage.concept_sequence else -1
            if concept_index > 0:
                parent_concepts.append(lineage.concept_sequence[concept_index - 1])
        
        # Look for conceptual similarities (simplified)
        similar_concepts = self._find_conceptually_similar_concepts(concept)
        for similar_concept in similar_concepts:
            if self._concept_predates(similar_concept, concept):
                parent_concepts.append(similar_concept)
        
        return parent_concepts
    
    def _calculate_relationship_confidence(self, parent: str, child: str) -> float:
        """Calculate confidence in parent-child relationship"""
        # Simplified confidence calculation
        # In practice, this would use sophisticated NLP and historical analysis
        
        # Textual similarity factor
        similarity_score = self._calculate_concept_similarity(parent, child)
        
        # Temporal factor (closer in time = higher confidence)
        temporal_distance = self._calculate_temporal_distance(parent, child)
        temporal_factor = max(0.1, 1.0 - (temporal_distance / 100.0))  # 100 year max
        
        # Institutional connection factor
        institutional_connection = self._calculate_institutional_connection(parent, child)
        
        return (similarity_score * 0.4 + temporal_factor * 0.3 + institutional_connection * 0.3)
    
    def _identify_original_constructor(self, coalescence_point: CoalescencePoint) -> Optional[str]:
        """Identify the original constructor at coalescence point"""
        if coalescence_point.ancestral_constructor:
            return coalescence_point.ancestral_constructor
        
        # Try to infer from historical context
        historical_context = coalescence_point.historical_context
        power_config = coalescence_point.power_configuration
        
        # Find dominant power entity at coalescence time
        if power_config:
            dominant_entity = max(power_config.keys(), key=lambda k: power_config[k])
            return dominant_entity
        
        return None
    
    def _trace_mutations(
        self, 
        genealogy_tree: nx.DiGraph, 
        coalescence_point: CoalescencePoint
    ) -> List[MutationRecord]:
        """Trace mutation events in genealogy"""
        mutations = []
        
        # Walk through genealogy tree from coalescence point
        for edge in genealogy_tree.edges(data=True):
            parent, child, data = edge
            
            # Check if this represents a significant mutation
            mutation_magnitude = self._calculate_mutation_magnitude(parent, child)
            
            if mutation_magnitude > self.mutation_significance_threshold:
                mutation = MutationRecord(
                    mutation_id=str(uuid.uuid4()),
                    mutation_date=self._estimate_mutation_date(parent, child),
                    mutation_type=self._classify_mutation_type(parent, child),
                    original_concept=parent,
                    mutated_concept=child,
                    triggering_pressure=self._identify_triggering_pressure(parent, child),
                    mutation_magnitude=mutation_magnitude,
                    environmental_context=self._extract_mutation_context(parent, child),
                    constructor_involvement=self._identify_mutation_constructor(parent, child),
                    fitness_change=self._calculate_mutation_fitness_change(parent, child),
                    survival_impact=self._calculate_survival_impact(parent, child),
                    propagation_success=self._calculate_propagation_success(child)
                )
                mutations.append(mutation)
        
        return mutations
    
    # Additional utility methods
    
    def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between two concepts"""
        # Simplified similarity - would use advanced NLP
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _concept_predates(self, concept1: str, concept2: str) -> bool:
        """Check if concept1 predates concept2"""
        # Simplified temporal check - would implement sophisticated dating
        date1 = self._estimate_concept_origin_date(concept1)
        date2 = self._estimate_concept_origin_date(concept2)
        return date1 < date2
    
    def _estimate_concept_origin_date(self, concept: str) -> datetime:
        """Estimate origin date of concept"""
        # Simplified dating - would implement sophisticated historical analysis
        # For now, return a reasonable default
        return datetime(1800, 1, 1)  # Default to 19th century
    
    def __repr__(self) -> str:
        return f"CoalescenceTracker(lineages={len(self.lineage_database)}, coalescence_points={len(self.coalescence_points)})"