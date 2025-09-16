"""
Intra-Genomic Conflict Analyzer - Implementation of Dawkins 2024 Parliament of Genes

Models internal conflicts within constructors as "parliaments of genes" where
different interests (genes) within the same entity compete and form coalitions
to influence legal construction decisions.

Captures the reality that constructors are not monolithic but contain
competing internal interests that must be resolved through internal
decision-making processes.

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
from collections import defaultdict, Counter
import networkx as nx


class ConflictType(Enum):
    """Types of intra-genomic conflicts"""
    TEMPORAL_CONFLICT = "temporal_conflict"        # Short vs long term interests
    FUNCTIONAL_CONFLICT = "functional_conflict"    # Different functional objectives
    CONSTITUENCY_CONFLICT = "constituency_conflict" # Different constituencies served
    RESOURCE_CONFLICT = "resource_conflict"        # Competition for limited resources
    IDEOLOGICAL_CONFLICT = "ideological_conflict" # Conflicting belief systems
    STRATEGIC_CONFLICT = "strategic_conflict"     # Different strategic approaches
    REGULATORY_CONFLICT = "regulatory_conflict"   # Compliance vs profit conflicts


class GeneType(Enum):
    """Types of interest genes within constructors"""
    SURVIVAL_GENE = "survival_gene"              # Basic organizational survival
    GROWTH_GENE = "growth_gene"                  # Expansion and growth
    PROFIT_GENE = "profit_gene"                  # Financial optimization
    LEGITIMACY_GENE = "legitimacy_gene"         # Maintaining legitimacy
    POWER_GENE = "power_gene"                   # Accumulating power/influence
    COMPLIANCE_GENE = "compliance_gene"         # Regulatory compliance
    INNOVATION_GENE = "innovation_gene"         # Innovation and adaptation
    STABILITY_GENE = "stability_gene"           # Maintaining stable operations
    REPUTATION_GENE = "reputation_gene"         # Protecting reputation


class VotingMechanism(Enum):
    """Mechanisms for resolving gene conflicts"""
    WEIGHTED_VOTING = "weighted_voting"         # Votes weighted by gene power
    CONSENSUS_BUILDING = "consensus_building"   # Seeking unanimous agreement
    DOMINANT_COALITION = "dominant_coalition"   # Majority coalition rules
    HIERARCHICAL_OVERRIDE = "hierarchical_override" # CEO/leader override
    SEQUENTIAL_CONSIDERATION = "sequential_consideration" # Consider each gene in turn
    MARKET_MECHANISM = "market_mechanism"       # Internal market for decisions
    ROTATING_LEADERSHIP = "rotating_leadership" # Different genes lead different decisions


class CoalitionType(Enum):
    """Types of coalitions genes can form"""
    PERMANENT_ALLIANCE = "permanent_alliance"   # Long-term stable alliance
    TACTICAL_COALITION = "tactical_coalition"   # Temporary strategic alliance
    ISSUE_SPECIFIC = "issue_specific"          # Coalition for specific issues
    DEFENSIVE_ALLIANCE = "defensive_alliance"   # Alliance against common threat
    OPPORTUNISTIC = "opportunistic"            # Coalition of convenience


@dataclass
class InterestGene:
    """Individual interest gene within constructor"""
    gene_id: str
    gene_type: GeneType
    description: str
    priority_weight: float              # Base priority of this gene (0-1)
    
    # Resource requirements
    resource_demands: Dict[str, float]  # Resources this gene requires
    resource_constraints: Dict[str, float] # Constraints this gene faces
    
    # Temporal characteristics  
    temporal_discount_rate: float       # How much gene discounts future (0-1)
    planning_horizon: int              # Years gene plans ahead
    urgency_level: float              # How urgent gene's needs are (0-1)
    
    # Interaction characteristics
    compatible_genes: List[str]        # Genes this gene works well with
    conflicting_genes: List[str]       # Genes this gene conflicts with
    coalition_preferences: List[str]   # Preferred coalition partners
    
    # Environmental responsiveness
    environmental_triggers: Dict[str, float] # Environmental factors that activate gene
    adaptability: float               # How adaptable this gene is (0-1)
    
    # Decision influence
    veto_power: float                # Ability to veto decisions (0-1)
    agenda_setting_power: float      # Ability to set agenda (0-1)
    information_access: float        # Access to relevant information (0-1)


@dataclass
class GeneCoalition:
    """Coalition of genes within constructor"""
    coalition_id: str
    coalition_type: CoalitionType
    member_genes: List[str]           # Gene IDs in coalition
    formation_date: datetime
    dissolution_date: Optional[datetime]
    
    # Coalition characteristics
    coalition_strength: float        # Overall strength (0-1)
    internal_cohesion: float         # How unified coalition is (0-1)
    external_influence: float        # Influence on constructor decisions (0-1)
    
    # Coalition dynamics
    dominant_gene: Optional[str]     # Most influential gene in coalition
    coalition_agenda: List[str]      # Shared priorities
    resource_pooling: Dict[str, float] # Resources shared within coalition
    
    # Stability factors
    stability_factors: List[str]     # What keeps coalition together
    instability_risks: List[str]    # What could break coalition apart
    expected_duration: Optional[int] # Expected duration in years


@dataclass
class ConflictEvent:
    """Specific conflict event between genes"""
    conflict_id: str
    conflict_type: ConflictType
    conflicting_genes: List[str]
    conflict_date: datetime
    resolution_date: Optional[datetime]
    
    # Conflict details
    conflict_issue: str              # What the conflict is about
    conflict_intensity: float        # Intensity of conflict (0-1)
    resolution_mechanism: Optional[VotingMechanism]
    
    # Stakes and outcomes
    stakes_involved: Dict[str, float] # What each gene stands to gain/lose
    actual_outcome: Optional[str]    # How conflict was actually resolved
    winner_genes: List[str]          # Genes that got their way
    loser_genes: List[str]           # Genes that were overruled
    
    # Impact assessment
    constructor_impact: float        # Impact on constructor overall (0-1)
    phenotype_impact: float         # Impact on resulting phenotypes (0-1)
    long_term_consequences: List[str] # Long-term effects of resolution


@dataclass
class ParliamentModel:
    """Model of parliamentary process within constructor"""
    constructor_id: str
    active_genes: List[InterestGene]
    voting_structure: Dict[str, float] # Voting weights by gene
    
    # Parliamentary dynamics
    parliamentary_dynamics: 'ParliamentaryProcess'
    predicted_outcome: Dict[str, Any]
    coalition_patterns: List[GeneCoalition]
    
    # Power distribution
    power_balance: Dict[str, float]  # Current power of each gene
    swing_genes: List[str]           # Genes that could change outcomes
    dominant_coalition: Optional[GeneCoalition]
    
    # Decision-making characteristics
    decision_speed: float           # How quickly decisions are made
    decision_quality: float         # Quality of decisions made
    decision_stability: float       # Stability of decisions over time
    
    # Minority dynamics
    minority_suppression: float     # How much minorities are suppressed (0-1)
    minority_protection_mechanisms: List[str] # Protections for minority genes
    veto_players: List[str]         # Genes with veto power


@dataclass
class InternalConflictAnalysis:
    """Complete analysis of internal conflicts within constructor"""
    constructor: 'Constructor'
    sub_interests: List[InterestGene]
    conflict_intensity: np.ndarray   # Matrix of conflict intensities
    
    # Phenotype analysis
    contradictory_phenotypes: List[str] # Phenotypes that contradict each other
    phenotype_conflicts: Dict[str, List[str]] # Conflicts by phenotype
    
    # Resolution prediction
    predicted_resolution: Dict[str, Any]
    stability_assessment: float      # Overall stability of constructor (0-1)
    
    # Conflict patterns
    recurring_conflicts: List[ConflictType] # Types of conflicts that recur
    seasonal_patterns: Dict[str, List[datetime]] # Timing patterns of conflicts
    trigger_events: List[str]       # Events that typically trigger conflicts
    
    # Performance impact
    decision_paralysis_risk: float  # Risk of being unable to decide (0-1)
    strategic_coherence: float      # Coherence of overall strategy (0-1)
    adaptation_capacity: float      # Ability to adapt under conflict (0-1)


class IntraGenomicConflictAnalyzer:
    """
    Analyzer for internal conflicts within constructors
    
    Models constructors as containing multiple competing "genes" (interests)
    that must be resolved through internal parliamentary processes,
    following Dawkins' parliament of genes concept.
    """
    
    def __init__(self):
        self.gene_databases: Dict[str, List[InterestGene]] = {}
        self.conflict_history: List[ConflictEvent] = {}
        self.coalition_networks: Dict[str, nx.Graph] = {}
        
        # Analysis parameters
        self.conflict_threshold = 0.3       # Minimum intensity to count as conflict
        self.coalition_stability_threshold = 0.6  # Minimum stability for viable coalition
        self.parliamentary_efficiency_target = 0.7 # Target efficiency for decisions
        
        # Conflict resolution patterns
        self.resolution_success_rates = {
            VotingMechanism.WEIGHTED_VOTING: 0.7,
            VotingMechanism.CONSENSUS_BUILDING: 0.9,
            VotingMechanism.DOMINANT_COALITION: 0.6,
            VotingMechanism.HIERARCHICAL_OVERRIDE: 0.8,
            VotingMechanism.SEQUENTIAL_CONSIDERATION: 0.5,
            VotingMechanism.MARKET_MECHANISM: 0.6,
            VotingMechanism.ROTATING_LEADERSHIP: 0.4
        }
    
    def analyze_constructor_internal_conflicts(
        self, 
        constructor: 'Constructor'
    ) -> InternalConflictAnalysis:
        """
        Analyze internal conflicts within constructor
        
        Args:
            constructor: Constructor to analyze
            
        Returns:
            Complete internal conflict analysis
        """
        # Decompose constructor into constituent interest genes
        sub_interests = self._decompose_constructor_interests(constructor)
        
        # Build conflict intensity matrix
        conflict_matrix = self._build_internal_conflict_matrix(sub_interests)
        
        # Identify contradictory phenotypes
        contradictory_phenotypes = self._identify_contradictory_phenotypes(
            constructor.phenotype_portfolio if hasattr(constructor, 'phenotype_portfolio') else [],
            conflict_matrix
        )
        
        # Map phenotype conflicts
        phenotype_conflicts = self._map_phenotype_conflicts(
            constructor.phenotype_portfolio if hasattr(constructor, 'phenotype_portfolio') else [],
            sub_interests,
            conflict_matrix
        )
        
        # Predict conflict resolution
        conflict_resolution = self._predict_conflict_resolution(
            conflict_matrix,
            constructor
        )
        
        # Assess constructor stability
        stability_assessment = self._assess_constructor_stability(
            conflict_matrix,
            sub_interests
        )
        
        # Analyze conflict patterns
        conflict_patterns = self._analyze_conflict_patterns(
            constructor,
            conflict_matrix
        )
        
        # Assess performance impact
        performance_impact = self._assess_performance_impact(
            conflict_matrix,
            sub_interests,
            constructor
        )
        
        return InternalConflictAnalysis(
            constructor=constructor,
            sub_interests=sub_interests,
            conflict_intensity=conflict_matrix,
            contradictory_phenotypes=contradictory_phenotypes,
            phenotype_conflicts=phenotype_conflicts,
            predicted_resolution=conflict_resolution,
            stability_assessment=stability_assessment,
            recurring_conflicts=conflict_patterns['recurring_types'],
            seasonal_patterns=conflict_patterns['seasonal_patterns'],
            trigger_events=conflict_patterns['trigger_events'],
            decision_paralysis_risk=performance_impact['paralysis_risk'],
            strategic_coherence=performance_impact['strategic_coherence'],
            adaptation_capacity=performance_impact['adaptation_capacity']
        )
    
    def model_parliament_of_genes(
        self, 
        constructor: 'Constructor',
        decision_context: 'DecisionContext'
    ) -> ParliamentModel:
        """
        Model parliamentary process within constructor
        
        Args:
            constructor: Constructor to model
            decision_context: Context for decision being made
            
        Returns:
            Complete parliamentary model
        """
        # Identify genes active for this decision
        active_genes = self._identify_active_genes(constructor, decision_context)
        
        # Calculate voting weights
        voting_weights = self._calculate_gene_voting_weights(
            active_genes,
            constructor,
            decision_context
        )
        
        # Simulate parliamentary process
        parliamentary_process = self._simulate_parliamentary_process(
            active_genes,
            voting_weights,
            decision_context
        )
        
        # Predict decision outcome
        decision_outcome = self._predict_parliamentary_outcome(
            parliamentary_process,
            active_genes,
            voting_weights
        )
        
        # Analyze coalition patterns
        coalition_patterns = self._analyze_coalition_patterns(
            active_genes,
            parliamentary_process
        )
        
        # Calculate power balance
        power_balance = self._calculate_current_power_balance(
            active_genes,
            voting_weights,
            coalition_patterns
        )
        
        # Identify swing genes
        swing_genes = self._identify_swing_genes(
            active_genes,
            voting_weights,
            coalition_patterns
        )
        
        # Analyze minority dynamics
        minority_analysis = self._analyze_minority_suppression(
            active_genes,
            parliamentary_process,
            power_balance
        )
        
        return ParliamentModel(
            constructor_id=constructor.constructor_id if hasattr(constructor, 'constructor_id') else 'unknown',
            active_genes=active_genes,
            voting_structure=voting_weights,
            parliamentary_dynamics=parliamentary_process,
            predicted_outcome=decision_outcome,
            coalition_patterns=coalition_patterns,
            power_balance=power_balance,
            swing_genes=swing_genes,
            dominant_coalition=self._identify_dominant_coalition(coalition_patterns),
            decision_speed=self._calculate_decision_speed(parliamentary_process),
            decision_quality=self._estimate_decision_quality(parliamentary_process, decision_outcome),
            decision_stability=self._estimate_decision_stability(decision_outcome, coalition_patterns),
            minority_suppression=minority_analysis['suppression_level'],
            minority_protection_mechanisms=minority_analysis['protection_mechanisms'],
            veto_players=minority_analysis['veto_players']
        )
    
    def predict_conflict_evolution(
        self,
        constructor: 'Constructor',
        future_pressures: Dict[str, float],
        time_horizon: int = 5
    ) -> Dict[str, Any]:
        """
        Predict how internal conflicts will evolve
        
        Args:
            constructor: Constructor to analyze
            future_pressures: Expected future environmental pressures
            time_horizon: Years to predict ahead
            
        Returns:
            Conflict evolution predictions
        """
        # Get current conflict state
        current_conflicts = self.analyze_constructor_internal_conflicts(constructor)
        
        # Predict how genes will respond to future pressures
        gene_responses = self._predict_gene_responses_to_pressure(
            current_conflicts.sub_interests,
            future_pressures
        )
        
        # Predict new conflicts that may emerge
        emerging_conflicts = self._predict_emerging_conflicts(
            current_conflicts.sub_interests,
            gene_responses,
            future_pressures
        )
        
        # Predict resolution of existing conflicts
        conflict_resolutions = self._predict_existing_conflict_resolution(
            current_conflicts,
            gene_responses,
            time_horizon
        )
        
        # Predict changes in gene power distribution
        power_evolution = self._predict_power_distribution_evolution(
            current_conflicts.sub_interests,
            future_pressures,
            time_horizon
        )
        
        # Assess overall constructor evolution
        constructor_evolution = self._assess_constructor_evolution(
            constructor,
            emerging_conflicts,
            conflict_resolutions,
            power_evolution
        )
        
        return {
            'current_state': current_conflicts,
            'gene_responses': gene_responses,
            'emerging_conflicts': emerging_conflicts,
            'conflict_resolutions': conflict_resolutions,
            'power_evolution': power_evolution,
            'constructor_evolution': constructor_evolution,
            'stability_trajectory': self._predict_stability_trajectory(
                current_conflicts, emerging_conflicts, time_horizon
            ),
            'adaptation_scenarios': self._generate_adaptation_scenarios(
                constructor, future_pressures, time_horizon
            )
        }
    
    def analyze_coalition_dynamics(
        self,
        constructor: 'Constructor',
        time_period: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Analyze coalition formation and dissolution dynamics
        
        Args:
            constructor: Constructor to analyze
            time_period: Time period for analysis
            
        Returns:
            Coalition dynamics analysis
        """
        # Extract gene coalitions during time period
        historical_coalitions = self._extract_historical_coalitions(
            constructor,
            time_period
        )
        
        # Analyze coalition formation patterns
        formation_patterns = self._analyze_coalition_formation_patterns(
            historical_coalitions
        )
        
        # Analyze coalition stability
        stability_analysis = self._analyze_coalition_stability(
            historical_coalitions
        )
        
        # Identify successful vs failed coalitions
        coalition_success_analysis = self._analyze_coalition_success_rates(
            historical_coalitions
        )
        
        # Predict future coalition formations
        future_coalitions = self._predict_future_coalitions(
            constructor,
            formation_patterns,
            stability_analysis
        )
        
        return {
            'historical_coalitions': historical_coalitions,
            'formation_patterns': formation_patterns,
            'stability_analysis': stability_analysis,
            'success_rates': coalition_success_analysis,
            'future_predictions': future_coalitions,
            'optimal_coalitions': self._identify_optimal_coalitions(
                constructor, formation_patterns
            ),
            'coalition_networks': self._build_coalition_networks(historical_coalitions)
        }
    
    # Private implementation methods
    
    def _decompose_constructor_interests(
        self, 
        constructor: 'Constructor'
    ) -> List[InterestGene]:
        """Decompose constructor into constituent interest genes"""
        genes = []
        
        # Extract genes from constructor's interests genome if available
        if hasattr(constructor, 'interests_genome'):
            for i, interest in enumerate(constructor.interests_genome):
                gene = InterestGene(
                    gene_id=f"{constructor.constructor_id}_{i}" if hasattr(constructor, 'constructor_id') else f"gene_{i}",
                    gene_type=self._classify_gene_type(interest),
                    description=interest.description if hasattr(interest, 'description') else str(interest),
                    priority_weight=interest.priority if hasattr(interest, 'priority') else 1.0 / len(constructor.interests_genome),
                    resource_demands=self._estimate_gene_resource_demands(interest),
                    resource_constraints=self._estimate_gene_constraints(interest),
                    temporal_discount_rate=self._estimate_temporal_discount_rate(interest),
                    planning_horizon=self._estimate_planning_horizon(interest),
                    urgency_level=self._estimate_urgency_level(interest),
                    compatible_genes=self._identify_compatible_genes(interest, constructor.interests_genome),
                    conflicting_genes=self._identify_conflicting_genes(interest, constructor.interests_genome),
                    coalition_preferences=self._identify_coalition_preferences(interest, constructor.interests_genome),
                    environmental_triggers=self._identify_environmental_triggers(interest),
                    adaptability=self._estimate_gene_adaptability(interest),
                    veto_power=self._estimate_veto_power(interest),
                    agenda_setting_power=self._estimate_agenda_power(interest),
                    information_access=self._estimate_information_access(interest)
                )
                genes.append(gene)
        else:
            # Create default genes for typical constructors
            genes = self._create_default_genes(constructor)
        
        return genes
    
    def _build_internal_conflict_matrix(
        self, 
        sub_interests: List[InterestGene]
    ) -> np.ndarray:
        """Build matrix of conflict intensities between genes"""
        n = len(sub_interests)
        conflict_matrix = np.zeros((n, n))
        
        for i, gene_i in enumerate(sub_interests):
            for j, gene_j in enumerate(sub_interests):
                if i != j:
                    conflict_intensity = self._calculate_gene_conflict_intensity(
                        gene_i, gene_j
                    )
                    conflict_matrix[i, j] = conflict_intensity
        
        return conflict_matrix
    
    def _calculate_gene_conflict_intensity(
        self, 
        gene_i: InterestGene, 
        gene_j: InterestGene
    ) -> float:
        """Calculate conflict intensity between two genes"""
        conflict_factors = []
        
        # Resource competition
        resource_conflict = self._calculate_resource_conflict(gene_i, gene_j)
        conflict_factors.append(resource_conflict * 0.3)
        
        # Temporal conflict (different time horizons)
        temporal_conflict = abs(gene_i.planning_horizon - gene_j.planning_horizon) / 20.0  # Normalize by 20 years
        conflict_factors.append(temporal_conflict * 0.2)
        
        # Priority conflict  
        priority_conflict = abs(gene_i.priority_weight - gene_j.priority_weight)
        conflict_factors.append(priority_conflict * 0.2)
        
        # Explicit conflicts
        if gene_j.gene_id in gene_i.conflicting_genes:
            conflict_factors.append(0.8)
        
        # Urgency conflict
        urgency_conflict = abs(gene_i.urgency_level - gene_j.urgency_level)
        conflict_factors.append(urgency_conflict * 0.15)
        
        # Temporal discount conflict
        discount_conflict = abs(gene_i.temporal_discount_rate - gene_j.temporal_discount_rate)
        conflict_factors.append(discount_conflict * 0.15)
        
        return min(1.0, sum(conflict_factors))
    
    def _calculate_resource_conflict(
        self, 
        gene_i: InterestGene, 
        gene_j: InterestGene
    ) -> float:
        """Calculate resource competition between genes"""
        # Calculate overlap in resource demands
        resources_i = set(gene_i.resource_demands.keys())
        resources_j = set(gene_j.resource_demands.keys())
        
        common_resources = resources_i.intersection(resources_j)
        
        if not common_resources:
            return 0.0
        
        # Calculate intensity of competition for common resources
        competition_intensity = 0.0
        for resource in common_resources:
            demand_i = gene_i.resource_demands[resource]
            demand_j = gene_j.resource_demands[resource]
            
            # Higher demands = more competition
            resource_competition = (demand_i + demand_j) / 2.0
            competition_intensity += resource_competition
        
        return min(1.0, competition_intensity / len(common_resources))
    
    def _simulate_parliamentary_process(
        self,
        active_genes: List[InterestGene],
        voting_weights: Dict[str, float],
        decision_context: 'DecisionContext'
    ) -> 'ParliamentaryProcess':
        """Simulate parliamentary process among genes"""
        
        # Create simplified parliamentary process simulation
        parliamentary_process = {
            'agenda_setting_phase': self._simulate_agenda_setting(active_genes, voting_weights),
            'coalition_formation_phase': self._simulate_coalition_formation(active_genes, voting_weights),
            'deliberation_phase': self._simulate_deliberation(active_genes, decision_context),
            'voting_phase': self._simulate_voting(active_genes, voting_weights),
            'implementation_phase': self._simulate_implementation(active_genes, voting_weights)
        }
        
        return parliamentary_process
    
    def _identify_active_genes(
        self, 
        constructor: 'Constructor', 
        decision_context: 'DecisionContext'
    ) -> List[InterestGene]:
        """Identify genes active for specific decision context"""
        all_genes = self._decompose_constructor_interests(constructor)
        active_genes = []
        
        for gene in all_genes:
            # Check if gene is triggered by decision context
            activation_score = self._calculate_gene_activation_score(gene, decision_context)
            
            if activation_score > 0.3:  # Activation threshold
                active_genes.append(gene)
        
        return active_genes
    
    def _calculate_gene_activation_score(
        self, 
        gene: InterestGene, 
        decision_context: 'DecisionContext'
    ) -> float:
        """Calculate how activated a gene is by decision context"""
        activation_factors = []
        
        # Base activation from priority
        activation_factors.append(gene.priority_weight)
        
        # Environmental trigger activation
        context_factors = getattr(decision_context, 'environmental_factors', {})
        for trigger, threshold in gene.environmental_triggers.items():
            if trigger in context_factors and context_factors[trigger] > threshold:
                activation_factors.append(0.8)
        
        # Urgency factor
        if hasattr(decision_context, 'urgency') and decision_context.urgency > 0.5:
            activation_factors.append(gene.urgency_level)
        
        return min(1.0, np.mean(activation_factors) if activation_factors else 0.0)
    
    def _classify_gene_type(self, interest) -> GeneType:
        """Classify interest as specific gene type"""
        # Simplified classification based on interest description
        if hasattr(interest, 'description'):
            desc = interest.description.lower()
            if 'survival' in desc or 'exist' in desc:
                return GeneType.SURVIVAL_GENE
            elif 'profit' in desc or 'revenue' in desc or 'financial' in desc:
                return GeneType.PROFIT_GENE
            elif 'growth' in desc or 'expand' in desc:
                return GeneType.GROWTH_GENE
            elif 'legitimacy' in desc or 'reputation' in desc:
                return GeneType.LEGITIMACY_GENE
            elif 'power' in desc or 'influence' in desc:
                return GeneType.POWER_GENE
            elif 'compliance' in desc or 'regulation' in desc:
                return GeneType.COMPLIANCE_GENE
            elif 'innovation' in desc or 'technology' in desc:
                return GeneType.INNOVATION_GENE
            elif 'stability' in desc or 'maintain' in desc:
                return GeneType.STABILITY_GENE
        
        return GeneType.SURVIVAL_GENE  # Default
    
    def __repr__(self) -> str:
        return f"IntraGenomicConflictAnalyzer(gene_databases={len(self.gene_databases)}, conflicts={len(self.conflict_history)})"