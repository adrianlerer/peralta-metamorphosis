"""
Legal Phenotype - Base class for extended legal phenotypes

Implements the core concept of legal structures as extended phenotypes
in Dawkins' evolutionary framework applied to legal systems.

Legal phenotypes are structures constructed by entities (constructors) 
to extend their influence into the legal environment, serving their
fundamental interests (genes).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import uuid


class PhenotypeType(Enum):
    """Types of legal phenotypes"""
    REGULATORY = "regulatory"
    INSTITUTIONAL = "institutional" 
    DOCTRINAL = "doctrinal"
    LEGISLATIVE = "legislative"
    JUDICIAL = "judicial"
    PROCEDURAL = "procedural"


class PhenotypeStatus(Enum):
    """Status of legal phenotype"""
    PROPOSED = "proposed"
    UNDER_CONSTRUCTION = "under_construction"
    ACTIVE = "active"
    DECLINING = "declining"
    EXTINCT = "extinct"
    MUTATED = "mutated"


@dataclass
class FitnessMetrics:
    """Fitness metrics for legal phenotype"""
    current_fitness: float  # Overall current fitness (0-1)
    institutional_support: float  # Support from institutions
    public_acceptance: float  # Public acceptance level
    enforcement_effectiveness: float  # How well it's enforced
    resistance_level: float  # Level of resistance faced
    adaptation_capacity: float  # Ability to adapt to changes
    
    def calculate_overall_fitness(self) -> float:
        """Calculate overall fitness score"""
        weights = {
            'institutional_support': 0.25,
            'public_acceptance': 0.15,
            'enforcement_effectiveness': 0.30,
            'resistance_level': -0.20,  # Negative weight
            'adaptation_capacity': 0.15
        }
        
        fitness = (
            weights['institutional_support'] * self.institutional_support +
            weights['public_acceptance'] * self.public_acceptance +
            weights['enforcement_effectiveness'] * self.enforcement_effectiveness +
            weights['resistance_level'] * (1 - self.resistance_level) +  # Invert resistance
            weights['adaptation_capacity'] * self.adaptation_capacity
        )
        
        return max(0.0, min(1.0, fitness))


@dataclass
class PhenotypeEffect:
    """Effect of phenotype on legal environment"""
    effect_type: str
    target_domain: str
    magnitude: float  # -1 to 1 scale
    temporal_pattern: str  # "immediate", "gradual", "delayed", "cyclical"
    affected_actors: List[str]
    side_effects: Dict[str, float] = field(default_factory=dict)


@dataclass
class MutationEvent:
    """Record of phenotype mutation"""
    mutation_id: str
    timestamp: datetime
    mutation_type: str  # "adaptation", "drift", "selection_pressure", "constructor_modification"
    original_features: Dict[str, Any]
    new_features: Dict[str, Any]
    triggering_pressure: Optional[str]
    fitness_change: float


class LegalPhenotype(ABC):
    """
    Base class for all legal phenotypes
    
    Legal phenotypes are extended structures that constructors build
    to influence their legal environment and serve their interests.
    """
    
    def __init__(
        self,
        phenotype_id: str,
        constructor: 'Constructor',
        target_domain: str,
        phenotype_type: PhenotypeType,
        construction_strategy: 'ConstructionStrategy',
        resource_investment: Dict[str, float],
        expected_fitness: float,
        construction_timestamp: datetime = None,
        name: str = None,
        description: str = None
    ):
        """
        Initialize legal phenotype
        
        Args:
            phenotype_id: Unique identifier
            constructor: Constructor that created this phenotype
            target_domain: Legal domain being targeted
            phenotype_type: Type of phenotype
            construction_strategy: Strategy used for construction
            resource_investment: Resources invested by constructor
            expected_fitness: Expected fitness at creation
            construction_timestamp: When phenotype was constructed
            name: Human-readable name
            description: Description of phenotype
        """
        self.phenotype_id = phenotype_id
        self.constructor = constructor
        self.target_domain = target_domain
        self.phenotype_type = phenotype_type
        self.construction_strategy = construction_strategy
        self.resource_investment = resource_investment
        self.expected_fitness = expected_fitness
        self.construction_timestamp = construction_timestamp or datetime.now()
        self.name = name or f"Phenotype_{phenotype_id}"
        self.description = description or f"{phenotype_type.value} phenotype in {target_domain}"
        
        # Status and fitness tracking
        self.status = PhenotypeStatus.PROPOSED
        self.current_fitness = 0.0
        self.fitness_history: List[Tuple[datetime, float]] = []
        self.fitness_metrics: Optional[FitnessMetrics] = None
        
        # Environmental interaction
        self.phenotype_effects: List[PhenotypeEffect] = []
        self.environmental_dependencies: Dict[str, float] = {}
        self.competitor_phenotypes: List[str] = []  # IDs of competing phenotypes
        
        # Evolution tracking
        self.mutation_history: List[MutationEvent] = []
        self.parent_phenotypes: List[str] = []  # IDs of parent phenotypes
        self.descendant_phenotypes: List[str] = []  # IDs of descendant phenotypes
        
        # Performance metrics
        self.effectiveness_metrics: Dict[str, float] = {}
        self.cost_benefit_ratio: Optional[float] = None
        self.stakeholder_impacts: Dict[str, float] = {}
    
    def activate(self, legal_landscape: 'LegalLandscape') -> bool:
        """
        Activate the phenotype in the legal landscape
        
        Args:
            legal_landscape: Target legal landscape
            
        Returns:
            Success of activation
        """
        if self.status != PhenotypeStatus.PROPOSED:
            return False
        
        # Check if phenotype can be successfully introduced
        activation_success = self._calculate_activation_success(legal_landscape)
        
        if activation_success > 0.5:  # Threshold for successful activation
            self.status = PhenotypeStatus.ACTIVE
            self.current_fitness = self._calculate_initial_fitness(legal_landscape)
            self.fitness_history.append((datetime.now(), self.current_fitness))
            
            # Apply phenotype effects to landscape
            for effect in self.phenotype_effects:
                legal_landscape.apply_phenotype_effect(effect, self)
            
            return True
        else:
            self.status = PhenotypeStatus.EXTINCT
            return False
    
    def update_fitness(self, legal_landscape: 'LegalLandscape') -> float:
        """
        Update phenotype fitness based on current landscape
        
        Args:
            legal_landscape: Current legal landscape
            
        Returns:
            Updated fitness value
        """
        if self.status not in [PhenotypeStatus.ACTIVE, PhenotypeStatus.DECLINING]:
            return 0.0
        
        # Calculate new fitness metrics
        new_fitness_metrics = self._calculate_fitness_metrics(legal_landscape)
        self.fitness_metrics = new_fitness_metrics
        
        # Update current fitness
        new_fitness = new_fitness_metrics.calculate_overall_fitness()
        self.current_fitness = new_fitness
        
        # Record fitness history
        self.fitness_history.append((datetime.now(), new_fitness))
        
        # Update status based on fitness trends
        self._update_status_from_fitness_trend()
        
        return new_fitness
    
    def mutate(
        self,
        mutation_trigger: str,
        environmental_pressure: Dict[str, float],
        mutation_intensity: float = 0.1
    ) -> 'LegalPhenotype':
        """
        Create mutated version of phenotype in response to pressure
        
        Args:
            mutation_trigger: What triggered the mutation
            environmental_pressure: Current environmental pressures
            mutation_intensity: Intensity of mutation (0-1)
            
        Returns:
            Mutated phenotype instance
        """
        # Generate mutation
        mutation_features = self._generate_mutation_features(
            environmental_pressure, mutation_intensity
        )
        
        # Create mutation record
        mutation_event = MutationEvent(
            mutation_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            mutation_type=mutation_trigger,
            original_features=self._extract_features(),
            new_features=mutation_features,
            triggering_pressure=str(environmental_pressure),
            fitness_change=0.0  # Will be calculated after mutation
        )
        
        # Create mutated phenotype
        mutated_phenotype = self._create_mutated_instance(mutation_features)
        mutated_phenotype.parent_phenotypes.append(self.phenotype_id)
        mutated_phenotype.mutation_history.append(mutation_event)
        
        # Update this phenotype's records
        self.descendant_phenotypes.append(mutated_phenotype.phenotype_id)
        self.mutation_history.append(mutation_event)
        
        return mutated_phenotype
    
    def calculate_competitive_pressure(
        self,
        competitor_phenotypes: List['LegalPhenotype']
    ) -> Dict[str, float]:
        """
        Calculate competitive pressure from other phenotypes
        
        Args:
            competitor_phenotypes: List of competing phenotypes
            
        Returns:
            Competitive pressure scores by competitor
        """
        pressure_scores = {}
        
        for competitor in competitor_phenotypes:
            if competitor.phenotype_id == self.phenotype_id:
                continue
                
            # Calculate overlap in target domain
            domain_overlap = self._calculate_domain_overlap(competitor)
            
            # Calculate resource competition
            resource_competition = self._calculate_resource_competition(competitor)
            
            # Calculate fitness differential
            fitness_differential = competitor.current_fitness - self.current_fitness
            
            # Combined competitive pressure
            competitive_pressure = (
                domain_overlap * 0.4 +
                resource_competition * 0.3 +
                max(0, fitness_differential) * 0.3
            )
            
            pressure_scores[competitor.phenotype_id] = competitive_pressure
        
        return pressure_scores
    
    def predict_survival_probability(
        self,
        future_landscape: 'LegalLandscape',
        time_horizon: int = 5,
        competing_phenotypes: List['LegalPhenotype'] = None
    ) -> float:
        """
        Predict probability of phenotype survival in future landscape
        
        Args:
            future_landscape: Projected future legal landscape
            time_horizon: Years into future
            competing_phenotypes: List of competing phenotypes
            
        Returns:
            Survival probability (0-1)
        """
        competing_phenotypes = competing_phenotypes or []
        
        # Base survival from current fitness
        base_survival = self.current_fitness ** 0.5  # Square root for diminishing returns
        
        # Environmental compatibility in future
        env_compatibility = self._calculate_future_environment_compatibility(future_landscape)
        
        # Constructor support sustainability
        constructor_support = self.constructor.predict_phenotype_survival(
            self, future_landscape, time_horizon
        )
        
        # Competitive pressure factor
        competitive_pressure = sum(
            self.calculate_competitive_pressure(competing_phenotypes).values()
        )
        competitive_factor = max(0.1, 1.0 - competitive_pressure * 0.1)
        
        # Adaptation capacity
        adaptation_factor = self.fitness_metrics.adaptation_capacity if self.fitness_metrics else 0.5
        
        # Time decay factor
        decay_factor = max(0.1, 1.0 - (0.05 * time_horizon))
        
        # Combined survival probability
        survival_probability = (
            base_survival * env_compatibility * constructor_support * 
            competitive_factor * adaptation_factor * decay_factor
        )
        
        return min(1.0, max(0.0, survival_probability))
    
    @abstractmethod
    def _calculate_activation_success(self, legal_landscape: 'LegalLandscape') -> float:
        """Calculate probability of successful activation"""
        pass
    
    @abstractmethod
    def _calculate_initial_fitness(self, legal_landscape: 'LegalLandscape') -> float:
        """Calculate initial fitness upon activation"""
        pass
    
    @abstractmethod
    def _calculate_fitness_metrics(self, legal_landscape: 'LegalLandscape') -> FitnessMetrics:
        """Calculate detailed fitness metrics"""
        pass
    
    @abstractmethod
    def _generate_mutation_features(
        self, 
        environmental_pressure: Dict[str, float],
        mutation_intensity: float
    ) -> Dict[str, Any]:
        """Generate features for mutated phenotype"""
        pass
    
    @abstractmethod
    def _create_mutated_instance(self, mutation_features: Dict[str, Any]) -> 'LegalPhenotype':
        """Create new instance with mutated features"""
        pass
    
    def _extract_features(self) -> Dict[str, Any]:
        """Extract current phenotype features for mutation tracking"""
        return {
            'target_domain': self.target_domain,
            'phenotype_type': self.phenotype_type.value,
            'construction_strategy': self.construction_strategy.value,
            'current_fitness': self.current_fitness,
            'status': self.status.value
        }
    
    def _calculate_domain_overlap(self, other_phenotype: 'LegalPhenotype') -> float:
        """Calculate overlap in target domains with another phenotype"""
        if self.target_domain == other_phenotype.target_domain:
            return 1.0
        
        # Could implement more sophisticated domain similarity logic
        domain_similarities = {
            ('corporate_law', 'securities_regulation'): 0.7,
            ('environmental_law', 'energy_regulation'): 0.6,
            ('tax_law', 'corporate_law'): 0.5,
            ('constitutional_law', 'administrative_law'): 0.4
        }
        
        domain_pair = tuple(sorted([self.target_domain, other_phenotype.target_domain]))
        return domain_similarities.get(domain_pair, 0.1)
    
    def _calculate_resource_competition(self, other_phenotype: 'LegalPhenotype') -> float:
        """Calculate resource competition with another phenotype"""
        # If same constructor, no resource competition
        if self.constructor.constructor_id == other_phenotype.constructor.constructor_id:
            return 0.0
        
        # Calculate overlap in required resources
        self_resources = set(self.resource_investment.keys())
        other_resources = set(other_phenotype.resource_investment.keys())
        
        resource_overlap = len(self_resources.intersection(other_resources))
        total_resources = len(self_resources.union(other_resources))
        
        if total_resources == 0:
            return 0.0
        
        return resource_overlap / total_resources
    
    def _calculate_future_environment_compatibility(
        self, 
        future_landscape: 'LegalLandscape'
    ) -> float:
        """Calculate compatibility with future legal landscape"""
        # Simplified compatibility calculation
        # Would implement detailed landscape compatibility logic
        return 0.7
    
    def _update_status_from_fitness_trend(self):
        """Update phenotype status based on fitness trend"""
        if len(self.fitness_history) < 2:
            return
        
        # Look at recent fitness trend
        recent_fitness = [f[1] for f in self.fitness_history[-3:]]
        
        if len(recent_fitness) >= 2:
            trend = recent_fitness[-1] - recent_fitness[0]
            
            if self.current_fitness < 0.1:
                self.status = PhenotypeStatus.EXTINCT
            elif trend < -0.1 and self.current_fitness < 0.3:
                self.status = PhenotypeStatus.DECLINING
            elif self.current_fitness > 0.3:
                self.status = PhenotypeStatus.ACTIVE
    
    def get_effectiveness_score(self) -> float:
        """Calculate overall effectiveness score for the phenotype"""
        if not self.fitness_metrics:
            return 0.0
        
        # Effectiveness combines fitness with cost-benefit ratio
        fitness_component = self.fitness_metrics.calculate_overall_fitness()
        
        # Cost-benefit component
        if self.cost_benefit_ratio is not None:
            cb_component = min(1.0, max(0.0, self.cost_benefit_ratio))
        else:
            cb_component = 0.5  # Neutral if unknown
        
        return (fitness_component * 0.7) + (cb_component * 0.3)
    
    def __repr__(self) -> str:
        return (f"LegalPhenotype({self.phenotype_id}: {self.name}, "
                f"type={self.phenotype_type.value}, "
                f"fitness={self.current_fitness:.2f}, "
                f"status={self.status.value})")
    
    def __str__(self) -> str:
        return f"{self.name} ({self.phenotype_type.value}, fitness: {self.current_fitness:.2f})"