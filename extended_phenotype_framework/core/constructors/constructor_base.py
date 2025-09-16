"""
Base Constructor Class - Foundational class for legal phenotype constructors

This implements the core concept from Dawkins' extended phenotype theory:
entities that actively construct legal structures as extensions of their interests/memes.

Based on the comprehensive analysis by Ignacio Adrián Lerer applying 
Dawkins' 2024 concepts to legal systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import uuid


class ConstructorType(Enum):
    """Types of legal constructors"""
    CORPORATE = "corporate"
    STATE = "state"
    TECHNOLOGY = "technology"
    INTERNATIONAL = "international"
    CIVIL_SOCIETY = "civil_society"
    HYBRID = "hybrid"


class ConstructionStrategy(Enum):
    """Strategies for constructing legal phenotypes"""
    LOBBYING = "lobbying"
    REGULATORY_CAPTURE = "regulatory_capture"
    LITIGATION = "litigation"
    ACADEMIC_INFLUENCE = "academic_influence"
    PUBLIC_PRESSURE = "public_pressure"
    ECONOMIC_COERCION = "economic_coercion"
    TREATY_MAKING = "treaty_making"
    STANDARD_SETTING = "standard_setting"


@dataclass
class PowerResource:
    """Power resource available to a constructor"""
    resource_type: str
    amount: float
    sustainability: float  # 0-1 scale
    temporal_availability: Tuple[datetime, Optional[datetime]]
    
    
@dataclass
class InterestGene:
    """Fundamental interest/"gene" of a constructor"""
    gene_id: str
    description: str
    priority: float  # 0-1 scale
    temporal_stability: float  # How stable this interest is over time
    compatibility_matrix: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConstructionCapability:
    """Capability for legal construction in specific domains"""
    domain: str
    expertise_level: float  # 0-1 scale
    resource_efficiency: float  # How efficiently resources convert to results
    historical_success_rate: float
    network_strength: float  # Strength of influence networks in this domain


class Constructor(ABC):
    """
    Base class for all legal phenotype constructors
    
    Constructors are entities that actively build legal structures 
    as extensions of their fundamental interests (genes).
    """
    
    def __init__(
        self,
        constructor_id: str,
        constructor_type: ConstructorType,
        name: str,
        power_index: float,
        interests_genome: List[InterestGene],
        power_resources: Dict[str, PowerResource],
        construction_capabilities: Dict[str, ConstructionCapability],
        geographic_scope: List[str] = None,
        temporal_scope: Tuple[datetime, Optional[datetime]] = None
    ):
        """
        Initialize a constructor
        
        Args:
            constructor_id: Unique identifier
            constructor_type: Type of constructor (corporate, state, etc.)
            name: Human-readable name
            power_index: Overall power index (0-1)
            interests_genome: List of fundamental interests
            power_resources: Available power resources by type
            construction_capabilities: Capabilities by legal domain
            geographic_scope: Geographic reach (jurisdictions)
            temporal_scope: Time period of activity
        """
        self.constructor_id = constructor_id
        self.constructor_type = constructor_type
        self.name = name
        self.power_index = power_index
        self.interests_genome = interests_genome
        self.power_resources = power_resources
        self.construction_capabilities = construction_capabilities
        self.geographic_scope = geographic_scope or []
        self.temporal_scope = temporal_scope or (datetime.now(), None)
        
        # Track constructed phenotypes
        self.phenotype_portfolio: List['LegalPhenotype'] = []
        self.construction_history: List[Dict[str, Any]] = []
        
        # Internal state
        self._gene_expression_weights: Dict[str, float] = {}
        self._update_gene_expression_weights()
    
    def _update_gene_expression_weights(self):
        """Update gene expression weights based on current environment"""
        total_priority = sum(gene.priority for gene in self.interests_genome)
        for gene in self.interests_genome:
            self._gene_expression_weights[gene.gene_id] = gene.priority / total_priority
    
    def calculate_construction_fitness(
        self, 
        target_domain: str,
        environmental_pressure: Dict[str, float],
        resource_budget: Dict[str, float]
    ) -> float:
        """
        Calculate expected fitness of a construction in target domain
        
        Args:
            target_domain: Legal domain for construction
            environmental_pressure: Current environmental pressures
            resource_budget: Available resources for construction
            
        Returns:
            Expected fitness score (0-1)
        """
        if target_domain not in self.construction_capabilities:
            return 0.0
        
        capability = self.construction_capabilities[target_domain]
        
        # Base fitness from capability
        base_fitness = capability.expertise_level * capability.historical_success_rate
        
        # Resource adequacy factor
        total_needed = sum(
            self._estimate_resource_need(target_domain, res_type) 
            for res_type in resource_budget.keys()
        )
        total_available = sum(resource_budget.values())
        resource_factor = min(1.0, total_available / max(total_needed, 0.001))
        
        # Environmental alignment factor
        alignment_factor = self._calculate_environmental_alignment(
            target_domain, environmental_pressure
        )
        
        return base_fitness * resource_factor * alignment_factor
    
    def _estimate_resource_need(self, domain: str, resource_type: str) -> float:
        """Estimate resource need for construction in domain"""
        if domain not in self.construction_capabilities:
            return float('inf')
        
        capability = self.construction_capabilities[domain]
        base_need = 1.0 / max(capability.resource_efficiency, 0.001)
        
        # Adjust based on domain complexity
        domain_complexity_factors = {
            'constitutional_law': 1.5,
            'international_law': 1.3,
            'corporate_law': 1.0,
            'administrative_law': 0.8,
            'local_regulation': 0.6
        }
        
        complexity_factor = domain_complexity_factors.get(domain, 1.0)
        return base_need * complexity_factor
    
    def _calculate_environmental_alignment(
        self, 
        domain: str,
        environmental_pressure: Dict[str, float]
    ) -> float:
        """Calculate alignment between constructor interests and environment"""
        alignment_scores = []
        
        for gene in self.interests_genome:
            gene_weight = self._gene_expression_weights.get(gene.gene_id, 0.0)
            
            # Calculate how well this gene aligns with environmental pressure
            gene_alignment = 0.0
            for pressure_type, pressure_value in environmental_pressure.items():
                # This would be domain-specific logic
                alignment_contribution = self._calculate_gene_pressure_alignment(
                    gene, pressure_type, pressure_value, domain
                )
                gene_alignment += alignment_contribution
            
            alignment_scores.append(gene_alignment * gene_weight)
        
        return sum(alignment_scores) / max(len(alignment_scores), 1)
    
    def _calculate_gene_pressure_alignment(
        self,
        gene: InterestGene,
        pressure_type: str,
        pressure_value: float,
        domain: str
    ) -> float:
        """Calculate alignment between a gene and specific environmental pressure"""
        # This would contain domain-specific logic for gene-pressure alignment
        # For now, return a simplified calculation
        
        alignment_matrix = {
            ('fiscal_centralization', 'economic_crisis'): 0.8,
            ('regulatory_arbitrage', 'globalization'): 0.9,
            ('privacy_protection', 'tech_surveillance'): -0.7,
            ('market_efficiency', 'economic_liberalization'): 0.8,
        }
        
        key = (gene.description.lower(), pressure_type.lower())
        base_alignment = alignment_matrix.get(key, 0.0)
        
        return base_alignment * pressure_value
    
    @abstractmethod
    def construct_phenotype(
        self,
        target_area: str,
        environmental_pressure: Dict[str, float],
        strategy: ConstructionStrategy = None
    ) -> 'LegalPhenotype':
        """
        Construct a new legal phenotype
        
        Args:
            target_area: Legal area to target
            environmental_pressure: Current environmental pressures
            strategy: Construction strategy to use
            
        Returns:
            Constructed legal phenotype
        """
        pass
    
    @abstractmethod
    def modify_environment(
        self,
        legal_landscape: 'LegalLandscape',
        intervention_intensity: float,
        target_changes: Dict[str, float]
    ) -> 'LegalLandscape':
        """
        Modify the legal environment through constructor intervention
        
        Args:
            legal_landscape: Current legal landscape
            intervention_intensity: Intensity of intervention (0-1)
            target_changes: Desired changes by category
            
        Returns:
            Modified legal landscape
        """
        pass
    
    def allocate_resources(
        self,
        construction_projects: List[Dict[str, Any]],
        total_budget: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Allocate resources across multiple construction projects
        
        Args:
            construction_projects: List of construction project specifications
            total_budget: Total available budget by resource type
            
        Returns:
            Resource allocation by project
        """
        allocations = {}
        
        # Calculate fitness for each project
        project_fitnesses = {}
        for project in construction_projects:
            fitness = self.calculate_construction_fitness(
                project.get('target_domain', ''),
                project.get('environmental_pressure', {}),
                total_budget
            )
            project_fitnesses[project['id']] = fitness
        
        # Allocate resources proportionally to fitness
        total_fitness = sum(project_fitnesses.values())
        
        for project in construction_projects:
            project_id = project['id']
            if total_fitness > 0:
                fitness_share = project_fitnesses[project_id] / total_fitness
                allocations[project_id] = {
                    resource_type: amount * fitness_share
                    for resource_type, amount in total_budget.items()
                }
            else:
                allocations[project_id] = {
                    resource_type: 0.0
                    for resource_type in total_budget.keys()
                }
        
        return allocations
    
    def predict_phenotype_survival(
        self,
        phenotype: 'LegalPhenotype',
        future_landscape: 'LegalLandscape',
        time_horizon: int = 10
    ) -> float:
        """
        Predict survival probability of a phenotype in future landscape
        
        Args:
            phenotype: Legal phenotype to analyze
            future_landscape: Projected future legal landscape
            time_horizon: Prediction time horizon in years
            
        Returns:
            Survival probability (0-1)
        """
        # Base survival from phenotype fitness
        base_survival = phenotype.current_fitness
        
        # Environmental compatibility factor
        env_compatibility = self._calculate_phenotype_environment_compatibility(
            phenotype, future_landscape
        )
        
        # Constructor support factor
        support_factor = self._calculate_constructor_support_factor(
            phenotype, time_horizon
        )
        
        # Temporal decay factor
        decay_factor = max(0.1, 1.0 - (0.05 * time_horizon))
        
        survival_probability = (
            base_survival * env_compatibility * support_factor * decay_factor
        )
        
        return min(1.0, max(0.0, survival_probability))
    
    def _calculate_phenotype_environment_compatibility(
        self,
        phenotype: 'LegalPhenotype',
        landscape: 'LegalLandscape'
    ) -> float:
        """Calculate compatibility between phenotype and landscape"""
        # Simplified compatibility calculation
        return 0.7  # Would implement complex compatibility logic
    
    def _calculate_constructor_support_factor(
        self,
        phenotype: 'LegalPhenotype', 
        time_horizon: int
    ) -> float:
        """Calculate ongoing support factor for phenotype"""
        # Factors: resource sustainability, interest alignment, power projection
        
        # Resource sustainability
        resource_sustainability = sum(
            res.sustainability for res in self.power_resources.values()
        ) / max(len(self.power_resources), 1)
        
        # Interest alignment with phenotype
        interest_alignment = self._calculate_phenotype_interest_alignment(phenotype)
        
        # Power projection over time horizon
        power_projection = self._project_power_over_time(time_horizon)
        
        return (resource_sustainability + interest_alignment + power_projection) / 3
    
    def _calculate_phenotype_interest_alignment(self, phenotype: 'LegalPhenotype') -> float:
        """Calculate alignment between phenotype and constructor interests"""
        # Simplified alignment calculation
        return 0.8  # Would implement detailed interest-phenotype alignment logic
    
    def _project_power_over_time(self, time_horizon: int) -> float:
        """Project constructor power over specified time horizon"""
        # Simple linear decay model
        decay_rate = 0.02  # 2% per year
        projected_power = self.power_index * (1 - decay_rate) ** time_horizon
        return projected_power
    
    def get_active_genes(self, context: Dict[str, Any] = None) -> List[InterestGene]:
        """
        Get genes that are currently active given context
        
        Args:
            context: Environmental or situational context
            
        Returns:
            List of active genes
        """
        if not context:
            return self.interests_genome
        
        # Filter genes based on context relevance
        active_genes = []
        for gene in self.interests_genome:
            activation_score = self._calculate_gene_activation(gene, context)
            if activation_score > 0.3:  # Activation threshold
                active_genes.append(gene)
        
        return active_genes
    
    def _calculate_gene_activation(self, gene: InterestGene, context: Dict[str, Any]) -> float:
        """Calculate gene activation score given context"""
        # Simplified activation calculation
        base_activation = gene.priority
        
        # Context modifiers
        for context_key, context_value in context.items():
            if context_key in gene.compatibility_matrix:
                base_activation *= (1 + gene.compatibility_matrix[context_key] * context_value)
        
        return min(1.0, max(0.0, base_activation))
    
    def __repr__(self) -> str:
        return f"Constructor({self.constructor_id}: {self.name}, power={self.power_index:.2f})"
    
    def __str__(self) -> str:
        return f"{self.name} ({self.constructor_type.value}, power: {self.power_index:.2f})"