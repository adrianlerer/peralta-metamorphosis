"""
Corporate Constructor - Models corporations as active legal phenotype constructors

Implements corporate entities as constructors that build legal structures
to serve their business interests through lobbying, regulatory capture,
litigation strategies, and standard-setting.

Based on extended phenotype theory: corporations construct legal environments
as extensions of their profit/market dominance "genes".
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass

from .constructor_base import (
    Constructor, ConstructorType, ConstructionStrategy,
    PowerResource, InterestGene, ConstructionCapability
)


@dataclass
class CorporateMetrics:
    """Corporate-specific metrics"""
    market_capitalization: float
    revenue: float
    regulatory_spending: float  # Annual spending on regulatory activities
    lobbying_expenditure: float
    legal_department_size: int
    external_legal_spend: float
    regulatory_violations: int  # Historical count
    successful_regulatory_outcomes: int


@dataclass
class MarketPosition:
    """Market position in specific sector"""
    sector: str
    market_share: float  # 0-1
    competitive_position: str  # "dominant", "major", "niche", "emerging"
    regulatory_influence_level: float  # 0-1


class CorporateConstructor(Constructor):
    """
    Corporate constructor of legal phenotypes
    
    Models how corporations actively construct regulatory environments
    to serve their business interests, implementing extended phenotype
    theory for corporate legal strategy.
    """
    
    def __init__(
        self,
        constructor_id: str,
        name: str,
        corporate_metrics: CorporateMetrics,
        market_positions: List[MarketPosition],
        business_model: str,
        primary_jurisdictions: List[str],
        regulatory_strategy: str = "defensive",
        **kwargs
    ):
        """
        Initialize corporate constructor
        
        Args:
            constructor_id: Unique corporate identifier
            name: Corporate name
            corporate_metrics: Financial and operational metrics
            market_positions: Market positions by sector
            business_model: Description of business model
            primary_jurisdictions: Key operating jurisdictions
            regulatory_strategy: Overall regulatory approach
        """
        
        # Calculate power index from corporate metrics
        power_index = self._calculate_corporate_power_index(corporate_metrics, market_positions)
        
        # Generate corporate interest genome
        interests_genome = self._generate_corporate_interests(
            corporate_metrics, market_positions, business_model
        )
        
        # Generate power resources
        power_resources = self._generate_corporate_power_resources(
            corporate_metrics, market_positions
        )
        
        # Generate construction capabilities
        construction_capabilities = self._generate_corporate_capabilities(
            corporate_metrics, market_positions, primary_jurisdictions
        )
        
        super().__init__(
            constructor_id=constructor_id,
            constructor_type=ConstructorType.CORPORATE,
            name=name,
            power_index=power_index,
            interests_genome=interests_genome,
            power_resources=power_resources,
            construction_capabilities=construction_capabilities,
            geographic_scope=primary_jurisdictions,
            **kwargs
        )
        
        self.corporate_metrics = corporate_metrics
        self.market_positions = market_positions
        self.business_model = business_model
        self.regulatory_strategy = regulatory_strategy
        
        # Corporate-specific tracking
        self.regulatory_wins: List[Dict[str, Any]] = []
        self.regulatory_losses: List[Dict[str, Any]] = []
        self.active_lobbying_campaigns: List[Dict[str, Any]] = []
        
    def _calculate_corporate_power_index(
        self,
        metrics: CorporateMetrics,
        positions: List[MarketPosition]
    ) -> float:
        """Calculate overall corporate power index"""
        
        # Financial power component (0-0.4)
        financial_power = min(0.4, (
            (metrics.market_capitalization / 1e12) * 0.2 +  # Trillion dollar scale
            (metrics.revenue / 1e11) * 0.1 +  # Hundred billion scale
            (metrics.regulatory_spending / 1e8) * 0.1  # Hundred million scale
        ))
        
        # Market dominance component (0-0.4)
        market_power = 0.0
        if positions:
            avg_market_share = sum(pos.market_share for pos in positions) / len(positions)
            dominant_positions = sum(1 for pos in positions if pos.market_share > 0.3)
            market_power = min(0.4, avg_market_share * 0.3 + (dominant_positions / len(positions)) * 0.1)
        
        # Regulatory influence component (0-0.2)
        regulatory_power = 0.0
        if positions:
            avg_regulatory_influence = sum(pos.regulatory_influence_level for pos in positions) / len(positions)
            regulatory_power = min(0.2, avg_regulatory_influence * 0.2)
        
        return min(1.0, financial_power + market_power + regulatory_power)
    
    def _generate_corporate_interests(
        self,
        metrics: CorporateMetrics,
        positions: List[MarketPosition],
        business_model: str
    ) -> List[InterestGene]:
        """Generate corporate interest genome"""
        
        interests = []
        
        # Core profit maximization gene (always present)
        interests.append(InterestGene(
            gene_id="profit_maximization",
            description="Maximize shareholder value and profitability",
            priority=0.9,
            temporal_stability=0.95
        ))
        
        # Market dominance gene
        market_dominance_priority = sum(pos.market_share for pos in positions) / max(len(positions), 1)
        interests.append(InterestGene(
            gene_id="market_dominance",
            description="Maintain and expand market position",
            priority=market_dominance_priority,
            temporal_stability=0.8
        ))
        
        # Regulatory arbitrage gene
        reg_arbitrage_priority = min(0.8, metrics.regulatory_spending / metrics.revenue * 10)
        interests.append(InterestGene(
            gene_id="regulatory_arbitrage",
            description="Optimize regulatory environment for competitive advantage",
            priority=reg_arbitrage_priority,
            temporal_stability=0.7
        ))
        
        # Business model specific interests
        if "tech" in business_model.lower():
            interests.append(InterestGene(
                gene_id="innovation_protection",
                description="Protect intellectual property and innovation advantages",
                priority=0.8,
                temporal_stability=0.6
            ))
            interests.append(InterestGene(
                gene_id="data_monetization",
                description="Maximize data collection and monetization opportunities",
                priority=0.7,
                temporal_stability=0.5
            ))
        
        if "financial" in business_model.lower():
            interests.append(InterestGene(
                gene_id="systemic_risk_socialization",
                description="Socialize risks while privatizing profits",
                priority=0.6,
                temporal_stability=0.8
            ))
        
        if "extractive" in business_model.lower():
            interests.append(InterestGene(
                gene_id="resource_access",
                description="Secure access to natural resources",
                priority=0.8,
                temporal_stability=0.9
            ))
            interests.append(InterestGene(
                gene_id="environmental_cost_externalization",
                description="Minimize internalization of environmental costs",
                priority=0.7,
                temporal_stability=0.6
            ))
        
        return interests
    
    def _generate_corporate_power_resources(
        self,
        metrics: CorporateMetrics,
        positions: List[MarketPosition]
    ) -> Dict[str, PowerResource]:
        """Generate corporate power resources"""
        
        resources = {}
        
        # Financial resources
        resources["financial"] = PowerResource(
            resource_type="financial",
            amount=metrics.regulatory_spending,
            sustainability=0.8,
            temporal_availability=(datetime.now(), None)
        )
        
        # Legal expertise
        resources["legal_expertise"] = PowerResource(
            resource_type="legal_expertise",
            amount=metrics.legal_department_size + (metrics.external_legal_spend / 1e6),
            sustainability=0.9,
            temporal_availability=(datetime.now(), None)
        )
        
        # Market influence
        avg_market_influence = sum(pos.market_share for pos in positions) / max(len(positions), 1)
        resources["market_influence"] = PowerResource(
            resource_type="market_influence",
            amount=avg_market_influence,
            sustainability=0.6,
            temporal_availability=(datetime.now(), None)
        )
        
        # Lobbying capacity
        resources["lobbying_capacity"] = PowerResource(
            resource_type="lobbying_capacity",
            amount=metrics.lobbying_expenditure / 1e6,  # Normalize to millions
            sustainability=0.7,
            temporal_availability=(datetime.now(), None)
        )
        
        return resources
    
    def _generate_corporate_capabilities(
        self,
        metrics: CorporateMetrics,
        positions: List[MarketPosition],
        jurisdictions: List[str]
    ) -> Dict[str, ConstructionCapability]:
        """Generate corporate construction capabilities"""
        
        capabilities = {}
        
        # Base success rate from historical performance
        base_success_rate = (
            metrics.successful_regulatory_outcomes / 
            max(metrics.successful_regulatory_outcomes + metrics.regulatory_violations, 1)
        )
        
        # Corporate law capability (strongest for all corporations)
        capabilities["corporate_law"] = ConstructionCapability(
            domain="corporate_law",
            expertise_level=0.8,
            resource_efficiency=0.7,
            historical_success_rate=base_success_rate,
            network_strength=0.8
        )
        
        # Sector-specific capabilities
        for position in positions:
            sector = position.sector
            capabilities[f"{sector}_regulation"] = ConstructionCapability(
                domain=f"{sector}_regulation",
                expertise_level=position.market_share,
                resource_efficiency=position.regulatory_influence_level,
                historical_success_rate=base_success_rate,
                network_strength=position.market_share * 0.8
            )
        
        # International law capability (for multinationals)
        if len(jurisdictions) > 3:
            capabilities["international_law"] = ConstructionCapability(
                domain="international_law",
                expertise_level=0.6,
                resource_efficiency=0.5,
                historical_success_rate=base_success_rate * 0.7,  # Lower success internationally
                network_strength=0.6
            )
        
        return capabilities
    
    def construct_phenotype(
        self,
        target_area: str,
        environmental_pressure: Dict[str, float],
        strategy: ConstructionStrategy = None
    ) -> 'LegalPhenotype':
        """
        Construct new legal phenotype through corporate action
        
        Args:
            target_area: Legal domain to target
            environmental_pressure: Current environmental pressures
            strategy: Construction strategy (defaults to corporate-optimal)
            
        Returns:
            Constructed legal phenotype
        """
        from ..phenotypes.legal_phenotype import LegalPhenotype, PhenotypeType
        
        if not strategy:
            strategy = self._select_optimal_corporate_strategy(target_area, environmental_pressure)
        
        # Calculate construction fitness
        fitness = self.calculate_construction_fitness(
            target_area, environmental_pressure, self._get_available_resources()
        )
        
        # Estimate resource investment needed
        resource_investment = self._calculate_resource_investment(
            target_area, strategy, fitness
        )
        
        # Create phenotype
        phenotype = LegalPhenotype(
            phenotype_id=f"{self.constructor_id}_{target_area}_{datetime.now().strftime('%Y%m%d')}",
            constructor=self,
            target_domain=target_area,
            phenotype_type=PhenotypeType.REGULATORY,
            construction_strategy=strategy,
            resource_investment=resource_investment,
            expected_fitness=fitness,
            construction_timestamp=datetime.now()
        )
        
        # Add to portfolio
        self.phenotype_portfolio.append(phenotype)
        
        # Record construction event
        self.construction_history.append({
            'timestamp': datetime.now(),
            'phenotype_id': phenotype.phenotype_id,
            'target_area': target_area,
            'strategy': strategy,
            'resource_investment': resource_investment,
            'expected_fitness': fitness
        })
        
        return phenotype
    
    def modify_environment(
        self,
        legal_landscape: 'LegalLandscape',
        intervention_intensity: float,
        target_changes: Dict[str, float]
    ) -> 'LegalLandscape':
        """
        Modify legal environment through corporate intervention
        
        Args:
            legal_landscape: Current legal landscape
            intervention_intensity: Intensity of intervention (0-1)
            target_changes: Desired changes by category
            
        Returns:
            Modified legal landscape
        """
        # Calculate intervention capacity
        intervention_capacity = self._calculate_intervention_capacity(intervention_intensity)
        
        # Apply corporate influence to landscape
        modified_landscape = legal_landscape.copy()
        
        for change_category, target_change in target_changes.items():
            # Calculate achievable change based on corporate power and strategy
            achievable_change = min(
                target_change,
                intervention_capacity * self._get_category_influence(change_category)
            )
            
            # Apply change to landscape
            modified_landscape = modified_landscape.apply_change(
                change_category, achievable_change, source=self
            )
        
        return modified_landscape
    
    def _select_optimal_corporate_strategy(
        self,
        target_area: str,
        environmental_pressure: Dict[str, float]
    ) -> ConstructionStrategy:
        """Select optimal construction strategy for corporate interests"""
        
        # Strategy scoring based on corporate characteristics
        strategy_scores = {
            ConstructionStrategy.LOBBYING: 0.0,
            ConstructionStrategy.REGULATORY_CAPTURE: 0.0,
            ConstructionStrategy.LITIGATION: 0.0,
            ConstructionStrategy.ECONOMIC_COERCION: 0.0,
            ConstructionStrategy.STANDARD_SETTING: 0.0
        }
        
        # Score based on available resources
        lobbying_capacity = self.power_resources.get("lobbying_capacity")
        if lobbying_capacity and lobbying_capacity.amount > 10:  # >$10M lobbying capacity
            strategy_scores[ConstructionStrategy.LOBBYING] += 0.3
        
        legal_capacity = self.power_resources.get("legal_expertise") 
        if legal_capacity and legal_capacity.amount > 50:  # Substantial legal team
            strategy_scores[ConstructionStrategy.LITIGATION] += 0.3
        
        market_influence = self.power_resources.get("market_influence")
        if market_influence and market_influence.amount > 0.3:  # >30% market influence
            strategy_scores[ConstructionStrategy.ECONOMIC_COERCION] += 0.4
            strategy_scores[ConstructionStrategy.STANDARD_SETTING] += 0.3
        
        # Score based on environmental factors
        if environmental_pressure.get("regulatory_uncertainty", 0) > 0.5:
            strategy_scores[ConstructionStrategy.REGULATORY_CAPTURE] += 0.4
        
        if environmental_pressure.get("public_scrutiny", 0) < 0.3:
            strategy_scores[ConstructionStrategy.REGULATORY_CAPTURE] += 0.2
        else:
            strategy_scores[ConstructionStrategy.LOBBYING] += 0.2
        
        # Return highest scoring strategy
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def _get_available_resources(self) -> Dict[str, float]:
        """Get currently available resources"""
        return {
            resource_type: resource.amount
            for resource_type, resource in self.power_resources.items()
        }
    
    def _calculate_resource_investment(
        self,
        target_area: str,
        strategy: ConstructionStrategy,
        expected_fitness: float
    ) -> Dict[str, float]:
        """Calculate resource investment for construction"""
        
        base_investment = {
            resource_type: resource.amount * 0.1  # 10% base investment
            for resource_type, resource in self.power_resources.items()
        }
        
        # Adjust based on expected fitness (higher fitness = higher investment)
        fitness_multiplier = 0.5 + expected_fitness  # 0.5-1.5 range
        
        # Adjust based on strategy requirements
        strategy_multipliers = {
            ConstructionStrategy.LOBBYING: {"lobbying_capacity": 2.0, "financial": 1.5},
            ConstructionStrategy.LITIGATION: {"legal_expertise": 2.5, "financial": 2.0},
            ConstructionStrategy.REGULATORY_CAPTURE: {"financial": 3.0, "market_influence": 1.5},
            ConstructionStrategy.ECONOMIC_COERCION: {"market_influence": 2.0, "financial": 1.5},
            ConstructionStrategy.STANDARD_SETTING: {"market_influence": 1.5, "legal_expertise": 1.5}
        }
        
        strategy_mods = strategy_multipliers.get(strategy, {})
        
        investment = {}
        for resource_type, base_amount in base_investment.items():
            strategy_mod = strategy_mods.get(resource_type, 1.0)
            investment[resource_type] = base_amount * fitness_multiplier * strategy_mod
        
        return investment
    
    def _calculate_intervention_capacity(self, intensity: float) -> float:
        """Calculate intervention capacity based on intensity and corporate power"""
        return min(1.0, intensity * self.power_index * 1.2)  # Corporate bonus
    
    def _get_category_influence(self, category: str) -> float:
        """Get corporate influence in specific category"""
        influence_map = {
            "regulatory_framework": 0.8,
            "enforcement_intensity": 0.6,
            "judicial_interpretation": 0.4,
            "legislative_priority": 0.7,
            "public_opinion": 0.3
        }
        
        return influence_map.get(category, 0.5)
    
    def calculate_regulatory_roi(
        self,
        regulatory_investment: Dict[str, float],
        expected_outcomes: Dict[str, float],
        time_horizon: int = 5
    ) -> float:
        """
        Calculate return on investment for regulatory activities
        
        Args:
            regulatory_investment: Investment by resource type
            expected_outcomes: Expected outcomes by category
            time_horizon: Time horizon for ROI calculation
            
        Returns:
            ROI estimate
        """
        total_investment = sum(regulatory_investment.values())
        
        # Estimate financial returns from regulatory outcomes
        outcome_values = {
            "tax_reduction": 0.1 * self.corporate_metrics.revenue,
            "compliance_cost_reduction": 0.05 * self.corporate_metrics.revenue,
            "market_access": 0.15 * self.corporate_metrics.revenue,
            "competitive_advantage": 0.08 * self.corporate_metrics.revenue
        }
        
        total_expected_value = sum(
            expected_outcomes.get(outcome, 0) * value
            for outcome, value in outcome_values.items()
        )
        
        # Annualized over time horizon
        annual_value = total_expected_value / time_horizon
        
        # ROI calculation
        if total_investment > 0:
            return (annual_value - total_investment) / total_investment
        else:
            return 0.0
    
    def __repr__(self) -> str:
        return (f"CorporateConstructor({self.constructor_id}: {self.name}, "
                f"power={self.power_index:.2f}, "
                f"market_cap=${self.corporate_metrics.market_capitalization/1e9:.1f}B)")