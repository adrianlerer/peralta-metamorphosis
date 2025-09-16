"""
Enhanced Universal Framework v3.0 - Dynamic Variable Analysis Protocol
====================================================================

CRITICAL UPGRADE: Implements mandatory dynamic variable identification
before any economic/budgetary analysis to prevent static analysis errors.

Key Improvements:
- Pre-analysis variable scanning protocol
- Inflation adjustment automation  
- Currency/temporal normalization
- Dynamic context awareness
- Multi-dimensional validation matrix

Author: LexCertainty Enterprise System
Version: 3.0.0 Dynamic Variables
License: Proprietary - Enhanced after Critical User Feedback
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
import warnings
from abc import ABC, abstractmethod

class VariableType(Enum):
    """Types of variables that must be considered in dynamic analysis"""
    TEMPORAL = "temporal"
    MONETARY = "monetary"  
    INFLATION = "inflation"
    CURRENCY = "currency"
    DEMOGRAPHIC = "demographic"
    POLICY = "policy"
    EXTERNAL = "external"
    STRUCTURAL = "structural"

class CriticalityLevel(Enum):
    """Criticality levels for variable impact assessment"""
    BLOCKING = "blocking"  # Analysis cannot proceed without this variable
    CRITICAL = "critical"  # Results invalid without proper treatment  
    IMPORTANT = "important"  # Significant impact on conclusions
    MODERATE = "moderate"  # Minor impact but should be considered
    INFORMATIONAL = "informational"  # Context only

@dataclass
class DynamicVariable:
    """Represents a variable identified in dynamic pre-analysis"""
    name: str
    type: VariableType
    criticality: CriticalityLevel
    detected_value: Optional[Any] = None
    official_source: Optional[str] = None
    adjustment_method: Optional[str] = None
    impact_description: str = ""
    requires_normalization: bool = False
    
    def __post_init__(self):
        if self.criticality == CriticalityLevel.BLOCKING and self.detected_value is None:
            raise ValueError(f"Blocking variable {self.name} must have detected value")

@dataclass 
class VariableAnalysisResult:
    """Results of dynamic variable identification"""
    identified_variables: List[DynamicVariable]
    blocking_variables: List[DynamicVariable] = field(default_factory=list)
    critical_variables: List[DynamicVariable] = field(default_factory=list)
    analysis_feasible: bool = True
    recommended_adjustments: List[str] = field(default_factory=list)
    data_requirements: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.blocking_variables = [v for v in self.identified_variables if v.criticality == CriticalityLevel.BLOCKING]
        self.critical_variables = [v for v in self.identified_variables if v.criticality == CriticalityLevel.CRITICAL]
        self.analysis_feasible = len([v for v in self.blocking_variables if v.detected_value is None]) == 0

class DynamicVariableScanner:
    """
    Scans data and context to identify ALL variables that must be considered
    before proceeding with economic/budgetary analysis
    """
    
    def __init__(self, domain: str = "budgetary_analysis"):
        self.domain = domain
        self.logger = logging.getLogger(f"DynamicVariableScanner_{domain}")
        
        # Pre-configured variable detection patterns
        self.variable_patterns = self._initialize_detection_patterns()
        
        # External data sources for variable validation
        self.external_sources = {
            'inflation': ['central_bank', 'statistics_office', 'budget_documents'],
            'currency': ['central_bank', 'exchange_rate_apis'],
            'demographics': ['census_data', 'statistics_office'],
            'policy': ['official_documents', 'legal_frameworks']
        }
    
    def _initialize_detection_patterns(self) -> Dict[str, Dict]:
        """Initialize patterns for automatic variable detection"""
        
        return {
            'temporal_patterns': {
                'multi_year_comparison': r'(\d{4})\s*(?:vs?|-)?\s*(\d{4})',
                'percentage_change': r'(?:incremento|aumento|reducción|cambio).*?(\d+\.?\d*)%',
                'budget_periods': r'(presupuesto|ejercicio)\s*(\d{4})',
                'projection_language': r'(proyecci[óo]n|estima|prevé|espera).*?(\d{4})'
            },
            
            'monetary_patterns': {
                'currency_amounts': r'(?:\$|pesos?)\s*(\d+(?:\.\d+)*(?:\.\d+)?)',
                'millions_billions': r'(\d+(?:\.\d+)?)\s*(millones?|billones?)',
                'budget_items': r'(gastos?|ingresos?|recursos?).*?(\d+)',
                'fiscal_terms': r'(déficit|superávit|equilibrio).*?(\d+)'
            },
            
            'inflation_indicators': {
                'inflation_mentions': r'(inflaci[óo]n|IPC|[íi]ndice.*precios?).*?(\d+\.?\d*)%?',
                'real_vs_nominal': r'(real|nominal|ajustado|constante)',
                'deflation_terms': r'(deflactor|ajuste.*inflaci[óo]n)',
                'price_level_changes': r'(precios?).*?(sube|baja|incrementa|reduce)'
            },
            
            'policy_indicators': {
                'reform_language': r'(reforma|transformaci[óo]n|modernizaci[óo]n)',
                'efficiency_terms': r'(eficiencia|reducci[óo]n|optimizaci[óo]n)',
                'employment_changes': r'(empleos?|personal|puestos?).*?(\d+)',
                'structural_changes': r'(estructural|sistémico|institucional)'
            }
        }
    
    def scan_for_variables(self, 
                          data_sources: Dict[str, Any], 
                          analysis_context: Dict[str, Any]) -> VariableAnalysisResult:
        """
        Main method: scans all provided data to identify critical variables
        that must be considered before proceeding with analysis
        """
        
        self.logger.info("🔍 Iniciando escaneo dinámico de variables críticas...")
        
        identified_variables = []
        
        # 1. Scan for temporal variables
        temporal_vars = self._scan_temporal_variables(data_sources, analysis_context)
        identified_variables.extend(temporal_vars)
        
        # 2. Scan for monetary/currency variables  
        monetary_vars = self._scan_monetary_variables(data_sources, analysis_context)
        identified_variables.extend(monetary_vars)
        
        # 3. Scan for inflation variables (CRITICAL after user feedback)
        inflation_vars = self._scan_inflation_variables(data_sources, analysis_context)
        identified_variables.extend(inflation_vars)
        
        # 4. Scan for policy/structural variables
        policy_vars = self._scan_policy_variables(data_sources, analysis_context)
        identified_variables.extend(policy_vars)
        
        # 5. Scan for external/contextual variables
        external_vars = self._scan_external_variables(data_sources, analysis_context)
        identified_variables.extend(external_vars)
        
        # 6. Cross-validate and prioritize variables
        validated_variables = self._cross_validate_variables(identified_variables, data_sources)
        
        return VariableAnalysisResult(identified_variables=validated_variables)
    
    def _scan_temporal_variables(self, data_sources: Dict, context: Dict) -> List[DynamicVariable]:
        """Identify temporal variables requiring normalization"""
        
        variables = []
        
        # Check for multi-year comparisons
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, str):
                year_matches = re.findall(self.variable_patterns['temporal_patterns']['multi_year_comparison'], source_data)
                
                if year_matches:
                    for year_pair in year_matches:
                        year1, year2 = int(year_pair[0]), int(year_pair[1])
                        
                        if abs(year2 - year1) > 0:  # Multi-year comparison detected
                            variables.append(DynamicVariable(
                                name=f"temporal_comparison_{year1}_{year2}",
                                type=VariableType.TEMPORAL,
                                criticality=CriticalityLevel.CRITICAL,
                                detected_value={'start_year': year1, 'end_year': year2},
                                impact_description=f"Multi-year comparison {year1}-{year2} requires temporal normalization",
                                requires_normalization=True
                            ))
        
        return variables
    
    def _scan_monetary_variables(self, data_sources: Dict, context: Dict) -> List[DynamicVariable]:
        """Identify monetary variables requiring currency/inflation adjustment"""
        
        variables = []
        
        # Detect currency amounts that may need adjustment
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, str):
                # Check for large monetary amounts (likely budget figures)
                currency_matches = re.findall(self.variable_patterns['monetary_patterns']['currency_amounts'], source_data)
                millions_matches = re.findall(self.variable_patterns['monetary_patterns']['millions_billions'], source_data)
                
                if currency_matches or millions_matches:
                    variables.append(DynamicVariable(
                        name=f"monetary_values_{source_name}",
                        type=VariableType.MONETARY,
                        criticality=CriticalityLevel.CRITICAL,
                        detected_value={'source': source_name, 'has_amounts': True},
                        impact_description="Monetary values detected - require currency normalization and inflation adjustment",
                        requires_normalization=True
                    ))
        
        return variables
    
    def _scan_inflation_variables(self, data_sources: Dict, context: Dict) -> List[DynamicVariable]:
        """
        CRITICAL: Identify inflation variables that MUST be considered
        This addresses the critical error identified by user feedback
        """
        
        variables = []
        
        # Check if multi-year monetary comparisons are present
        has_multi_year_monetary = False
        detected_years = set()
        
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, str):
                # Look for budget years
                year_matches = re.findall(r'(\d{4})', source_data)
                detected_years.update([int(y) for y in year_matches if 2020 <= int(y) <= 2030])
                
                # Look for monetary amounts
                has_monetary = bool(re.search(self.variable_patterns['monetary_patterns']['currency_amounts'], source_data))
                
                if has_monetary and len(detected_years) > 1:
                    has_multi_year_monetary = True
        
        # If multi-year monetary comparison detected, inflation adjustment is BLOCKING
        if has_multi_year_monetary:
            min_year, max_year = min(detected_years), max(detected_years)
            
            variables.append(DynamicVariable(
                name="inflation_adjustment_mandatory",
                type=VariableType.INFLATION,
                criticality=CriticalityLevel.BLOCKING,  # BLOCKING prevents analysis without this
                detected_value={'year_range': (min_year, max_year), 'requires_official_data': True},
                impact_description=f"Multi-year monetary comparison {min_year}-{max_year} requires MANDATORY inflation adjustment",
                requires_normalization=True,
                adjustment_method="official_government_projections"
            ))
        
        # Check for explicit inflation mentions
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, str):
                inflation_matches = re.findall(self.variable_patterns['inflation_indicators']['inflation_mentions'], source_data)
                
                if inflation_matches:
                    for match in inflation_matches:
                        variables.append(DynamicVariable(
                            name=f"explicit_inflation_{source_name}",
                            type=VariableType.INFLATION, 
                            criticality=CriticalityLevel.CRITICAL,
                            detected_value={'mentioned_rate': match[1] if len(match) > 1 else None},
                            official_source=source_name,
                            impact_description="Explicit inflation data available - must be integrated into analysis"
                        ))
        
        return variables
    
    def _scan_policy_variables(self, data_sources: Dict, context: Dict) -> List[DynamicVariable]:
        """Identify policy variables affecting structural analysis"""
        
        variables = []
        
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, str):
                # Check for employment/personnel changes
                employment_matches = re.findall(self.variable_patterns['policy_indicators']['employment_changes'], source_data)
                
                if employment_matches:
                    variables.append(DynamicVariable(
                        name=f"employment_policy_changes_{source_name}",
                        type=VariableType.POLICY,
                        criticality=CriticalityLevel.IMPORTANT,
                        detected_value={'source': source_name},
                        impact_description="Employment policy changes detected - affects structural analysis"
                    ))
                
                # Check for structural reforms
                reform_detected = bool(re.search(self.variable_patterns['policy_indicators']['reform_language'], source_data))
                
                if reform_detected:
                    variables.append(DynamicVariable(
                        name=f"structural_reforms_{source_name}",
                        type=VariableType.STRUCTURAL,
                        criticality=CriticalityLevel.IMPORTANT,
                        detected_value={'source': source_name},
                        impact_description="Structural reforms detected - affects baseline assumptions"
                    ))
        
        return variables
    
    def _scan_external_variables(self, data_sources: Dict, context: Dict) -> List[DynamicVariable]:
        """Identify external variables affecting analysis context"""
        
        variables = []
        
        # Check analysis domain for specific external factors
        if self.domain == "budgetary_analysis":
            # Economic cycle variables
            variables.append(DynamicVariable(
                name="economic_cycle_position",
                type=VariableType.EXTERNAL,
                criticality=CriticalityLevel.MODERATE,
                impact_description="Economic cycle position affects budget execution probabilities"
            ))
            
            # Political context variables  
            variables.append(DynamicVariable(
                name="political_implementation_capacity",
                type=VariableType.EXTERNAL,
                criticality=CriticalityLevel.IMPORTANT,
                impact_description="Political capacity affects reform implementation likelihood"
            ))
        
        return variables
    
    def _cross_validate_variables(self, variables: List[DynamicVariable], data_sources: Dict) -> List[DynamicVariable]:
        """Cross-validate identified variables and remove duplicates/conflicts"""
        
        # Remove duplicates by name
        unique_variables = {}
        for var in variables:
            if var.name not in unique_variables:
                unique_variables[var.name] = var
            else:
                # Keep the one with higher criticality
                if var.criticality.value < unique_variables[var.name].criticality.value:
                    unique_variables[var.name] = var
        
        return list(unique_variables.values())
    
    def generate_adjustment_protocol(self, variable_results: VariableAnalysisResult) -> Dict[str, Any]:
        """Generate specific adjustment protocol based on identified variables"""
        
        protocol = {
            'mandatory_adjustments': [],
            'recommended_adjustments': [],
            'data_requirements': [],
            'analysis_modifications': []
        }
        
        for var in variable_results.blocking_variables:
            if var.type == VariableType.INFLATION:
                protocol['mandatory_adjustments'].append({
                    'type': 'inflation_adjustment',
                    'method': 'deflate_to_base_year', 
                    'data_required': 'official_inflation_projections',
                    'implementation': 'apply_compound_deflator'
                })
                
                protocol['data_requirements'].append(
                    f"Official inflation projections for years {var.detected_value['year_range']}"
                )
        
        for var in variable_results.critical_variables:
            if var.type == VariableType.MONETARY:
                protocol['recommended_adjustments'].append({
                    'type': 'currency_normalization',
                    'method': 'convert_to_base_currency_year',
                    'rationale': 'Ensure comparability across time periods'
                })
        
        return protocol

class InflationAdjustedAnalyzer:
    """
    Specialized analyzer that automatically applies inflation adjustments
    based on dynamic variable scanning results
    """
    
    def __init__(self, base_year: int = None):
        self.base_year = base_year
        self.inflation_data = {}
        self.logger = logging.getLogger("InflationAdjustedAnalyzer")
    
    def load_official_inflation_projections(self, projections: Dict[str, float]):
        """Load official inflation projections for adjustment calculations"""
        
        self.inflation_data = projections
        self.logger.info(f"Loaded inflation projections: {projections}")
    
    def calculate_compound_deflator(self, start_year: int, end_year: int) -> float:
        """Calculate compound inflation deflator between years"""
        
        if start_year == end_year:
            return 1.0
        
        deflator = 1.0
        year_range = range(start_year + 1, end_year + 1) if end_year > start_year else range(end_year + 1, start_year + 1)
        
        for year in year_range:
            year_inflation = self.inflation_data.get(str(year), 0.0)
            deflator *= (1 + year_inflation)
        
        return deflator if end_year > start_year else 1.0 / deflator
    
    def adjust_value_to_real(self, nominal_value: float, nominal_year: int, base_year: int) -> float:
        """Convert nominal value to real value in base year terms"""
        
        deflator = self.calculate_compound_deflator(base_year, nominal_year) 
        real_value = nominal_value / deflator
        
        self.logger.debug(f"Adjusted {nominal_value:,.0f} ({nominal_year}) -> {real_value:,.0f} ({base_year} real)")
        
        return real_value
    
    def calculate_real_growth_rate(self, value_start: float, value_end: float, 
                                  year_start: int, year_end: int, base_year: int = None) -> float:
        """Calculate real growth rate between two periods"""
        
        if base_year is None:
            base_year = year_start
        
        real_start = self.adjust_value_to_real(value_start, year_start, base_year) 
        real_end = self.adjust_value_to_real(value_end, year_end, base_year)
        
        real_growth = (real_end - real_start) / real_start if real_start != 0 else 0
        
        return real_growth

def demonstrate_dynamic_variable_analysis():
    """Demonstration of the dynamic variable analysis protocol"""
    
    # Example: Budget analysis data (simulating the original Argentina 2026 case)
    sample_data = {
        'budget_document': """
        Gastos Totales:
        - 2025: $122.557.389 millones
        - 2026: $147.820.252 millones
        Incremento: +20.6%
        
        Administración Gubernamental:
        - 2025: $7.323.494 millones  
        - 2026: $8.859.072 millones
        Incremento: +21.0%
        """,
        
        'inflation_projections': {
            '2025': 0.245,  # 24.5%
            '2026': 0.101   # 10.1%
        }
    }
    
    # Initialize scanner
    scanner = DynamicVariableScanner(domain="budgetary_analysis")
    
    # Scan for variables
    variable_results = scanner.scan_for_variables(sample_data, {'analysis_type': 'budgetary_memetic'})
    
    print("🔍 DYNAMIC VARIABLE ANALYSIS RESULTS:")
    print(f"Analysis Feasible: {variable_results.analysis_feasible}")
    print(f"Total Variables Identified: {len(variable_results.identified_variables)}")
    print(f"Blocking Variables: {len(variable_results.blocking_variables)}")
    
    # Generate adjustment protocol
    protocol = scanner.generate_adjustment_protocol(variable_results)
    
    # Apply adjustments
    if variable_results.analysis_feasible:
        adjuster = InflationAdjustedAnalyzer()
        adjuster.load_official_inflation_projections(sample_data['inflation_projections'])
        
        # Example adjustment
        nominal_2026 = 147820252  # millions
        real_2026 = adjuster.adjust_value_to_real(nominal_2026, 2026, 2025)
        real_growth = adjuster.calculate_real_growth_rate(122557389, nominal_2026, 2025, 2026)
        
        print(f"\n📊 CORRECTED ANALYSIS:")
        print(f"Nominal 2026: ${nominal_2026:,.0f} millions")
        print(f"Real 2026 (2025 terms): ${real_2026:,.0f} millions") 
        print(f"Real Growth Rate: {real_growth:.1%}")
        
        if real_growth < 0:
            print("✅ CONCLUSION: Real reduction, not increase - no paradox detected")
        else:
            print("⚠️  CONCLUSION: Real increase confirmed - requires further analysis")

if __name__ == "__main__":
    demonstrate_dynamic_variable_analysis()