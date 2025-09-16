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

@dataclass 
class VariableAnalysisResult:
    """Results of dynamic variable identification"""
    identified_variables: List[DynamicVariable]
    blocking_variables: List[DynamicVariable] = field(default_factory=list)
    critical_variables: List[DynamicVariable] = field(default_factory=list)
    analysis_feasible: bool = True
    recommended_adjustments: List[str] = field(default_factory=list)
    data_requirements: List[str] = field(default_factory=list)

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
            }
        }
    
    def scan_for_variables(self, 
                          data_sources: Dict[str, Any], 
                          analysis_context: Dict[str, Any]) -> VariableAnalysisResult:
        """
        Main method: scans all provided data to identify critical variables
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
        
        # 4. Cross-validate variables
        validated_variables = self._cross_validate_variables(identified_variables, data_sources)
        
        # 5. Build result with categorization
        result = VariableAnalysisResult(identified_variables=validated_variables)
        result.blocking_variables = [v for v in validated_variables if v.criticality == CriticalityLevel.BLOCKING]
        result.critical_variables = [v for v in validated_variables if v.criticality == CriticalityLevel.CRITICAL]
        result.analysis_feasible = len([v for v in result.blocking_variables if v.detected_value is None]) == 0
        
        return result
    
    def _scan_temporal_variables(self, data_sources: Dict, context: Dict) -> List[DynamicVariable]:
        """Identify temporal variables requiring normalization"""
        
        variables = []
        
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, str):
                year_matches = re.findall(self.variable_patterns['temporal_patterns']['multi_year_comparison'], source_data)
                
                if year_matches:
                    for year_pair in year_matches:
                        year1, year2 = int(year_pair[0]), int(year_pair[1])
                        
                        if abs(year2 - year1) > 0:
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
        
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, str):
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
        """
        
        variables = []
        
        # Check if multi-year monetary comparisons are present
        has_multi_year_monetary = False
        detected_years = set()
        
        for source_name, source_data in data_sources.items():
            if isinstance(source_data, str):
                year_matches = re.findall(r'(\d{4})', source_data)
                detected_years.update([int(y) for y in year_matches if 2020 <= int(y) <= 2030])
                
                has_monetary = bool(re.search(self.variable_patterns['monetary_patterns']['currency_amounts'], source_data))
                
                if has_monetary and len(detected_years) > 1:
                    has_multi_year_monetary = True
        
        # If multi-year monetary comparison detected, inflation adjustment is BLOCKING
        if has_multi_year_monetary:
            min_year, max_year = min(detected_years), max(detected_years)
            
            variables.append(DynamicVariable(
                name="inflation_adjustment_mandatory",
                type=VariableType.INFLATION,
                criticality=CriticalityLevel.BLOCKING,
                detected_value={'year_range': (min_year, max_year), 'requires_official_data': True},
                impact_description=f"Multi-year monetary comparison {min_year}-{max_year} requires MANDATORY inflation adjustment",
                requires_normalization=True,
                adjustment_method="official_government_projections"
            ))
        
        return variables
    
    def _cross_validate_variables(self, variables: List[DynamicVariable], data_sources: Dict) -> List[DynamicVariable]:
        """Cross-validate identified variables and remove duplicates"""
        
        unique_variables = {}
        for var in variables:
            if var.name not in unique_variables:
                unique_variables[var.name] = var
            else:
                if var.criticality.value < unique_variables[var.name].criticality.value:
                    unique_variables[var.name] = var
        
        return list(unique_variables.values())

class InflationAdjustedAnalyzer:
    """
    Specialized analyzer that automatically applies inflation adjustments
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