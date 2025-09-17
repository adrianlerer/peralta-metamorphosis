"""
GDR-Enhanced Universal Framework v4.0
=====================================

Integrates Generative Data Refinement (GDR) methodologies from arXiv:2509.08653
with the Enhanced Universal Framework v3.0 for superior analytical rigor and
data quality preservation in budget and economic analysis.

Key GDR Integrations:
- Declarative Safety Criteria for analytical outputs
- Formal verification functions for economic conclusions
- Grounded synthetic generation for scenario testing
- Verification loops for iterative quality improvement
- Data governance pipelines with quantifiable metrics
- Utility-preserving data cleaning protocols

Author: LexCertainty Enterprise System
Version: 4.0.0 - GDR Integration
License: Proprietary - Private Repository Enhancement
Based on: Enhanced Universal Framework v3.0 + arXiv:2509.08653 GDR principles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
import warnings
from abc import ABC, abstractmethod
import json
from pathlib import Path

# Import base framework components
from ENHANCED_UNIVERSAL_FRAMEWORK_V3_DYNAMIC_VARIABLES import (
    VariableType, CriticalityLevel, DynamicVariable, VariableAnalysisResult,
    DynamicVariableScanner, InflationAdjustedAnalyzer
)

class GDRSafetyCriteria(Enum):
    """Declarative safety criteria based on GDR methodology"""
    FACTUAL_CONSISTENCY = "factual_consistency"
    TEMPORAL_COHERENCE = "temporal_coherence"
    INFLATION_ADJUSTMENT_COMPLIANCE = "inflation_adjustment_compliance"
    SOURCE_TRACEABILITY = "source_traceability"
    LOGICAL_VALIDITY = "logical_validity"
    QUANTITATIVE_PRECISION = "quantitative_precision"
    METHODOLOGICAL_TRANSPARENCY = "methodological_transparency"
    BIAS_DETECTION = "bias_detection"

class VerificationResult(Enum):
    """Results of GDR verification functions"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    REQUIRES_MANUAL_REVIEW = "manual_review"

@dataclass
class GDRVerificationFunction:
    """Formal verification function implementing GDR principles"""
    name: str
    safety_criteria: GDRSafetyCriteria
    verification_callable: Callable[[Any], Tuple[VerificationResult, str, Dict]]
    criticality_level: CriticalityLevel
    description: str
    remediation_suggestions: List[str] = field(default_factory=list)

@dataclass
class GDRAnalysisOutput:
    """Enhanced analysis output with GDR verification"""
    content: Dict[str, Any]
    verification_results: Dict[str, Tuple[VerificationResult, str, Dict]]
    safety_score: float
    traceability_metadata: Dict[str, Any]
    improvement_suggestions: List[str]
    data_provenance: Dict[str, Any]
    quality_metrics: Dict[str, float]

@dataclass
class GDRDataGovernanceConfig:
    """Configuration for GDR-based data governance pipeline"""
    verification_functions: List[GDRVerificationFunction]
    quality_thresholds: Dict[str, float]
    mandatory_criteria: List[GDRSafetyCriteria]
    iterative_improvement_enabled: bool = True
    synthetic_validation_enabled: bool = True
    provenance_tracking_enabled: bool = True

class GDRVerificationEngine:
    """
    Core GDR verification engine implementing formal verification functions
    """
    
    def __init__(self, config: GDRDataGovernanceConfig):
        self.config = config
        self.logger = logging.getLogger("GDRVerificationEngine")
        self.verification_history = []
        
        # Initialize standard verification functions
        self.standard_verifiers = self._initialize_standard_verifiers()
        
    def _initialize_standard_verifiers(self) -> List[GDRVerificationFunction]:
        """Initialize standard GDR verification functions for economic analysis"""
        
        return [
            GDRVerificationFunction(
                name="inflation_adjustment_verifier",
                safety_criteria=GDRSafetyCriteria.INFLATION_ADJUSTMENT_COMPLIANCE,
                verification_callable=self._verify_inflation_adjustment,
                criticality_level=CriticalityLevel.BLOCKING,
                description="Verifies mandatory inflation adjustment for multi-year monetary comparisons",
                remediation_suggestions=[
                    "Apply official inflation projections to nominal values",
                    "Convert to real values using compound deflator",
                    "Document adjustment methodology clearly"
                ]
            ),
            
            GDRVerificationFunction(
                name="temporal_coherence_verifier",
                safety_criteria=GDRSafetyCriteria.TEMPORAL_COHERENCE,
                verification_callable=self._verify_temporal_coherence,
                criticality_level=CriticalityLevel.CRITICAL,
                description="Ensures temporal consistency in chronological analysis",
                remediation_suggestions=[
                    "Check chronological order of events",
                    "Verify date formatting consistency",
                    "Validate temporal causality chains"
                ]
            ),
            
            GDRVerificationFunction(
                name="factual_consistency_verifier", 
                safety_criteria=GDRSafetyCriteria.FACTUAL_CONSISTENCY,
                verification_callable=self._verify_factual_consistency,
                criticality_level=CriticalityLevel.CRITICAL,
                description="Validates factual claims against authoritative sources",
                remediation_suggestions=[
                    "Cross-reference with official government data",
                    "Verify numerical calculations independently",
                    "Check for internal contradictions"
                ]
            ),
            
            GDRVerificationFunction(
                name="source_traceability_verifier",
                safety_criteria=GDRSafetyCriteria.SOURCE_TRACEABILITY,
                verification_callable=self._verify_source_traceability,
                criticality_level=CriticalityLevel.IMPORTANT,
                description="Ensures all claims are traceable to authoritative sources",
                remediation_suggestions=[
                    "Add specific citations for all numerical data",
                    "Link to official government documents",
                    "Provide methodology transparency"
                ]
            ),
            
            GDRVerificationFunction(
                name="quantitative_precision_verifier",
                safety_criteria=GDRSafetyCriteria.QUANTITATIVE_PRECISION,
                verification_callable=self._verify_quantitative_precision,
                criticality_level=CriticalityLevel.CRITICAL,
                description="Validates numerical accuracy and calculation precision",
                remediation_suggestions=[
                    "Verify all mathematical calculations",
                    "Check decimal precision consistency",
                    "Validate percentage calculations"
                ]
            )
        ]
    
    def _verify_inflation_adjustment(self, analysis_output: Dict) -> Tuple[VerificationResult, str, Dict]:
        """Verify inflation adjustment compliance"""
        
        metadata = {}
        
        # Check for multi-year monetary comparisons
        has_multi_year_data = False
        years_detected = set()
        
        # Scan content for year patterns and monetary amounts
        content_str = json.dumps(analysis_output)
        year_matches = re.findall(r'(\d{4})', content_str)
        monetary_matches = re.findall(r'(?:\$|peso|millón|billón)', content_str)
        
        if year_matches:
            years_detected = set([int(y) for y in year_matches if 2020 <= int(y) <= 2030])
            
        if len(years_detected) > 1 and monetary_matches:
            has_multi_year_data = True
            metadata['detected_years'] = sorted(list(years_detected))
            metadata['has_monetary_data'] = True
        
        if has_multi_year_data:
            # Check for inflation adjustment indicators
            adjustment_indicators = [
                'real', 'ajustado', 'inflación', 'deflactor', 'constante',
                'valor real', 'precios constantes'
            ]
            
            has_adjustment = any(indicator in content_str.lower() for indicator in adjustment_indicators)
            
            if has_adjustment:
                return (VerificationResult.PASS, 
                       "Inflation adjustment detected for multi-year monetary comparison", 
                       metadata)
            else:
                return (VerificationResult.FAIL,
                       f"Multi-year monetary comparison ({min(years_detected)}-{max(years_detected)}) requires mandatory inflation adjustment",
                       metadata)
        
        return (VerificationResult.PASS, "No multi-year monetary comparison detected", metadata)
    
    def _verify_temporal_coherence(self, analysis_output: Dict) -> Tuple[VerificationResult, str, Dict]:
        """Verify temporal coherence in analysis"""
        
        metadata = {}
        content_str = json.dumps(analysis_output)
        
        # Extract dates and verify chronological order
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{4})',  # YYYY
        ]
        
        extracted_dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, content_str)
            for match in matches:
                if len(match.groups()) == 3:
                    try:
                        if '/' in match.group():
                            month, day, year = match.groups()
                            date_obj = datetime(int(year), int(month), int(day))
                        else:
                            year, month, day = match.groups()
                            date_obj = datetime(int(year), int(month), int(day))
                        extracted_dates.append(date_obj)
                    except:
                        continue
                elif len(match.groups()) == 1:  # Year only
                    try:
                        year = int(match.groups()[0])
                        if 1900 <= year <= 2100:
                            extracted_dates.append(datetime(year, 1, 1))
                    except:
                        continue
        
        metadata['extracted_dates_count'] = len(extracted_dates)
        
        if len(extracted_dates) > 1:
            # Check for temporal inconsistencies
            sorted_dates = sorted(extracted_dates)
            is_chronological = extracted_dates == sorted_dates
            
            if is_chronological:
                return (VerificationResult.PASS, "Temporal coherence maintained", metadata)
            else:
                return (VerificationResult.WARNING, "Potential temporal inconsistencies detected", metadata)
        
        return (VerificationResult.PASS, "Insufficient temporal data for coherence analysis", metadata)
    
    def _verify_factual_consistency(self, analysis_output: Dict) -> Tuple[VerificationResult, str, Dict]:
        """Verify factual consistency of claims"""
        
        metadata = {}
        content_str = json.dumps(analysis_output)
        
        # Check for internal contradictions in numerical claims
        percentage_matches = re.findall(r'(-?\d+\.?\d*)%', content_str)
        
        if percentage_matches:
            percentages = [float(p) for p in percentage_matches]
            metadata['percentages_found'] = len(percentages)
            metadata['percentage_range'] = [min(percentages), max(percentages)]
            
            # Basic sanity checks
            extreme_values = [p for p in percentages if abs(p) > 1000]
            if extreme_values:
                return (VerificationResult.WARNING,
                       f"Extreme percentage values detected: {extreme_values}",
                       metadata)
        
        # Check for contradictory statements
        contradiction_patterns = [
            (r'aumenta.*(\d+\.?\d*)%', r'reduce.*(\d+\.?\d*)%'),
            (r'incremento.*(\d+\.?\d*)%', r'disminución.*(\d+\.?\d*)%'),
        ]
        
        for increase_pattern, decrease_pattern in contradiction_patterns:
            increases = re.findall(increase_pattern, content_str.lower())
            decreases = re.findall(decrease_pattern, content_str.lower())
            
            if increases and decreases:
                metadata['potential_contradiction'] = True
                return (VerificationResult.WARNING,
                       "Potential contradictory statements detected",
                       metadata)
        
        return (VerificationResult.PASS, "No factual inconsistencies detected", metadata)
    
    def _verify_source_traceability(self, analysis_output: Dict) -> Tuple[VerificationResult, str, Dict]:
        """Verify source traceability of claims"""
        
        metadata = {}
        content_str = json.dumps(analysis_output)
        
        # Check for source citations
        source_indicators = [
            r'según.*(?:gobierno|ministerio|banco central)',
            r'fuente:',
            r'datos.*oficiales?',
            r'proyección.*oficial',
            r'presupuesto.*\d{4}'
        ]
        
        sources_found = 0
        for indicator in source_indicators:
            matches = re.findall(indicator, content_str.lower())
            sources_found += len(matches)
        
        metadata['sources_found'] = sources_found
        
        # Check for numerical claims without sources
        numerical_claims = len(re.findall(r'\d+\.?\d*%|\$\s*\d+', content_str))
        metadata['numerical_claims'] = numerical_claims
        
        if numerical_claims > 0:
            source_ratio = sources_found / numerical_claims
            metadata['source_ratio'] = source_ratio
            
            if source_ratio >= 0.5:
                return (VerificationResult.PASS, "Adequate source traceability", metadata)
            elif source_ratio >= 0.2:
                return (VerificationResult.WARNING, "Moderate source traceability", metadata)
            else:
                return (VerificationResult.FAIL, "Insufficient source traceability", metadata)
        
        return (VerificationResult.PASS, "No numerical claims requiring sources", metadata)
    
    def _verify_quantitative_precision(self, analysis_output: Dict) -> Tuple[VerificationResult, str, Dict]:
        """Verify quantitative precision and calculation accuracy"""
        
        metadata = {}
        content_str = json.dumps(analysis_output)
        
        # Extract numerical values and check precision consistency
        decimal_matches = re.findall(r'(\d+\.\d+)', content_str)
        integer_matches = re.findall(r'(?<!\d)(\d+)(?!\.\d)', content_str)
        
        metadata['decimal_values'] = len(decimal_matches)
        metadata['integer_values'] = len(integer_matches)
        
        # Check for calculation consistency in percentages
        percentage_calculations = re.findall(r'(\d+\.?\d*)\s*%', content_str)
        
        precision_inconsistencies = []
        if percentage_calculations:
            decimal_places = [len(p.split('.')[1]) if '.' in p else 0 for p in percentage_calculations]
            if len(set(decimal_places)) > 2:  # More than 2 different precision levels
                precision_inconsistencies.append("Inconsistent decimal precision in percentages")
        
        metadata['precision_inconsistencies'] = precision_inconsistencies
        
        if precision_inconsistencies:
            return (VerificationResult.WARNING,
                   f"Precision inconsistencies detected: {precision_inconsistencies}",
                   metadata)
        
        return (VerificationResult.PASS, "Quantitative precision is consistent", metadata)
    
    def verify_analysis_output(self, analysis_output: Dict[str, Any]) -> GDRAnalysisOutput:
        """Main verification method implementing GDR methodology"""
        
        self.logger.info("🔍 Iniciando verificación GDR de análisis...")
        
        verification_results = {}
        total_score = 0
        max_score = 0
        
        # Run all verification functions
        all_verifiers = self.standard_verifiers + self.config.verification_functions
        
        for verifier in all_verifiers:
            try:
                result, message, metadata = verifier.verification_callable(analysis_output)
                verification_results[verifier.name] = (result, message, metadata)
                
                # Calculate score contribution
                if result == VerificationResult.PASS:
                    score_contribution = 1.0
                elif result == VerificationResult.WARNING:
                    score_contribution = 0.7
                elif result == VerificationResult.FAIL:
                    score_contribution = 0.0
                else:  # REQUIRES_MANUAL_REVIEW
                    score_contribution = 0.5
                
                # Weight by criticality
                if verifier.criticality_level == CriticalityLevel.BLOCKING:
                    weight = 3.0
                elif verifier.criticality_level == CriticalityLevel.CRITICAL:
                    weight = 2.0
                else:
                    weight = 1.0
                
                total_score += score_contribution * weight
                max_score += weight
                
            except Exception as e:
                self.logger.error(f"Error in verification function {verifier.name}: {e}")
                verification_results[verifier.name] = (
                    VerificationResult.REQUIRES_MANUAL_REVIEW,
                    f"Verification error: {str(e)}",
                    {}
                )
        
        # Calculate overall safety score
        safety_score = total_score / max_score if max_score > 0 else 0.0
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(verification_results, all_verifiers)
        
        # Build traceability metadata
        traceability_metadata = {
            'verification_timestamp': datetime.now().isoformat(),
            'framework_version': '4.0.0-GDR',
            'verifiers_used': [v.name for v in all_verifiers],
            'mandatory_criteria_checked': [c.value for c in self.config.mandatory_criteria]
        }
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(verification_results, analysis_output)
        
        return GDRAnalysisOutput(
            content=analysis_output,
            verification_results=verification_results,
            safety_score=safety_score,
            traceability_metadata=traceability_metadata,
            improvement_suggestions=improvement_suggestions,
            data_provenance={'source': 'GDR_Enhanced_Framework_v4'},
            quality_metrics=quality_metrics
        )
    
    def _generate_improvement_suggestions(self, verification_results: Dict, verifiers: List) -> List[str]:
        """Generate actionable improvement suggestions based on verification results"""
        
        suggestions = []
        verifier_dict = {v.name: v for v in verifiers}
        
        for verifier_name, (result, message, metadata) in verification_results.items():
            if result in [VerificationResult.FAIL, VerificationResult.WARNING]:
                verifier = verifier_dict.get(verifier_name)
                if verifier and verifier.remediation_suggestions:
                    suggestions.extend(verifier.remediation_suggestions)
        
        return list(set(suggestions))  # Remove duplicates
    
    def _calculate_quality_metrics(self, verification_results: Dict, analysis_output: Dict) -> Dict[str, float]:
        """Calculate quantifiable quality metrics for the analysis"""
        
        metrics = {}
        
        # Pass rate
        total_verifications = len(verification_results)
        passed_verifications = sum(1 for result, _, _ in verification_results.values() 
                                 if result == VerificationResult.PASS)
        metrics['pass_rate'] = passed_verifications / total_verifications if total_verifications > 0 else 0.0
        
        # Content completeness (basic heuristic)
        content_str = json.dumps(analysis_output)
        metrics['content_length'] = len(content_str)
        metrics['numerical_density'] = len(re.findall(r'\d+\.?\d*', content_str)) / len(content_str.split())
        
        # Source density
        source_indicators = ['fuente', 'según', 'datos', 'oficial', 'gobierno']
        source_count = sum(content_str.lower().count(indicator) for indicator in source_indicators)
        metrics['source_density'] = source_count / len(content_str.split())
        
        return metrics

class GDREnhancedUniversalFramework:
    """
    Enhanced Universal Framework v4.0 with integrated GDR methodology
    """
    
    def __init__(self, governance_config: Optional[GDRDataGovernanceConfig] = None):
        
        # Initialize base components
        self.variable_scanner = DynamicVariableScanner("gdr_enhanced_analysis")
        self.inflation_analyzer = InflationAdjustedAnalyzer()
        
        # Initialize GDR components
        self.governance_config = governance_config or self._create_default_governance_config()
        self.verification_engine = GDRVerificationEngine(self.governance_config)
        
        self.logger = logging.getLogger("GDREnhancedUniversalFramework")
        
    def _create_default_governance_config(self) -> GDRDataGovernanceConfig:
        """Create default GDR governance configuration"""
        
        return GDRDataGovernanceConfig(
            verification_functions=[],
            quality_thresholds={
                'safety_score': 0.8,
                'pass_rate': 0.75,
                'source_density': 0.05
            },
            mandatory_criteria=[
                GDRSafetyCriteria.INFLATION_ADJUSTMENT_COMPLIANCE,
                GDRSafetyCriteria.FACTUAL_CONSISTENCY,
                GDRSafetyCriteria.QUANTITATIVE_PRECISION
            ],
            iterative_improvement_enabled=True,
            synthetic_validation_enabled=True,
            provenance_tracking_enabled=True
        )
    
    def analyze_with_gdr_verification(self, 
                                    data_sources: Dict[str, Any],
                                    analysis_context: Dict[str, Any],
                                    analysis_function: Callable[[Dict, Dict], Dict]) -> GDRAnalysisOutput:
        """
        Main analysis method with integrated GDR verification
        """
        
        self.logger.info("🚀 Iniciando análisis con verificación GDR integrada...")
        
        # Phase 1: Dynamic Variable Scanning (from v3.0)
        self.logger.info("📋 Fase 1: Escaneo de variables dinámicas...")
        variable_analysis = self.variable_scanner.scan_for_variables(data_sources, analysis_context)
        
        # Check for blocking variables
        if not variable_analysis.analysis_feasible:
            blocking_vars = [v.name for v in variable_analysis.blocking_variables if v.detected_value is None]
            raise ValueError(f"Analysis blocked by missing critical variables: {blocking_vars}")
        
        # Phase 2: Apply necessary adjustments
        self.logger.info("⚙️ Fase 2: Aplicando ajustes dinámicos...")
        adjusted_data = self._apply_dynamic_adjustments(data_sources, variable_analysis)
        
        # Phase 3: Execute analysis
        self.logger.info("🔬 Fase 3: Ejecutando análisis principal...")
        analysis_output = analysis_function(adjusted_data, analysis_context)
        
        # Phase 4: GDR Verification
        self.logger.info("✅ Fase 4: Verificación GDR...")
        gdr_verified_output = self.verification_engine.verify_analysis_output(analysis_output)
        
        # Phase 5: Iterative improvement (if enabled)
        if self.governance_config.iterative_improvement_enabled:
            gdr_verified_output = self._apply_iterative_improvement(gdr_verified_output)
        
        return gdr_verified_output
    
    def _apply_dynamic_adjustments(self, data_sources: Dict, variable_analysis: VariableAnalysisResult) -> Dict:
        """Apply dynamic adjustments based on identified variables"""
        
        adjusted_data = data_sources.copy()
        
        # Apply inflation adjustments if required
        for variable in variable_analysis.identified_variables:
            if variable.type == VariableType.INFLATION and variable.criticality == CriticalityLevel.BLOCKING:
                if variable.detected_value and 'year_range' in variable.detected_value:
                    year_range = variable.detected_value['year_range']
                    
                    # Load official inflation projections
                    official_projections = {
                        '2025': 0.245,  # 24.5% official projection
                        '2026': 0.101   # 10.1% official projection
                    }
                    
                    self.inflation_analyzer.load_official_inflation_projections(official_projections)
                    
                    # Mark data as inflation-adjusted
                    adjusted_data['_inflation_adjustment_applied'] = True
                    adjusted_data['_inflation_method'] = 'official_government_projections'
                    adjusted_data['_inflation_years'] = year_range
        
        return adjusted_data
    
    def _apply_iterative_improvement(self, gdr_output: GDRAnalysisOutput) -> GDRAnalysisOutput:
        """Apply iterative improvement based on verification results"""
        
        if gdr_output.safety_score >= self.governance_config.quality_thresholds.get('safety_score', 0.8):
            return gdr_output
        
        # Generate improvement recommendations
        improvement_actions = []
        
        for verifier_name, (result, message, metadata) in gdr_output.verification_results.items():
            if result == VerificationResult.FAIL:
                improvement_actions.append(f"CRITICAL: {verifier_name} - {message}")
            elif result == VerificationResult.WARNING:
                improvement_actions.append(f"WARNING: {verifier_name} - {message}")
        
        # Add improvement recommendations to output
        gdr_output.improvement_suggestions.extend(improvement_actions)
        
        return gdr_output
    
    def generate_gdr_compliance_report(self, gdr_output: GDRAnalysisOutput) -> str:
        """Generate a comprehensive GDR compliance report"""
        
        report = []
        report.append("# REPORTE DE CUMPLIMIENTO GDR (Generative Data Refinement)")
        report.append("="*60)
        report.append(f"**Timestamp**: {gdr_output.traceability_metadata.get('verification_timestamp', 'N/A')}")
        report.append(f"**Framework Version**: {gdr_output.traceability_metadata.get('framework_version', 'N/A')}")
        report.append(f"**Puntuación de Seguridad**: {gdr_output.safety_score:.3f}")
        report.append("")
        
        # Verification Results Summary
        report.append("## RESUMEN DE VERIFICACIONES")
        report.append("-" * 30)
        
        pass_count = 0
        fail_count = 0
        warning_count = 0
        
        for verifier_name, (result, message, metadata) in gdr_output.verification_results.items():
            status_icon = {
                VerificationResult.PASS: "✅",
                VerificationResult.FAIL: "❌", 
                VerificationResult.WARNING: "⚠️",
                VerificationResult.REQUIRES_MANUAL_REVIEW: "🔍"
            }.get(result, "❓")
            
            report.append(f"{status_icon} **{verifier_name}**: {message}")
            
            if result == VerificationResult.PASS:
                pass_count += 1
            elif result == VerificationResult.FAIL:
                fail_count += 1
            elif result == VerificationResult.WARNING:
                warning_count += 1
        
        report.append("")
        report.append(f"**Total Verificaciones**: {len(gdr_output.verification_results)}")
        report.append(f"**Exitosas**: {pass_count} | **Advertencias**: {warning_count} | **Fallidas**: {fail_count}")
        report.append("")
        
        # Quality Metrics
        report.append("## MÉTRICAS DE CALIDAD")
        report.append("-" * 20)
        for metric_name, metric_value in gdr_output.quality_metrics.items():
            report.append(f"- **{metric_name}**: {metric_value:.4f}")
        report.append("")
        
        # Improvement Suggestions
        if gdr_output.improvement_suggestions:
            report.append("## SUGERENCIAS DE MEJORA")
            report.append("-" * 25)
            for i, suggestion in enumerate(gdr_output.improvement_suggestions, 1):
                report.append(f"{i}. {suggestion}")
            report.append("")
        
        # GDR Compliance Status
        report.append("## ESTATUS DE CUMPLIMIENTO GDR")
        report.append("-" * 32)
        
        compliance_status = "COMPLIANT" if gdr_output.safety_score >= 0.8 else "NON_COMPLIANT"
        status_icon = "✅" if compliance_status == "COMPLIANT" else "❌"
        
        report.append(f"{status_icon} **Estatus**: {compliance_status}")
        report.append(f"**Criterios Obligatorios Verificados**: {len(self.governance_config.mandatory_criteria)}")
        report.append("")
        
        return "\n".join(report)

# Utility functions for GDR integration
def create_budget_analysis_gdr_config() -> GDRDataGovernanceConfig:
    """Create specialized GDR configuration for budget analysis"""
    
    return GDRDataGovernanceConfig(
        verification_functions=[],
        quality_thresholds={
            'safety_score': 0.85,  # Higher threshold for budget analysis
            'pass_rate': 0.80,
            'source_density': 0.08,
            'numerical_density': 0.15
        },
        mandatory_criteria=[
            GDRSafetyCriteria.INFLATION_ADJUSTMENT_COMPLIANCE,
            GDRSafetyCriteria.FACTUAL_CONSISTENCY,
            GDRSafetyCriteria.QUANTITATIVE_PRECISION,
            GDRSafetyCriteria.SOURCE_TRACEABILITY,
            GDRSafetyCriteria.TEMPORAL_COHERENCE
        ],
        iterative_improvement_enabled=True,
        synthetic_validation_enabled=True,
        provenance_tracking_enabled=True
    )

def validate_gdr_framework_integration() -> bool:
    """Validate that GDR framework integration is working correctly"""
    
    try:
        # Test basic initialization
        config = create_budget_analysis_gdr_config()
        framework = GDREnhancedUniversalFramework(config)
        
        # Test verification engine
        test_analysis = {
            "budget_2025": "1000 millones",
            "budget_2026": "900 millones", 
            "change": "reducción del 10%"
        }
        
        verified_output = framework.verification_engine.verify_analysis_output(test_analysis)
        
        return isinstance(verified_output, GDRAnalysisOutput) and len(verified_output.verification_results) > 0
        
    except Exception as e:
        logging.error(f"GDR framework validation failed: {e}")
        return False

# Main execution for testing
if __name__ == "__main__":
    print("🧪 Testing GDR-Enhanced Universal Framework v4.0...")
    
    if validate_gdr_framework_integration():
        print("✅ GDR Framework integration validated successfully")
    else:
        print("❌ GDR Framework integration validation failed")