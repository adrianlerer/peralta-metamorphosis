"""
Universal Analysis Framework
Implementación de los 8 meta-principios universales extraídos del desarrollo de LexCertainty Enterprise.

Meta-Principios Universales:
1. Marco de abstención matemática (Mathematical Abstention Framework)
2. Límites de confianza en decisiones (Confidence Bounds in Decisions)
3. Análisis genealógico de influencias (Genealogical Analysis of Influences)
4. Pipeline multi-etapa con validación (Multi-stage Pipeline with Validation)
5. Evaluación ensemble multi-modelo (Multi-model Ensemble Evaluation)
6. Cuantificación de incertidumbre (Uncertainty Quantification)
7. Salida estructurada con metadatos (Structured Output with Metadata)
8. Hibridación adaptativa (Adaptive Hybridization)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
import numpy as np
from pathlib import Path

# Type variables for generic implementation
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

class AnalysisStage(Enum):
    """Etapas del pipeline de análisis universal"""
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    CORE_ANALYSIS = "core_analysis"
    VALIDATION = "validation"
    ENSEMBLE_EVALUATION = "ensemble_evaluation"
    CONFIDENCE_CALCULATION = "confidence_calculation"
    ABSTENTION_DECISION = "abstention_decision"
    RESULT_SYNTHESIS = "result_synthesis"

class ConfidenceLevel(Enum):
    """Niveles de confianza para decisiones"""
    VERY_HIGH = 0.95
    HIGH = 0.90
    MEDIUM = 0.80
    LOW = 0.70
    VERY_LOW = 0.60

@dataclass
class UniversalMetadata:
    """Metadatos universales para cualquier análisis"""
    timestamp: datetime = field(default_factory=datetime.now)
    framework_version: str = "1.0.0"
    domain: str = "generic"
    analysis_id: str = ""
    input_hash: str = ""
    processing_stages: List[str] = field(default_factory=list)
    confidence_metrics: Dict[str, float] = field(default_factory=dict)
    uncertainty_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    genealogy_data: Dict[str, Any] = field(default_factory=dict)
    ensemble_results: List[Dict[str, Any]] = field(default_factory=list)
    abstention_reasons: List[str] = field(default_factory=list)

@dataclass
class UniversalResult(Generic[R]):
    """Resultado universal estructurado con metadatos completos"""
    result: Optional[R]
    confidence: float
    abstained: bool
    metadata: UniversalMetadata
    raw_outputs: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario para serialización"""
        return {
            "result": self.result,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "timestamp": self.metadata.timestamp.isoformat(),
            "framework_version": self.metadata.framework_version,
            "domain": self.metadata.domain,
            "analysis_id": self.metadata.analysis_id,
            "confidence_metrics": self.metadata.confidence_metrics,
            "uncertainty_bounds": self.metadata.uncertainty_bounds,
            "genealogy_data": self.metadata.genealogy_data,
            "ensemble_results": self.metadata.ensemble_results,
            "abstention_reasons": self.metadata.abstention_reasons,
            "raw_outputs": self.raw_outputs,
            "validation_metrics": self.validation_metrics
        }

class UniversalAnalyzer(ABC, Generic[T, R]):
    """Analizador base abstracto que implementa los 8 meta-principios universales"""
    
    def __init__(
        self,
        domain: str,
        confidence_threshold: float = 0.80,
        enable_abstention: bool = True,
        bootstrap_iterations: int = 1000,
        ensemble_models: Optional[List[str]] = None
    ):
        self.domain = domain
        self.confidence_threshold = confidence_threshold
        self.enable_abstention = enable_abstention
        self.bootstrap_iterations = bootstrap_iterations
        self.ensemble_models = ensemble_models or []
        self.logger = logging.getLogger(f"UniversalAnalyzer_{domain}")
        
    @abstractmethod
    def preprocess_input(self, input_data: T) -> Any:
        """Preprocesa los datos de entrada específicos del dominio"""
        pass
    
    @abstractmethod
    def extract_features(self, preprocessed_data: Any) -> Dict[str, Any]:
        """Extrae características relevantes para el análisis"""
        pass
    
    @abstractmethod
    def perform_core_analysis(self, features: Dict[str, Any]) -> R:
        """Realiza el análisis principal específico del dominio"""
        pass
    
    @abstractmethod
    def calculate_confidence_metrics(self, result: R, features: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de confianza específicas del dominio"""
        pass
    
    @abstractmethod
    def perform_genealogical_analysis(self, input_data: T, result: R) -> Dict[str, Any]:
        """Analiza las influencias y orígenes genealógicos del resultado"""
        pass
    
    def analyze(self, input_data: T) -> UniversalResult[R]:
        """
        Método principal que implementa el pipeline universal de análisis
        aplicando los 8 meta-principios
        """
        metadata = UniversalMetadata(
            domain=self.domain,
            analysis_id=self._generate_analysis_id(input_data)
        )
        
        try:
            # Etapa 1: Preprocesamiento
            metadata.processing_stages.append(AnalysisStage.PREPROCESSING.value)
            preprocessed_data = self.preprocess_input(input_data)
            
            # Etapa 2: Extracción de características
            metadata.processing_stages.append(AnalysisStage.FEATURE_EXTRACTION.value)
            features = self.extract_features(preprocessed_data)
            
            # Etapa 3: Análisis principal
            metadata.processing_stages.append(AnalysisStage.CORE_ANALYSIS.value)
            core_result = self.perform_core_analysis(features)
            
            # Etapa 4: Evaluación ensemble (Meta-Principio 5)
            metadata.processing_stages.append(AnalysisStage.ENSEMBLE_EVALUATION.value)
            ensemble_results = self._perform_ensemble_evaluation(features, core_result)
            metadata.ensemble_results = ensemble_results
            
            # Etapa 5: Análisis genealógico (Meta-Principio 3)
            genealogy_data = self.perform_genealogical_analysis(input_data, core_result)
            metadata.genealogy_data = genealogy_data
            
            # Etapa 6: Cálculo de confianza y límites (Meta-Principios 2, 6)
            metadata.processing_stages.append(AnalysisStage.CONFIDENCE_CALCULATION.value)
            confidence_metrics = self.calculate_confidence_metrics(core_result, features)
            uncertainty_bounds = self._calculate_uncertainty_bounds(confidence_metrics, ensemble_results)
            
            metadata.confidence_metrics = confidence_metrics
            metadata.uncertainty_bounds = uncertainty_bounds
            
            # Etapa 7: Decisión de abstención (Meta-Principio 1)
            metadata.processing_stages.append(AnalysisStage.ABSTENTION_DECISION.value)
            overall_confidence = self._calculate_overall_confidence(confidence_metrics, ensemble_results)
            abstention_decision, abstention_reasons = self._make_abstention_decision(
                overall_confidence, uncertainty_bounds
            )
            metadata.abstention_reasons = abstention_reasons
            
            # Etapa 8: Síntesis final con hibridación adaptativa (Meta-Principio 8)
            metadata.processing_stages.append(AnalysisStage.RESULT_SYNTHESIS.value)
            final_result = self._synthesize_result(
                core_result, ensemble_results, abstention_decision
            ) if not abstention_decision else None
            
            # Etapa 9: Validación (Meta-Principio 4)
            metadata.processing_stages.append(AnalysisStage.VALIDATION.value)
            validation_metrics = self._perform_validation(final_result, features) if final_result else {}
            
            # Meta-Principio 7: Salida estructurada con metadatos completos
            return UniversalResult(
                result=final_result,
                confidence=overall_confidence,
                abstained=abstention_decision,
                metadata=metadata,
                raw_outputs={
                    "preprocessed_data": preprocessed_data,
                    "features": features,
                    "core_result": core_result
                },
                validation_metrics=validation_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error en análisis universal: {str(e)}")
            metadata.abstention_reasons.append(f"Error en procesamiento: {str(e)}")
            
            return UniversalResult(
                result=None,
                confidence=0.0,
                abstained=True,
                metadata=metadata,
                validation_metrics={}
            )
    
    def _generate_analysis_id(self, input_data: T) -> str:
        """Genera ID único para el análisis"""
        import hashlib
        input_str = str(input_data)
        return hashlib.md5(
            f"{self.domain}_{datetime.now().isoformat()}_{input_str}".encode()
        ).hexdigest()[:12]
    
    def _perform_ensemble_evaluation(self, features: Dict[str, Any], core_result: R) -> List[Dict[str, Any]]:
        """Implementa evaluación ensemble multi-modelo (Meta-Principio 5)"""
        ensemble_results = []
        
        for model_name in self.ensemble_models:
            try:
                # Implementación base - las subclases pueden sobrescribir
                model_result = self._evaluate_with_model(model_name, features, core_result)
                ensemble_results.append({
                    "model": model_name,
                    "result": model_result,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.warning(f"Error en modelo {model_name}: {str(e)}")
                ensemble_results.append({
                    "model": model_name,
                    "result": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return ensemble_results
    
    def _evaluate_with_model(self, model_name: str, features: Dict[str, Any], core_result: R) -> Any:
        """Evalúa con un modelo específico - implementación base"""
        # Las subclases deben implementar la evaluación específica
        return {"model_evaluation": f"Base evaluation for {model_name}"}
    
    def _calculate_uncertainty_bounds(
        self, 
        confidence_metrics: Dict[str, float], 
        ensemble_results: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[float, float]]:
        """Calcula límites de incertidumbre usando bootstrap (Meta-Principio 6)"""
        bounds = {}
        
        # Bootstrap para métricas de confianza
        for metric_name, metric_value in confidence_metrics.items():
            # Simulación bootstrap simple - las subclases pueden implementar bootstrap específico
            std_dev = 0.1 * metric_value  # Estimación conservadora
            lower_bound = max(0.0, metric_value - 1.96 * std_dev)
            upper_bound = min(1.0, metric_value + 1.96 * std_dev)
            bounds[f"{metric_name}_95ci"] = (lower_bound, upper_bound)
        
        # Análisis de variabilidad ensemble
        if ensemble_results:
            valid_results = [r for r in ensemble_results if r.get("result") is not None]
            if len(valid_results) > 1:
                # Calcular variabilidad entre modelos
                ensemble_variance = len(valid_results) / len(ensemble_results)
                bounds["ensemble_agreement"] = (
                    max(0.0, ensemble_variance - 0.1),
                    min(1.0, ensemble_variance + 0.1)
                )
        
        return bounds
    
    def _calculate_overall_confidence(
        self, 
        confidence_metrics: Dict[str, float], 
        ensemble_results: List[Dict[str, Any]]
    ) -> float:
        """Calcula confianza general combinando múltiples métricas"""
        if not confidence_metrics:
            return 0.0
        
        # Promedio ponderado de métricas de confianza
        base_confidence = np.mean(list(confidence_metrics.values()))
        
        # Ajuste por resultados ensemble
        if ensemble_results:
            valid_results = [r for r in ensemble_results if r.get("result") is not None]
            ensemble_factor = len(valid_results) / len(ensemble_results) if ensemble_results else 1.0
            base_confidence *= ensemble_factor
        
        return min(1.0, max(0.0, base_confidence))
    
    def _make_abstention_decision(
        self, 
        confidence: float, 
        uncertainty_bounds: Dict[str, Tuple[float, float]]
    ) -> Tuple[bool, List[str]]:
        """Implementa decisión de abstención matemática (Meta-Principio 1)"""
        reasons = []
        
        if not self.enable_abstention:
            return False, reasons
        
        # Regla 1: Confianza por debajo del umbral
        if confidence < self.confidence_threshold:
            reasons.append(f"Confianza {confidence:.3f} < umbral {self.confidence_threshold}")
        
        # Regla 2: Límites de incertidumbre demasiado amplios
        for bound_name, (lower, upper) in uncertainty_bounds.items():
            if upper - lower > 0.4:  # Intervalo muy amplio
                reasons.append(f"Intervalo {bound_name} muy amplio: [{lower:.3f}, {upper:.3f}]")
        
        # Regla 3: Métricas de confianza inconsistentes
        if uncertainty_bounds.get("ensemble_agreement"):
            lower, upper = uncertainty_bounds["ensemble_agreement"]
            if lower < 0.6:  # Bajo acuerdo entre modelos
                reasons.append(f"Bajo acuerdo ensemble: [{lower:.3f}, {upper:.3f}]")
        
        return len(reasons) > 0, reasons
    
    def _synthesize_result(
        self, 
        core_result: R, 
        ensemble_results: List[Dict[str, Any]], 
        should_abstain: bool
    ) -> R:
        """Síntesis final con hibridación adaptativa (Meta-Principio 8)"""
        if should_abstain:
            return None
        
        # Implementación base: devuelve resultado principal
        # Las subclases pueden implementar hibridación específica
        return core_result
    
    def _perform_validation(self, result: R, features: Dict[str, Any]) -> Dict[str, float]:
        """Validación multi-etapa (Meta-Principio 4)"""
        validation_metrics = {}
        
        if result is None:
            return validation_metrics
        
        # Validación básica - las subclases pueden extender
        validation_metrics["result_not_null"] = 1.0
        validation_metrics["features_available"] = 1.0 if features else 0.0
        
        return validation_metrics

class UniversalFrameworkRegistry:
    """Registry para gestionar múltiples analizadores universales por dominio"""
    
    def __init__(self):
        self._analyzers: Dict[str, UniversalAnalyzer] = {}
    
    def register_analyzer(self, domain: str, analyzer: UniversalAnalyzer):
        """Registra un analizador para un dominio específico"""
        self._analyzers[domain] = analyzer
    
    def get_analyzer(self, domain: str) -> Optional[UniversalAnalyzer]:
        """Obtiene analizador para un dominio"""
        return self._analyzers.get(domain)
    
    def list_domains(self) -> List[str]:
        """Lista dominios disponibles"""
        return list(self._analyzers.keys())
    
    def analyze_with_domain(self, domain: str, input_data: Any) -> Optional[UniversalResult]:
        """Analiza datos con analizador de dominio específico"""
        analyzer = self.get_analyzer(domain)
        if analyzer:
            return analyzer.analyze(input_data)
        return None

# Instancia global del registry
universal_registry = UniversalFrameworkRegistry()