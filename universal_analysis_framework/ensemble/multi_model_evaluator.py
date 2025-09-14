"""
Universal Multi-Model Ensemble Evaluation System
Sistema universal de evaluación ensemble aplicable a cualquier dominio de análisis.

Permite integrar múltiples modelos, métodos o enfoques de análisis 
para obtener resultados más robustos y confiables.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib

class EnsembleStrategy(Enum):
    """Estrategias de combinación ensemble"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average" 
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    DYNAMIC_SELECTION = "dynamic_selection"

class ModelType(Enum):
    """Tipos de modelos en el ensemble"""
    MACHINE_LEARNING = "ml_model"
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    HEURISTIC = "heuristic"
    HUMAN_EXPERT = "human_expert"
    EXTERNAL_API = "external_api"
    HYBRID = "hybrid"

@dataclass
class ModelResult:
    """Resultado de un modelo individual en el ensemble"""
    model_id: str
    model_type: ModelType
    result: Any
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "result": self.result,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "error": self.error,
            "timestamp": datetime.now().isoformat()
        }

@dataclass
class EnsembleResult:
    """Resultado del análisis ensemble"""
    final_result: Any
    overall_confidence: float
    strategy_used: EnsembleStrategy
    individual_results: List[ModelResult] = field(default_factory=list)
    consensus_metrics: Dict[str, float] = field(default_factory=dict)
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    combination_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "final_result": self.final_result,
            "overall_confidence": self.overall_confidence,
            "strategy_used": self.strategy_used.value,
            "individual_results": [r.to_dict() for r in self.individual_results],
            "consensus_metrics": self.consensus_metrics,
            "diversity_metrics": self.diversity_metrics,
            "combination_metadata": self.combination_metadata,
            "num_models": len(self.individual_results),
            "successful_models": len([r for r in self.individual_results if r.error is None]),
            "average_confidence": np.mean([r.confidence for r in self.individual_results if r.error is None]),
            "timestamp": datetime.now().isoformat()
        }

class UniversalModel(ABC):
    """Interfaz abstracta para modelos universales en el ensemble"""
    
    def __init__(self, model_id: str, model_type: ModelType):
        self.model_id = model_id
        self.model_type = model_type
        self.logger = logging.getLogger(f"UniversalModel_{model_id}")
    
    @abstractmethod
    def predict(self, input_data: Any) -> Tuple[Any, float]:
        """
        Realiza predicción y devuelve (resultado, confianza)
        
        Returns:
            Tuple[Any, float]: (resultado, confianza entre 0 y 1)
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve metadatos del modelo"""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """Valida datos de entrada (implementación base)"""
        return input_data is not None

class FunctionBasedModel(UniversalModel):
    """Modelo basado en función para fácil integración"""
    
    def __init__(
        self, 
        model_id: str, 
        prediction_function: Callable[[Any], Tuple[Any, float]],
        model_type: ModelType = ModelType.HYBRID,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_id, model_type)
        self.prediction_function = prediction_function
        self.metadata = metadata or {}
    
    def predict(self, input_data: Any) -> Tuple[Any, float]:
        """Ejecuta función de predicción"""
        return self.prediction_function(input_data)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve metadatos del modelo función"""
        return {
            "model_type": "function_based",
            "function_name": getattr(self.prediction_function, '__name__', 'anonymous'),
            **self.metadata
        }

class UniversalEnsembleEvaluator:
    """Evaluador ensemble universal aplicable a cualquier dominio"""
    
    def __init__(
        self,
        models: Optional[List[UniversalModel]] = None,
        default_strategy: EnsembleStrategy = EnsembleStrategy.CONFIDENCE_WEIGHTED,
        parallel_execution: bool = True,
        timeout_seconds: float = 30.0
    ):
        self.models = models or []
        self.default_strategy = default_strategy
        self.parallel_execution = parallel_execution
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger("UniversalEnsembleEvaluator")
        
    def add_model(self, model: UniversalModel):
        """Añade modelo al ensemble"""
        self.models.append(model)
        self.logger.info(f"Modelo añadido: {model.model_id} ({model.model_type.value})")
    
    def add_function_model(
        self, 
        model_id: str, 
        prediction_function: Callable[[Any], Tuple[Any, float]],
        model_type: ModelType = ModelType.HYBRID,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Añade modelo basado en función"""
        model = FunctionBasedModel(model_id, prediction_function, model_type, metadata)
        self.add_model(model)
    
    def evaluate(
        self, 
        input_data: Any, 
        strategy: Optional[EnsembleStrategy] = None,
        model_weights: Optional[Dict[str, float]] = None
    ) -> EnsembleResult:
        """
        Evalúa input con todos los modelos del ensemble
        
        Args:
            input_data: Datos de entrada para análisis
            strategy: Estrategia de combinación (default: self.default_strategy)
            model_weights: Pesos específicos por modelo (para weighted_average)
        """
        if not self.models:
            raise ValueError("No hay modelos en el ensemble")
        
        strategy = strategy or self.default_strategy
        
        # Ejecutar modelos (en paralelo o secuencial)
        individual_results = self._execute_models(input_data)
        
        # Calcular métricas de consenso y diversidad
        consensus_metrics = self._calculate_consensus_metrics(individual_results)
        diversity_metrics = self._calculate_diversity_metrics(individual_results)
        
        # Combinar resultados según estrategia
        final_result, overall_confidence, combination_metadata = self._combine_results(
            individual_results, strategy, model_weights
        )
        
        return EnsembleResult(
            final_result=final_result,
            overall_confidence=overall_confidence,
            strategy_used=strategy,
            individual_results=individual_results,
            consensus_metrics=consensus_metrics,
            diversity_metrics=diversity_metrics,
            combination_metadata=combination_metadata
        )
    
    def _execute_models(self, input_data: Any) -> List[ModelResult]:
        """Ejecuta todos los modelos del ensemble"""
        results = []
        
        if self.parallel_execution:
            results = self._execute_models_parallel(input_data)
        else:
            results = self._execute_models_sequential(input_data)
        
        return results
    
    def _execute_models_parallel(self, input_data: Any) -> List[ModelResult]:
        """Ejecuta modelos en paralelo"""
        results = []
        
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            # Enviar tareas
            future_to_model = {
                executor.submit(self._execute_single_model, model, input_data): model 
                for model in self.models
            }
            
            # Recopilar resultados
            for future in as_completed(future_to_model, timeout=self.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    model = future_to_model[future]
                    self.logger.error(f"Error ejecutando {model.model_id}: {str(e)}")
                    results.append(ModelResult(
                        model_id=model.model_id,
                        model_type=model.model_type,
                        result=None,
                        confidence=0.0,
                        processing_time=0.0,
                        error=str(e)
                    ))
        
        return results
    
    def _execute_models_sequential(self, input_data: Any) -> List[ModelResult]:
        """Ejecuta modelos secuencialmente"""
        results = []
        
        for model in self.models:
            try:
                result = self._execute_single_model(model, input_data)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error ejecutando {model.model_id}: {str(e)}")
                results.append(ModelResult(
                    model_id=model.model_id,
                    model_type=model.model_type,
                    result=None,
                    confidence=0.0,
                    processing_time=0.0,
                    error=str(e)
                ))
        
        return results
    
    def _execute_single_model(self, model: UniversalModel, input_data: Any) -> ModelResult:
        """Ejecuta un modelo individual"""
        start_time = datetime.now()
        
        try:
            # Validar entrada
            if not model.validate_input(input_data):
                raise ValueError("Datos de entrada no válidos")
            
            # Ejecutar predicción
            result, confidence = model.predict(input_data)
            
            # Calcular tiempo de procesamiento
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ModelResult(
                model_id=model.model_id,
                model_type=model.model_type,
                result=result,
                confidence=confidence,
                processing_time=processing_time,
                metadata=model.get_metadata()
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            raise e
    
    def _calculate_consensus_metrics(self, results: List[ModelResult]) -> Dict[str, float]:
        """Calcula métricas de consenso entre modelos"""
        successful_results = [r for r in results if r.error is None]
        
        if len(successful_results) < 2:
            return {"consensus_impossible": 1.0, "num_successful": len(successful_results)}
        
        metrics = {}
        
        # Consenso en confianza
        confidences = [r.confidence for r in successful_results]
        metrics["confidence_mean"] = np.mean(confidences)
        metrics["confidence_std"] = np.std(confidences)
        metrics["confidence_consensus"] = 1 - (np.std(confidences) / max(np.mean(confidences), 0.01))
        
        # Consenso en resultados (para resultados comparables)
        try:
            if self._are_numeric_results(successful_results):
                metrics.update(self._calculate_numeric_consensus(successful_results))
            elif self._are_categorical_results(successful_results):
                metrics.update(self._calculate_categorical_consensus(successful_results))
        except Exception as e:
            self.logger.warning(f"Error calculando consenso de resultados: {str(e)}")
        
        # Métricas generales
        metrics["success_rate"] = len(successful_results) / len(results)
        metrics["num_successful"] = len(successful_results)
        metrics["num_total"] = len(results)
        
        return metrics
    
    def _calculate_diversity_metrics(self, results: List[ModelResult]) -> Dict[str, float]:
        """Calcula métricas de diversidad del ensemble"""
        successful_results = [r for r in results if r.error is None]
        
        metrics = {}
        
        # Diversidad de tipos de modelo
        model_types = [r.model_type.value for r in successful_results]
        unique_types = len(set(model_types))
        metrics["type_diversity"] = unique_types / len(successful_results) if successful_results else 0
        
        # Diversidad de tiempos de procesamiento
        if successful_results:
            processing_times = [r.processing_time for r in successful_results]
            metrics["time_diversity"] = np.std(processing_times) / max(np.mean(processing_times), 0.001)
        
        # Diversidad de confianzas
        if successful_results:
            confidences = [r.confidence for r in successful_results]
            metrics["confidence_diversity"] = np.std(confidences)
        
        return metrics
    
    def _are_numeric_results(self, results: List[ModelResult]) -> bool:
        """Verifica si los resultados son numéricos"""
        try:
            for result in results[:3]:  # Test con primeros 3
                float(result.result)
            return True
        except (ValueError, TypeError):
            return False
    
    def _are_categorical_results(self, results: List[ModelResult]) -> bool:
        """Verifica si los resultados son categóricos"""
        result_values = [str(r.result) for r in results]
        return len(set(result_values)) < len(result_values)
    
    def _calculate_numeric_consensus(self, results: List[ModelResult]) -> Dict[str, float]:
        """Calcula consenso para resultados numéricos"""
        values = [float(r.result) for r in results]
        
        return {
            "numeric_mean": np.mean(values),
            "numeric_std": np.std(values),
            "numeric_consensus": 1 - (np.std(values) / max(np.mean(np.abs(values)), 0.01)),
            "numeric_range": np.max(values) - np.min(values)
        }
    
    def _calculate_categorical_consensus(self, results: List[ModelResult]) -> Dict[str, float]:
        """Calcula consenso para resultados categóricos"""
        from collections import Counter
        
        values = [str(r.result) for r in results]
        counter = Counter(values)
        most_common_count = counter.most_common(1)[0][1]
        
        return {
            "categorical_agreement": most_common_count / len(values),
            "num_unique_results": len(counter),
            "majority_result": counter.most_common(1)[0][0],
            "entropy": self._calculate_categorical_entropy(counter)
        }
    
    def _calculate_categorical_entropy(self, counter: 'Counter') -> float:
        """Calcula entropía de resultados categóricos"""
        total = sum(counter.values())
        probs = [count/total for count in counter.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def _combine_results(
        self, 
        results: List[ModelResult], 
        strategy: EnsembleStrategy,
        model_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """Combina resultados según la estrategia especificada"""
        successful_results = [r for r in results if r.error is None]
        
        if not successful_results:
            return None, 0.0, {"error": "No hay resultados válidos"}
        
        combination_metadata = {"strategy": strategy.value, "num_combined": len(successful_results)}
        
        if strategy == EnsembleStrategy.SIMPLE_AVERAGE:
            return self._simple_average_combination(successful_results, combination_metadata)
        elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_combination(successful_results, model_weights, combination_metadata)
        elif strategy == EnsembleStrategy.MAJORITY_VOTE:
            return self._majority_vote_combination(successful_results, combination_metadata)
        elif strategy == EnsembleStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_combination(successful_results, combination_metadata)
        elif strategy == EnsembleStrategy.DYNAMIC_SELECTION:
            return self._dynamic_selection_combination(successful_results, combination_metadata)
        else:
            # Fallback a confidence weighted
            return self._confidence_weighted_combination(successful_results, combination_metadata)
    
    def _simple_average_combination(
        self, 
        results: List[ModelResult], 
        metadata: Dict[str, Any]
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """Combinación por promedio simple"""
        if self._are_numeric_results(results):
            values = [float(r.result) for r in results]
            combined_result = np.mean(values)
            metadata["method"] = "numeric_mean"
        else:
            # Para no numéricos, usar mayoría
            from collections import Counter
            values = [str(r.result) for r in results]
            combined_result = Counter(values).most_common(1)[0][0]
            metadata["method"] = "majority_fallback"
        
        # Confianza promedio
        combined_confidence = np.mean([r.confidence for r in results])
        
        return combined_result, combined_confidence, metadata
    
    def _weighted_average_combination(
        self, 
        results: List[ModelResult], 
        model_weights: Optional[Dict[str, float]],
        metadata: Dict[str, Any]
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """Combinación por promedio ponderado"""
        if model_weights is None:
            # Usar confianzas como pesos
            model_weights = {r.model_id: r.confidence for r in results}
        
        weights = [model_weights.get(r.model_id, 1.0) for r in results]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return self._simple_average_combination(results, metadata)
        
        if self._are_numeric_results(results):
            values = [float(r.result) for r in results]
            combined_result = np.average(values, weights=weights)
            metadata["method"] = "numeric_weighted_average"
        else:
            # Para no numéricos, seleccionar el de mayor peso
            max_weight_idx = np.argmax(weights)
            combined_result = results[max_weight_idx].result
            metadata["method"] = "max_weight_selection"
        
        # Confianza ponderada
        combined_confidence = np.average([r.confidence for r in results], weights=weights)
        metadata["weights_used"] = dict(zip([r.model_id for r in results], weights))
        
        return combined_result, combined_confidence, metadata
    
    def _majority_vote_combination(
        self, 
        results: List[ModelResult], 
        metadata: Dict[str, Any]
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """Combinación por voto mayoritario"""
        from collections import Counter
        
        values = [str(r.result) for r in results]
        counter = Counter(values)
        majority_result, majority_count = counter.most_common(1)[0]
        
        # Confianza basada en consenso
        consensus_ratio = majority_count / len(results)
        
        # Confianza promedio de los que votaron por la mayoría
        majority_confidences = [r.confidence for r in results if str(r.result) == majority_result]
        combined_confidence = np.mean(majority_confidences) * consensus_ratio
        
        metadata.update({
            "method": "majority_vote",
            "majority_count": majority_count,
            "consensus_ratio": consensus_ratio,
            "vote_distribution": dict(counter)
        })
        
        # Intentar convertir de vuelta al tipo original si es posible
        try:
            if self._are_numeric_results(results):
                majority_result = float(majority_result)
        except:
            pass
        
        return majority_result, combined_confidence, metadata
    
    def _confidence_weighted_combination(
        self, 
        results: List[ModelResult], 
        metadata: Dict[str, Any]
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """Combinación ponderada por confianza"""
        confidences = [r.confidence for r in results]
        total_confidence = sum(confidences)
        
        if total_confidence == 0:
            return self._simple_average_combination(results, metadata)
        
        if self._are_numeric_results(results):
            values = [float(r.result) for r in results]
            combined_result = np.average(values, weights=confidences)
            metadata["method"] = "confidence_weighted_average"
        else:
            # Seleccionar el resultado con mayor confianza
            max_conf_idx = np.argmax(confidences)
            combined_result = results[max_conf_idx].result
            metadata["method"] = "max_confidence_selection"
            metadata["selected_model"] = results[max_conf_idx].model_id
        
        # Confianza combinada con ajuste por diversidad
        combined_confidence = np.average(confidences, weights=confidences)
        
        # Ajuste por consenso
        if len(results) > 1:
            consensus_factor = 1 - (np.std(confidences) / max(np.mean(confidences), 0.01))
            combined_confidence *= consensus_factor
        
        metadata.update({
            "confidence_weights": confidences,
            "total_confidence": total_confidence,
            "consensus_factor": consensus_factor if len(results) > 1 else 1.0
        })
        
        return combined_result, combined_confidence, metadata
    
    def _dynamic_selection_combination(
        self, 
        results: List[ModelResult], 
        metadata: Dict[str, Any]
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """Selección dinámica basada en múltiples criterios"""
        # Criterios: confianza, tipo de modelo, tiempo de procesamiento
        scores = []
        
        for result in results:
            score = 0.0
            
            # Componente de confianza (60%)
            score += 0.6 * result.confidence
            
            # Componente de tipo de modelo (20%)
            type_bonus = {
                ModelType.MACHINE_LEARNING: 0.2,
                ModelType.STATISTICAL: 0.15,
                ModelType.HYBRID: 0.18,
                ModelType.RULE_BASED: 0.12,
                ModelType.HEURISTIC: 0.10,
                ModelType.HUMAN_EXPERT: 0.25,
                ModelType.EXTERNAL_API: 0.14
            }
            score += type_bonus.get(result.model_type, 0.1)
            
            # Componente de eficiencia temporal (20%)
            # Modelos más rápidos obtienen bonus
            max_time = max([r.processing_time for r in results])
            if max_time > 0:
                time_efficiency = 1 - (result.processing_time / max_time)
                score += 0.2 * time_efficiency
            
            scores.append(score)
        
        # Seleccionar el mejor
        best_idx = np.argmax(scores)
        selected_result = results[best_idx]
        
        metadata.update({
            "method": "dynamic_selection",
            "selection_scores": scores,
            "selected_model": selected_result.model_id,
            "selected_score": scores[best_idx]
        })
        
        return selected_result.result, selected_result.confidence, metadata

# Instancia global del evaluador ensemble
universal_ensemble_evaluator = UniversalEnsembleEvaluator()