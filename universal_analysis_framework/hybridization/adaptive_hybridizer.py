"""
Universal Adaptive Hybridization System
Sistema de hibridación adaptativa aplicable a cualquier dominio de análisis.

Combina dinámicamente múltiples enfoques, metodologías y fuentes de información
para optimizar la precisión y robustez del análisis según el contexto específico.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict
import json

class HybridizationStrategy(Enum):
    """Estrategias de hibridación disponibles"""
    CONTEXT_ADAPTIVE = "context_adaptive"
    PERFORMANCE_BASED = "performance_based"
    CONFIDENCE_DRIVEN = "confidence_driven"
    DATA_DEPENDENT = "data_dependent"
    TEMPORAL_ADAPTIVE = "temporal_adaptive"
    ENSEMBLE_OPTIMIZED = "ensemble_optimized"
    RISK_BALANCED = "risk_balanced"
    DOMAIN_SPECIFIC = "domain_specific"

class ComponentType(Enum):
    """Tipos de componentes que pueden hibridizarse"""
    ALGORITHM = "algorithm"
    DATA_SOURCE = "data_source"
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    MODEL = "model"
    VALIDATION_METHOD = "validation_method"
    POSTPROCESSING = "postprocessing"
    DECISION_RULE = "decision_rule"

@dataclass
class HybridizationComponent:
    """Componente individual en el sistema de hibridación"""
    component_id: str
    component_type: ComponentType
    implementation: Any  # Función, clase, o cualquier implementación
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    context_suitability: Dict[str, float] = field(default_factory=dict)
    computational_cost: float = 1.0
    reliability_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "performance_metrics": self.performance_metrics,
            "context_suitability": self.context_suitability,
            "computational_cost": self.computational_cost,
            "reliability_score": self.reliability_score,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }

@dataclass
class HybridizationContext:
    """Contexto para decisiones de hibridación"""
    domain: str
    data_characteristics: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    temporal_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "domain": self.domain,
            "data_characteristics": self.data_characteristics,
            "performance_requirements": self.performance_requirements,
            "resource_constraints": self.resource_constraints,
            "quality_requirements": self.quality_requirements,
            "temporal_constraints": self.temporal_constraints
        }

@dataclass
class HybridizationResult:
    """Resultado del proceso de hibridización"""
    selected_components: List[str]
    hybridization_strategy: HybridizationStrategy
    combination_weights: Dict[str, float]
    expected_performance: Dict[str, float]
    confidence_score: float
    adaptation_reasoning: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "selected_components": self.selected_components,
            "hybridization_strategy": self.hybridization_strategy.value,
            "combination_weights": self.combination_weights,
            "expected_performance": self.expected_performance,
            "confidence_score": self.confidence_score,
            "adaptation_reasoning": self.adaptation_reasoning,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }

class AdaptiveSelector(ABC):
    """Interfaz abstracta para selectores adaptativos"""
    
    @abstractmethod
    def select_components(
        self, 
        available_components: List[HybridizationComponent],
        context: HybridizationContext
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Selecciona componentes y calcula pesos
        
        Returns:
            Tuple[List[str], Dict[str, float]]: (component_ids, weights)
        """
        pass

class ContextAdaptiveSelector(AdaptiveSelector):
    """Selector que se adapta basado en el contexto del análisis"""
    
    def select_components(
        self, 
        available_components: List[HybridizationComponent],
        context: HybridizationContext
    ) -> Tuple[List[str], Dict[str, float]]:
        
        selected_ids = []
        weights = {}
        
        # Evaluar cada componente por su idoneidad contextual
        component_scores = []
        
        for component in available_components:
            score = 0.0
            
            # Idoneidad por dominio
            domain_suitability = component.context_suitability.get(context.domain, 0.5)
            score += 0.4 * domain_suitability
            
            # Idoneidad por características de datos
            data_match = 0.0
            for char_name, char_value in context.data_characteristics.items():
                component_match = component.context_suitability.get(f"data_{char_name}", 0.5)
                data_match += component_match
            
            if context.data_characteristics:
                data_match /= len(context.data_characteristics)
                score += 0.3 * data_match
            
            # Consideración de restricciones computacionales
            if 'max_computation_time' in context.resource_constraints:
                max_time = context.resource_constraints['max_computation_time']
                cost_penalty = min(1.0, component.computational_cost / max_time)
                score += 0.2 * (1 - cost_penalty)
            else:
                score += 0.2 * (1 / component.computational_cost)
            
            # Confiabilidad
            score += 0.1 * component.reliability_score
            
            component_scores.append((component.component_id, score, component))
        
        # Seleccionar top componentes
        component_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Selección adaptativa basada en diversidad y complementariedad
        selected_types = set()
        total_weight = 0.0
        
        for comp_id, score, component in component_scores:
            # Promover diversidad de tipos
            if component.component_type not in selected_types or len(selected_ids) < 3:
                selected_ids.append(comp_id)
                weights[comp_id] = score
                total_weight += score
                selected_types.add(component.component_type)
            
            # Límite máximo de componentes
            if len(selected_ids) >= 5:
                break
        
        # Normalizar pesos
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return selected_ids, weights

class PerformanceBasedSelector(AdaptiveSelector):
    """Selector basado en métricas de rendimiento históricas"""
    
    def select_components(
        self, 
        available_components: List[HybridizationComponent],
        context: HybridizationContext
    ) -> Tuple[List[str], Dict[str, float]]:
        
        selected_ids = []
        weights = {}
        
        # Evaluar componentes por rendimiento en métricas relevantes
        relevant_metrics = list(context.performance_requirements.keys())
        
        component_scores = []
        for component in available_components:
            score = 0.0
            
            # Calcular score basado en métricas de rendimiento
            for metric in relevant_metrics:
                if metric in component.performance_metrics:
                    metric_value = component.performance_metrics[metric]
                    required_value = context.performance_requirements[metric]
                    
                    # Ratio de rendimiento (mejor si metric_value >= required_value)
                    performance_ratio = min(1.0, metric_value / max(required_value, 0.001))
                    score += performance_ratio
                else:
                    # Penalizar falta de métricas
                    score += 0.5
            
            if relevant_metrics:
                score /= len(relevant_metrics)
            
            # Ajustar por costo computacional
            score *= (1 / component.computational_cost)
            
            component_scores.append((component.component_id, score, component))
        
        # Seleccionar mejores performers
        component_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Selección top-k con balanceamento
        k = min(len(component_scores), 4)
        for i in range(k):
            comp_id, score, _ = component_scores[i]
            selected_ids.append(comp_id)
            weights[comp_id] = score
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return selected_ids, weights

class UniversalAdaptiveHybridizer:
    """Hibridizador adaptativo universal aplicable a cualquier dominio"""
    
    def __init__(self, default_strategy: HybridizationStrategy = HybridizationStrategy.CONTEXT_ADAPTIVE):
        self.components: Dict[str, HybridizationComponent] = {}
        self.selectors: Dict[HybridizationStrategy, AdaptiveSelector] = {
            HybridizationStrategy.CONTEXT_ADAPTIVE: ContextAdaptiveSelector(),
            HybridizationStrategy.PERFORMANCE_BASED: PerformanceBasedSelector()
        }
        self.default_strategy = default_strategy
        self.adaptation_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("UniversalAdaptiveHybridizer")
    
    def add_component(
        self,
        component_id: str,
        component_type: ComponentType,
        implementation: Any,
        performance_metrics: Optional[Dict[str, float]] = None,
        context_suitability: Optional[Dict[str, float]] = None,
        computational_cost: float = 1.0,
        reliability_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Añade componente al sistema de hibridización"""
        component = HybridizationComponent(
            component_id=component_id,
            component_type=component_type,
            implementation=implementation,
            performance_metrics=performance_metrics or {},
            context_suitability=context_suitability or {},
            computational_cost=computational_cost,
            reliability_score=reliability_score,
            metadata=metadata or {}
        )
        
        self.components[component_id] = component
        self.logger.info(f"Componente añadido: {component_id} ({component_type.value})")
    
    def add_function_component(
        self,
        component_id: str,
        component_type: ComponentType,
        function: Callable,
        performance_metrics: Optional[Dict[str, float]] = None,
        context_suitability: Optional[Dict[str, float]] = None,
        computational_cost: float = 1.0,
        reliability_score: float = 1.0
    ):
        """Añade componente basado en función"""
        metadata = {
            "type": "function",
            "function_name": getattr(function, '__name__', 'anonymous')
        }
        
        self.add_component(
            component_id=component_id,
            component_type=component_type,
            implementation=function,
            performance_metrics=performance_metrics,
            context_suitability=context_suitability,
            computational_cost=computational_cost,
            reliability_score=reliability_score,
            metadata=metadata
        )
    
    def hybridize(
        self,
        context: HybridizationContext,
        strategy: Optional[HybridizationStrategy] = None,
        component_filter: Optional[Callable[[HybridizationComponent], bool]] = None
    ) -> HybridizationResult:
        """
        Realiza hibridización adaptativa basada en el contexto
        
        Args:
            context: Contexto del análisis
            strategy: Estrategia de hibridización (default: self.default_strategy)
            component_filter: Filtro opcional para componentes
        """
        strategy = strategy or self.default_strategy
        
        # Filtrar componentes disponibles
        available_components = list(self.components.values())
        if component_filter:
            available_components = [c for c in available_components if component_filter(c)]
        
        if not available_components:
            raise ValueError("No hay componentes disponibles para hibridización")
        
        # Obtener selector apropiado
        selector = self.selectors.get(strategy)
        if not selector:
            self.logger.warning(f"Estrategia {strategy.value} no disponible, usando {self.default_strategy.value}")
            selector = self.selectors[self.default_strategy]
            strategy = self.default_strategy
        
        # Seleccionar componentes y calcular pesos
        selected_ids, weights = selector.select_components(available_components, context)
        
        # Calcular rendimiento esperado
        expected_performance = self._calculate_expected_performance(selected_ids, context)
        
        # Calcular score de confianza
        confidence_score = self._calculate_confidence_score(selected_ids, weights, context)
        
        # Generar razonamiento de adaptación
        adaptation_reasoning = self._generate_adaptation_reasoning(
            selected_ids, weights, strategy, context
        )
        
        # Crear resultado
        result = HybridizationResult(
            selected_components=selected_ids,
            hybridization_strategy=strategy,
            combination_weights=weights,
            expected_performance=expected_performance,
            confidence_score=confidence_score,
            adaptation_reasoning=adaptation_reasoning,
            metadata={
                "available_components": len(available_components),
                "context": context.to_dict(),
                "selection_timestamp": datetime.now().isoformat()
            }
        )
        
        # Registrar en historial
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": context.to_dict(),
            "result": result.to_dict()
        })
        
        return result
    
    def execute_hybrid_analysis(
        self,
        input_data: Any,
        hybridization_result: HybridizationResult,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta análisis usando los componentes hibridizados seleccionados
        
        Args:
            input_data: Datos de entrada para el análisis
            hybridization_result: Resultado de la hibridización previa
            execution_context: Contexto adicional para la ejecución
        """
        results = {}
        component_outputs = {}
        
        # Ejecutar cada componente seleccionado
        for component_id in hybridization_result.selected_components:
            if component_id not in self.components:
                self.logger.warning(f"Componente {component_id} no encontrado")
                continue
            
            component = self.components[component_id]
            weight = hybridization_result.combination_weights.get(component_id, 1.0)
            
            try:
                # Ejecutar componente
                start_time = datetime.now()
                
                if callable(component.implementation):
                    output = component.implementation(input_data)
                else:
                    # Para implementaciones no callable, intentar método execute
                    if hasattr(component.implementation, 'execute'):
                        output = component.implementation.execute(input_data)
                    else:
                        self.logger.warning(f"No se puede ejecutar componente {component_id}")
                        continue
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                component_outputs[component_id] = {
                    "output": output,
                    "weight": weight,
                    "execution_time": execution_time,
                    "component_type": component.component_type.value
                }
                
            except Exception as e:
                self.logger.error(f"Error ejecutando componente {component_id}: {str(e)}")
                component_outputs[component_id] = {
                    "output": None,
                    "weight": weight,
                    "error": str(e),
                    "component_type": component.component_type.value
                }
        
        # Combinar resultados según pesos
        combined_result = self._combine_component_outputs(
            component_outputs, hybridization_result.hybridization_strategy
        )
        
        return {
            "combined_result": combined_result,
            "component_outputs": component_outputs,
            "hybridization_info": hybridization_result.to_dict(),
            "execution_metadata": {
                "total_components_executed": len(component_outputs),
                "successful_executions": len([c for c in component_outputs.values() if "error" not in c]),
                "total_execution_time": sum([c.get("execution_time", 0) for c in component_outputs.values()]),
                "execution_timestamp": datetime.now().isoformat()
            }
        }
    
    def _calculate_expected_performance(
        self, 
        selected_ids: List[str], 
        context: HybridizationContext
    ) -> Dict[str, float]:
        """Calcula rendimiento esperado de la hibridización"""
        expected_performance = {}
        
        for metric in context.performance_requirements.keys():
            metric_values = []
            
            for comp_id in selected_ids:
                component = self.components[comp_id]
                if metric in component.performance_metrics:
                    metric_values.append(component.performance_metrics[metric])
                else:
                    # Valor conservador para componentes sin métricas
                    metric_values.append(0.5)
            
            if metric_values:
                # Promedio ponderado por confiabilidad
                weights = [self.components[comp_id].reliability_score for comp_id in selected_ids]
                expected_performance[metric] = np.average(metric_values, weights=weights)
        
        return expected_performance
    
    def _calculate_confidence_score(
        self, 
        selected_ids: List[str], 
        weights: Dict[str, float],
        context: HybridizationContext
    ) -> float:
        """Calcula score de confianza en la hibridización"""
        confidence_factors = []
        
        # Factor 1: Diversidad de componentes
        component_types = set(self.components[comp_id].component_type for comp_id in selected_ids)
        diversity_score = len(component_types) / len(ComponentType)
        confidence_factors.append(0.3 * diversity_score)
        
        # Factor 2: Confiabilidad promedio de componentes
        reliability_scores = [self.components[comp_id].reliability_score for comp_id in selected_ids]
        avg_reliability = np.mean(reliability_scores)
        confidence_factors.append(0.4 * avg_reliability)
        
        # Factor 3: Balance de pesos
        weight_values = list(weights.values())
        weight_balance = 1 - np.std(weight_values) if len(weight_values) > 1 else 1.0
        confidence_factors.append(0.2 * weight_balance)
        
        # Factor 4: Cobertura de requisitos
        coverage_score = len(selected_ids) / min(len(self.components), 5)
        confidence_factors.append(0.1 * coverage_score)
        
        return sum(confidence_factors)
    
    def _generate_adaptation_reasoning(
        self, 
        selected_ids: List[str], 
        weights: Dict[str, float],
        strategy: HybridizationStrategy,
        context: HybridizationContext
    ) -> List[str]:
        """Genera explicación del razonamiento de adaptación"""
        reasoning = []
        
        reasoning.append(f"Estrategia de hibridización: {strategy.value}")
        reasoning.append(f"Dominio objetivo: {context.domain}")
        reasoning.append(f"Componentes seleccionados: {len(selected_ids)}")
        
        # Razones por componentes más importantes
        sorted_components = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for i, (comp_id, weight) in enumerate(sorted_components[:3]):
            component = self.components[comp_id]
            reasoning.append(
                f"Componente #{i+1}: {comp_id} ({component.component_type.value}) - "
                f"peso: {weight:.3f}, confiabilidad: {component.reliability_score:.3f}"
            )
        
        # Consideraciones especiales
        if any(self.components[comp_id].computational_cost > 2.0 for comp_id in selected_ids):
            reasoning.append("Se priorizó calidad sobre eficiencia computacional")
        
        if len(set(self.components[comp_id].component_type for comp_id in selected_ids)) > 2:
            reasoning.append("Se optimizó diversidad de enfoques metodológicos")
        
        return reasoning
    
    def _combine_component_outputs(
        self, 
        component_outputs: Dict[str, Dict[str, Any]], 
        strategy: HybridizationStrategy
    ) -> Any:
        """Combina salidas de componentes según la estrategia"""
        successful_outputs = {
            comp_id: output for comp_id, output in component_outputs.items() 
            if "error" not in output and output["output"] is not None
        }
        
        if not successful_outputs:
            return None
        
        # Estrategia de combinación simple - las subclases pueden sobrescribir
        if len(successful_outputs) == 1:
            return list(successful_outputs.values())[0]["output"]
        
        # Para múltiples salidas, intentar combinación inteligente
        outputs = [output["output"] for output in successful_outputs.values()]
        weights = [output["weight"] for output in successful_outputs.values()]
        
        # Si son valores numéricos, hacer promedio ponderado
        try:
            numeric_outputs = [float(output) for output in outputs]
            return np.average(numeric_outputs, weights=weights)
        except (ValueError, TypeError):
            pass
        
        # Si son categóricos, voto ponderado
        try:
            from collections import defaultdict
            vote_scores = defaultdict(float)
            
            for output, weight in zip(outputs, weights):
                vote_scores[str(output)] += weight
            
            return max(vote_scores.items(), key=lambda x: x[1])[0]
        except:
            pass
        
        # Fallback: devolver salida con mayor peso
        max_weight_idx = np.argmax(weights)
        return outputs[max_weight_idx]
    
    def learn_from_feedback(
        self, 
        hybridization_id: str, 
        actual_performance: Dict[str, float],
        feedback_context: Optional[Dict[str, Any]] = None
    ):
        """Aprende de feedback para mejorar futuras hibridizaciones"""
        # Buscar la hibridización correspondiente en el historial
        matching_entry = None
        for entry in self.adaptation_history:
            if entry.get("result", {}).get("metadata", {}).get("selection_timestamp") == hybridization_id:
                matching_entry = entry
                break
        
        if not matching_entry:
            self.logger.warning(f"No se encontró hibridización {hybridization_id} en historial")
            return
        
        # Actualizar métricas de rendimiento de componentes
        selected_components = matching_entry["result"]["selected_components"]
        
        for comp_id in selected_components:
            if comp_id in self.components:
                component = self.components[comp_id]
                
                # Actualizar métricas de rendimiento
                for metric, value in actual_performance.items():
                    if metric in component.performance_metrics:
                        # Promedio ponderado con métricas anteriores
                        old_value = component.performance_metrics[metric]
                        component.performance_metrics[metric] = 0.8 * old_value + 0.2 * value
                    else:
                        component.performance_metrics[metric] = value
                
                # Ajustar confiabilidad basada en rendimiento
                performance_score = np.mean(list(actual_performance.values()))
                reliability_adjustment = 0.1 * (performance_score - 0.5)  # [-0.05, +0.05]
                component.reliability_score = np.clip(
                    component.reliability_score + reliability_adjustment, 0.0, 1.0
                )
        
        self.logger.info(f"Aprendizaje completado para hibridización {hybridization_id}")
    
    def get_adaptation_insights(self) -> Dict[str, Any]:
        """Obtiene insights sobre el comportamiento de adaptación"""
        if not self.adaptation_history:
            return {"message": "No hay historial de adaptación disponible"}
        
        insights = {}
        
        # Estrategias más utilizadas
        strategies = [entry["result"]["hybridization_strategy"] for entry in self.adaptation_history]
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        insights["most_used_strategies"] = strategy_counts
        
        # Componentes más seleccionados
        all_components = []
        for entry in self.adaptation_history:
            all_components.extend(entry["result"]["selected_components"])
        
        component_counts = {}
        for comp_id in all_components:
            component_counts[comp_id] = component_counts.get(comp_id, 0) + 1
        
        insights["most_selected_components"] = dict(sorted(
            component_counts.items(), key=lambda x: x[1], reverse=True
        )[:10])
        
        # Scores de confianza promedio
        confidence_scores = [entry["result"]["confidence_score"] for entry in self.adaptation_history]
        insights["average_confidence"] = np.mean(confidence_scores)
        insights["confidence_trend"] = confidence_scores[-10:] if len(confidence_scores) >= 10 else confidence_scores
        
        # Dominios más frecuentes
        domains = [entry["context"]["domain"] for entry in self.adaptation_history]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        insights["domain_distribution"] = domain_counts
        
        return insights

# Instancia global del hibridizador adaptativo
universal_adaptive_hybridizer = UniversalAdaptiveHybridizer()