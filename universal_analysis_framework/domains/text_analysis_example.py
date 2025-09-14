"""
Ejemplo de implementación del Universal Analysis Framework para Análisis de Texto
Demuestra cómo aplicar los 8 meta-principios universales al dominio de análisis textual.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import logging
from datetime import datetime
import re
from collections import Counter
import hashlib

# Imports del framework universal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.universal_framework import UniversalAnalyzer, UniversalResult, universal_registry
from mathematical.abstention_framework import (
    UniversalMathematicalFramework, BoundCalculationMethod, 
    universal_math_framework, universal_uncertainty_quantifier
)
from ensemble.multi_model_evaluator import (
    UniversalModel, ModelType, universal_ensemble_evaluator
)
from genealogical.influence_tracker import (
    UniversalInfluenceTracker, NodeType, InfluenceType
)
from hybridization.adaptive_hybridizer import (
    UniversalAdaptiveHybridizer, HybridizationContext, ComponentType,
    universal_adaptive_hybridizer
)

class TextAnalysisResult:
    """Resultado específico del análisis de texto"""
    def __init__(self):
        self.sentiment_score: float = 0.0
        self.sentiment_label: str = "neutral"
        self.key_topics: List[str] = []
        self.readability_score: float = 0.0
        self.complexity_metrics: Dict[str, float] = {}
        self.language_detected: str = "unknown"
        self.quality_indicators: Dict[str, float] = {}

class SimpleTextAnalyzer(UniversalAnalyzer[str, TextAnalysisResult]):
    """Analizador de texto que implementa los 8 meta-principios universales"""
    
    def __init__(self):
        super().__init__(
            domain="text_analysis",
            confidence_threshold=0.75,
            enable_abstention=True,
            bootstrap_iterations=500,
            ensemble_models=["sentiment_model", "topic_model", "readability_model"]
        )
        
        # Configurar influence tracker
        self.influence_tracker = UniversalInfluenceTracker("text_analysis")
        
        # Configurar hibridizador
        self._setup_hybridization_components()
    
    def _setup_hybridization_components(self):
        """Configura componentes de hibridización para análisis de texto"""
        
        # Componente de análisis de sentimiento
        def sentiment_analysis(text: str) -> Tuple[Dict[str, Any], float]:
            # Análisis de sentimiento simple basado en palabras clave
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
            negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate"]
            
            words = text.lower().split()
            pos_count = sum(1 for word in words if any(pw in word for pw in positive_words))
            neg_count = sum(1 for word in words if any(nw in word for nw in negative_words))
            
            total_sentiment_words = pos_count + neg_count
            if total_sentiment_words == 0:
                return {"sentiment": "neutral", "score": 0.0}, 0.5
            
            sentiment_score = (pos_count - neg_count) / len(words)
            
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            confidence = min(1.0, total_sentiment_words / max(len(words) * 0.1, 1))
            
            return {
                "sentiment": sentiment_label,
                "score": sentiment_score,
                "positive_count": pos_count,
                "negative_count": neg_count
            }, confidence
        
        universal_adaptive_hybridizer.add_function_component(
            component_id="sentiment_analyzer",
            component_type=ComponentType.ALGORITHM,
            function=sentiment_analysis,
            performance_metrics={"accuracy": 0.75, "precision": 0.70},
            context_suitability={"text_analysis": 0.9, "sentiment": 0.95},
            computational_cost=1.0,
            reliability_score=0.8
        )
        
        # Componente de extracción de temas
        def topic_extraction(text: str) -> Tuple[List[str], float]:
            # Extracción simple de temas basada en frecuencia de palabras
            words = re.findall(r'\b\w+\b', text.lower())
            # Filtrar palabras comunes
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            words = [w for w in words if len(w) > 3 and w not in stop_words]
            
            word_freq = Counter(words)
            topics = [word for word, freq in word_freq.most_common(5)]
            
            confidence = min(1.0, len(topics) / 5.0) if topics else 0.0
            
            return topics, confidence
        
        universal_adaptive_hybridizer.add_function_component(
            component_id="topic_extractor",
            component_type=ComponentType.FEATURE_EXTRACTION,
            function=topic_extraction,
            performance_metrics={"relevance": 0.65, "coverage": 0.70},
            context_suitability={"text_analysis": 0.85, "topic_modeling": 0.90},
            computational_cost=1.2,
            reliability_score=0.7
        )
        
        # Componente de análisis de legibilidad
        def readability_analysis(text: str) -> Tuple[float, float]:
            # Análisis simple de legibilidad
            sentences = text.split('.')
            words = text.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0, 0.0
            
            avg_sentence_length = len(words) / max(len(sentences), 1)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Score simple de legibilidad (inverso de la complejidad)
            readability_score = max(0.0, 1.0 - (avg_sentence_length / 30.0) - (avg_word_length / 10.0))
            confidence = 0.8
            
            return readability_score, confidence
        
        universal_adaptive_hybridizer.add_function_component(
            component_id="readability_analyzer",
            component_type=ComponentType.ALGORITHM,
            function=readability_analysis,
            performance_metrics={"correlation": 0.60, "consistency": 0.85},
            context_suitability={"text_analysis": 0.80, "readability": 0.95},
            computational_cost=0.8,
            reliability_score=0.85
        )
    
    def preprocess_input(self, input_data: str) -> Dict[str, Any]:
        """Preprocesa el texto de entrada"""
        # Rastrear paso de preprocesamiento
        _, process_id, output_id = self.influence_tracker.track_processing_step(
            "text_preprocessing",
            input_data,
            {"cleaned_text": input_data.strip(), "original_length": len(input_data)},
            "text_cleaning_and_normalization"
        )
        
        return {
            "original_text": input_data,
            "cleaned_text": input_data.strip(),
            "text_length": len(input_data),
            "word_count": len(input_data.split()),
            "sentence_count": len(input_data.split('.')),
            "preprocessing_node_id": output_id
        }
    
    def extract_features(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae características del texto"""
        text = preprocessed_data["cleaned_text"]
        
        # Características básicas
        features = {
            "char_count": len(text),
            "word_count": preprocessed_data["word_count"],
            "sentence_count": preprocessed_data["sentence_count"],
            "avg_word_length": sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            "avg_sentence_length": preprocessed_data["word_count"] / max(preprocessed_data["sentence_count"], 1),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "punctuation_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        }
        
        # Rastrear extracción de características
        _, process_id, output_id = self.influence_tracker.track_processing_step(
            "feature_extraction",
            preprocessed_data,
            features,
            "text_feature_extraction"
        )
        
        features["feature_extraction_node_id"] = output_id
        return features
    
    def perform_core_analysis(self, features: Dict[str, Any]) -> TextAnalysisResult:
        """Realiza análisis principal del texto"""
        text = features.get("original_text", "")
        
        result = TextAnalysisResult()
        
        # Usar hibridización para análisis
        context = HybridizationContext(
            domain="text_analysis",
            data_characteristics={
                "text_length": features["char_count"],
                "complexity": features["avg_sentence_length"]
            },
            performance_requirements={
                "accuracy": 0.75,
                "speed": 0.8
            }
        )
        
        # Hibridización para análisis completo
        hybridization_result = universal_adaptive_hybridizer.hybridize(context)
        
        # Ejecutar análisis hibridizado (simulado)
        if "sentiment_analyzer" in hybridization_result.selected_components:
            # Análisis de sentimiento simple
            sentiment_words = ["good", "great", "bad", "terrible", "amazing", "awful"]
            sentiment_count = sum(1 for word in text.lower().split() if word in sentiment_words)
            result.sentiment_score = min(1.0, sentiment_count / max(len(text.split()) * 0.1, 1))
            
            if result.sentiment_score > 0.6:
                result.sentiment_label = "positive"
            elif result.sentiment_score < 0.3:
                result.sentiment_label = "negative"
            else:
                result.sentiment_label = "neutral"
        
        if "topic_extractor" in hybridization_result.selected_components:
            # Extracción de temas
            words = text.lower().split()
            word_freq = Counter(word for word in words if len(word) > 4)
            result.key_topics = [word for word, _ in word_freq.most_common(3)]
        
        if "readability_analyzer" in hybridization_result.selected_components:
            # Análisis de legibilidad
            result.readability_score = max(0.0, 1.0 - features["avg_sentence_length"] / 30.0)
        
        # Métricas de complejidad
        result.complexity_metrics = {
            "lexical_diversity": len(set(text.lower().split())) / max(len(text.split()), 1),
            "sentence_complexity": features["avg_sentence_length"],
            "vocabulary_complexity": features["avg_word_length"]
        }
        
        # Detectar idioma (simplificado)
        english_indicators = ["the", "and", "or", "but", "with", "from"]
        spanish_indicators = ["el", "la", "y", "o", "pero", "con", "de"]
        
        english_count = sum(1 for word in english_indicators if word in text.lower())
        spanish_count = sum(1 for word in spanish_indicators if word in text.lower())
        
        if english_count > spanish_count:
            result.language_detected = "english"
        elif spanish_count > english_count:
            result.language_detected = "spanish"
        else:
            result.language_detected = "unknown"
        
        # Indicadores de calidad
        result.quality_indicators = {
            "completeness": min(1.0, len(text) / 100),
            "coherence": min(1.0, result.readability_score * 1.2),
            "information_density": len(result.key_topics) / 5.0
        }
        
        # Rastrear análisis principal
        analysis_node = self.influence_tracker.add_node(
            "core_text_analysis",
            NodeType.PROCESSING_STEP,
            "comprehensive_text_analysis",
            processing_stage="core_analysis"
        )
        
        result_node = self.influence_tracker.add_node(
            "text_analysis_result",
            NodeType.FINAL_RESULT,
            result,
            processing_stage="final_output"
        )
        
        # Influencias del análisis
        self.influence_tracker.add_influence(
            features["feature_extraction_node_id"],
            analysis_node,
            InfluenceType.DATA_DEPENDENCY,
            1.0
        )
        
        self.influence_tracker.add_influence(
            analysis_node,
            result_node,
            InfluenceType.DIRECT_CAUSAL,
            0.95
        )
        
        return result
    
    def calculate_confidence_metrics(self, result: TextAnalysisResult, features: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de confianza específicas del análisis de texto"""
        metrics = {}
        
        # Confianza en análisis de sentimiento
        if result.sentiment_label != "neutral":
            sentiment_confidence = min(1.0, abs(result.sentiment_score) * 2)
        else:
            sentiment_confidence = 0.8 if abs(result.sentiment_score) < 0.1 else 0.5
        
        metrics["sentiment_confidence"] = sentiment_confidence
        
        # Confianza en extracción de temas
        topic_confidence = min(1.0, len(result.key_topics) / 3.0) if result.key_topics else 0.0
        metrics["topic_confidence"] = topic_confidence
        
        # Confianza en análisis de legibilidad
        readability_confidence = 0.85  # Alta confianza en métricas objetivas
        metrics["readability_confidence"] = readability_confidence
        
        # Confianza en detección de idioma
        if result.language_detected in ["english", "spanish"]:
            language_confidence = 0.8
        else:
            language_confidence = 0.3
        metrics["language_confidence"] = language_confidence
        
        # Confianza general basada en longitud del texto
        text_length_factor = min(1.0, features["char_count"] / 200)
        metrics["text_length_confidence"] = text_length_factor
        
        # Confianza en calidad general
        quality_scores = list(result.quality_indicators.values())
        overall_quality = np.mean(quality_scores) if quality_scores else 0.5
        metrics["overall_quality_confidence"] = overall_quality
        
        return metrics
    
    def perform_genealogical_analysis(self, input_data: str, result: TextAnalysisResult) -> Dict[str, Any]:
        """Analiza las influencias genealógicas del análisis de texto"""
        
        # Añadir nodo de entrada original
        input_node = self.influence_tracker.add_node(
            "original_text_input",
            NodeType.INPUT_DATA,
            input_data[:100] + "..." if len(input_data) > 100 else input_data
        )
        
        # Realizar análisis genealógico completo
        genealogy_analysis = self.influence_tracker.analyze_genealogy()
        
        # Encontrar influencias críticas
        critical_influences = self.influence_tracker.find_critical_influences(importance_threshold=0.7)
        
        return {
            "genealogy_summary": {
                "total_nodes": len(genealogy_analysis.nodes),
                "total_relations": len(genealogy_analysis.relations),
                "critical_influences_count": len(critical_influences)
            },
            "influence_metrics": genealogy_analysis.influence_metrics,
            "critical_influences": critical_influences[:5],  # Top 5
            "ancestry_paths": genealogy_analysis.ancestry_paths,
            "centrality_metrics": genealogy_analysis.centrality_metrics
        }
    
    def _evaluate_with_model(self, model_name: str, features: Dict[str, Any], core_result: TextAnalysisResult) -> Dict[str, Any]:
        """Evalúa con modelos específicos del ensemble"""
        
        if model_name == "sentiment_model":
            # Evaluación adicional de sentimiento
            return {
                "model_sentiment": core_result.sentiment_label,
                "model_confidence": 0.75,
                "alternative_score": core_result.sentiment_score * 0.9  # Variación
            }
        
        elif model_name == "topic_model":
            # Evaluación adicional de temas
            return {
                "model_topics": core_result.key_topics,
                "topic_coherence": len(core_result.key_topics) / 5.0,
                "alternative_topics": core_result.key_topics[:2]  # Subconjunto
            }
        
        elif model_name == "readability_model":
            # Evaluación adicional de legibilidad
            return {
                "model_readability": core_result.readability_score,
                "complexity_assessment": "high" if core_result.readability_score < 0.5 else "low",
                "alternative_score": core_result.readability_score * 1.1  # Ligera variación
            }
        
        else:
            return {"model_evaluation": f"Unknown model {model_name}"}

# Función de ejemplo de uso
def demonstrate_text_analysis():
    """Demuestra el uso del analizador de texto universal"""
    
    # Crear analizador
    analyzer = SimpleTextAnalyzer()
    
    # Registrar en registry universal
    universal_registry.register_analyzer("text_analysis", analyzer)
    
    # Texto de ejemplo
    sample_text = """
    This is an amazing example of text analysis using the universal framework. 
    The system can analyze sentiment, extract topics, and measure readability. 
    It demonstrates how the eight universal meta-principles can be applied 
    to any domain of analysis, providing mathematical abstention, confidence bounds, 
    genealogical tracking, and adaptive hybridization.
    """
    
    print("🔤 DEMOSTRACIÓN: Análisis de Texto Universal")
    print("=" * 60)
    
    # Realizar análisis
    result = analyzer.analyze(sample_text)
    
    print(f"📊 Resultado del Análisis:")
    print(f"   • Confianza General: {result.confidence:.3f}")
    print(f"   • Se Abstuvo: {'Sí' if result.abstained else 'No'}")
    
    if not result.abstained and result.result:
        text_result = result.result
        print(f"   • Sentimiento: {text_result.sentiment_label} (score: {text_result.sentiment_score:.3f})")
        print(f"   • Temas Clave: {text_result.key_topics}")
        print(f"   • Legibilidad: {text_result.readability_score:.3f}")
        print(f"   • Idioma: {text_result.language_detected}")
    
    print(f"\n📈 Métricas de Confianza:")
    for metric, value in result.metadata.confidence_metrics.items():
        print(f"   • {metric}: {value:.3f}")
    
    print(f"\n🔗 Análisis Genealógico:")
    genealogy_data = result.metadata.genealogy_data
    if genealogy_data:
        summary = genealogy_data.get("genealogy_summary", {})
        print(f"   • Nodos Totales: {summary.get('total_nodes', 0)}")
        print(f"   • Relaciones: {summary.get('total_relations', 0)}")
        print(f"   • Influencias Críticas: {summary.get('critical_influences_count', 0)}")
    
    print(f"\n🎯 Resultados Ensemble:")
    for i, ensemble_result in enumerate(result.metadata.ensemble_results[:3]):
        model_name = ensemble_result.get("model", "Unknown")
        print(f"   • Modelo {i+1} ({model_name}): {'✓' if not ensemble_result.get('error') else '✗'}")
    
    if result.abstained:
        print(f"\n⚠️  Razones de Abstención:")
        for reason in result.metadata.abstention_reasons:
            print(f"   • {reason}")
    
    print(f"\n🔍 Límites de Incertidumbre:")
    for bound_name, (lower, upper) in result.metadata.uncertainty_bounds.items():
        print(f"   • {bound_name}: [{lower:.3f}, {upper:.3f}]")
    
    return result

# Implementación de modelos específicos para el ensemble
class SentimentModel(UniversalModel):
    """Modelo de sentimiento para el ensemble"""
    
    def __init__(self):
        super().__init__("advanced_sentiment", ModelType.MACHINE_LEARNING)
    
    def predict(self, input_data: Any) -> Tuple[Any, float]:
        text = str(input_data)
        
        # Análisis más sofisticado de sentimiento
        positive_indicators = ["excellent", "amazing", "wonderful", "fantastic", "great", "good", "love", "perfect"]
        negative_indicators = ["terrible", "awful", "horrible", "hate", "worst", "bad", "disgusting", "poor"]
        
        words = text.lower().split()
        pos_score = sum(2 if word in positive_indicators else 1 for word in words 
                       if any(pos in word for pos in positive_indicators))
        neg_score = sum(2 if word in negative_indicators else 1 for word in words 
                       if any(neg in word for neg in negative_indicators))
        
        total_words = len(words)
        sentiment_intensity = (pos_score - neg_score) / max(total_words, 1)
        
        if sentiment_intensity > 0.05:
            sentiment = "positive"
        elif sentiment_intensity < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        confidence = min(1.0, (pos_score + neg_score) / max(total_words * 0.1, 1))
        
        return {
            "sentiment": sentiment,
            "intensity": sentiment_intensity,
            "pos_indicators": pos_score,
            "neg_indicators": neg_score
        }, confidence
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_type": "advanced_sentiment_analysis",
            "version": "1.0",
            "training_data": "sentiment_lexicon_based",
            "accuracy": 0.78
        }

if __name__ == "__main__":
    # Ejecutar demostración
    result = demonstrate_text_analysis()
    
    # Ejemplo adicional con modelo ensemble específico
    print("\n" + "="*60)
    print("🎭 EJEMPLO ENSEMBLE: Modelo de Sentimiento Específico")
    
    # Añadir modelo específico al ensemble
    sentiment_model = SentimentModel()
    universal_ensemble_evaluator.add_model(sentiment_model)
    
    # Evaluar con ensemble
    test_text = "This is a terrible example with some awful content, but it has amazing potential."
    ensemble_result = universal_ensemble_evaluator.evaluate(test_text)
    
    print(f"📊 Resultado Ensemble:")
    print(f"   • Resultado Final: {ensemble_result.final_result}")
    print(f"   • Confianza General: {ensemble_result.overall_confidence:.3f}")
    print(f"   • Estrategia Usada: {ensemble_result.strategy_used.value}")
    print(f"   • Modelos Exitosos: {ensemble_result.to_dict()['successful_models']}")
    
    print(f"\n📈 Métricas de Consenso:")
    for metric, value in ensemble_result.consensus_metrics.items():
        print(f"   • {metric}: {value:.3f}")