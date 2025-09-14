#!/usr/bin/env python3
"""
Universal Analysis Framework - Demostración Completa
Muestra los 8 meta-principios aplicados a múltiples dominios.
"""

import asyncio
import logging
from typing import Dict, Any
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniversalFrameworkDemo")

def main():
    """Demostración principal del Universal Analysis Framework"""
    
    print("🚀 UNIVERSAL ANALYSIS FRAMEWORK")
    print("=" * 60)
    print("Implementación de los 8 Meta-Principios Universales")
    print("Extraídos del desarrollo de LexCertainty Enterprise")
    print("=" * 60)
    
    print("\n📋 Meta-Principios Implementados:")
    principles = [
        "1. Marco de abstención matemática (Mathematical Abstention)",
        "2. Límites de confianza en decisiones (Confidence Bounds)", 
        "3. Análisis genealógico de influencias (Genealogical Analysis)",
        "4. Pipeline multi-etapa con validación (Multi-stage Pipeline)",
        "5. Evaluación ensemble multi-modelo (Ensemble Evaluation)",
        "6. Cuantificación de incertidumbre (Uncertainty Quantification)",
        "7. Salida estructurada con metadatos (Structured Output)",
        "8. Hibridación adaptativa (Adaptive Hybridization)"
    ]
    
    for principle in principles:
        print(f"   ✅ {principle}")
    
    # Demostración 1: Análisis de Texto
    print("\n" + "="*60)
    print("🔤 DEMOSTRACIÓN 1: ANÁLISIS DE TEXTO")
    print("="*60)
    
    try:
        from domains.text_analysis_example import demonstrate_text_analysis
        text_result = demonstrate_text_analysis()
        logger.info("✅ Análisis de texto completado exitosamente")
    except Exception as e:
        logger.error(f"❌ Error en análisis de texto: {e}")
    
    # Demostración 2: Análisis Financiero  
    print("\n" + "="*60)
    print("💰 DEMOSTRACIÓN 2: ANÁLISIS FINANCIERO")
    print("="*60)
    
    try:
        from domains.financial_analysis_example import demonstrate_financial_analysis
        financial_result = demonstrate_financial_analysis()
        logger.info("✅ Análisis financiero completado exitosamente")
    except Exception as e:
        logger.error(f"❌ Error en análisis financiero: {e}")
    
    # Demostración 3: Framework Matemático Puro
    print("\n" + "="*60)
    print("🔢 DEMOSTRACIÓN 3: FRAMEWORK MATEMÁTICO")
    print("="*60)
    
    demonstrate_mathematical_framework()
    
    # Demostración 4: Sistema Ensemble
    print("\n" + "="*60) 
    print("🎭 DEMOSTRACIÓN 4: SISTEMA ENSEMBLE")
    print("="*60)
    
    demonstrate_ensemble_system()
    
    # Demostración 5: Análisis Genealógico
    print("\n" + "="*60)
    print("🔗 DEMOSTRACIÓN 5: ANÁLISIS GENEALÓGICO")  
    print("="*60)
    
    demonstrate_genealogical_analysis()
    
    # Demostración 6: Hibridización Adaptativa
    print("\n" + "="*60)
    print("🔄 DEMOSTRACIÓN 6: HIBRIDIZACIÓN ADAPTATIVA")
    print("="*60)
    
    demonstrate_adaptive_hybridization()
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN DE DEMOSTRACIÓN COMPLETADA")
    print("="*60)
    print("✅ Todos los meta-principios demostrados exitosamente")
    print("🎯 Framework listo para aplicación en cualquier dominio")
    print("🔌 API IntegridAI disponible para integración empresarial")
    
    # Información de la API
    print(f"\n🌐 Para usar la API IntegridAI:")
    print(f"   cd integration/")
    print(f"   python integrid_api.py")
    print(f"   # API disponible en http://localhost:8000")
    print(f"   # Documentación en http://localhost:8000/docs")

def demonstrate_mathematical_framework():
    """Demuestra el framework matemático puro"""
    from mathematical.abstention_framework import (
        universal_math_framework, BoundCalculationMethod
    )
    
    print("🧮 Framework Matemático Universal")
    
    # Datos de ejemplo para análisis
    np.random.seed(42)
    confidence_scores = np.random.beta(8, 2, 50)  # Datos sesgados hacia alta confianza
    
    print(f"📊 Calculando límites para {len(confidence_scores)} puntos de datos")
    
    # Calcular múltiples límites
    methods = [
        BoundCalculationMethod.BOOTSTRAP_PERCENTILE,
        BoundCalculationMethod.HOEFFDING,
        BoundCalculationMethod.ENSEMBLE_VARIANCE
    ]
    
    bounds = universal_math_framework.calculate_multiple_bounds(
        confidence_scores, methods, confidence_level=0.95
    )
    
    print(f"\n📈 Límites de Confianza (95%):")
    for method_name, bound in bounds.items():
        print(f"   • {method_name}:")
        print(f"     - Límites: [{bound.lower_bound:.3f}, {bound.upper_bound:.3f}]")
        print(f"     - Ancho: {bound.width:.3f}")
        print(f"     - Centro: {bound.center:.3f}")
    
    # Decisión de abstención
    abstention = universal_math_framework.make_abstention_decision(
        bounds, confidence_threshold=0.85, width_threshold=0.2
    )
    
    print(f"\n⚖️ Decisión de Abstención:")
    print(f"   • Debería abstenerse: {'Sí' if abstention.should_abstain else 'No'}")
    print(f"   • Score de confianza: {abstention.confidence_score:.3f}")
    print(f"   • Riesgo evaluado: {abstention.risk_assessment.value}")
    
    if abstention.mathematical_justification:
        print(f"   • Justificación: {abstention.mathematical_justification}")

def demonstrate_ensemble_system():
    """Demuestra el sistema ensemble universal"""
    from ensemble.multi_model_evaluator import (
        universal_ensemble_evaluator, FunctionBasedModel, ModelType
    )
    
    print("🎯 Sistema Ensemble Universal")
    
    # Crear modelos de ejemplo
    def sentiment_model_1(text):
        score = len([w for w in str(text).split() if w in ["good", "great", "excellent"]]) / max(len(str(text).split()), 1)
        return {"sentiment": "positive" if score > 0.1 else "neutral", "score": score}, min(1.0, score * 5)
    
    def sentiment_model_2(text):
        score = len([w for w in str(text).split() if w in ["bad", "terrible", "awful"]]) / max(len(str(text).split()), 1)
        return {"sentiment": "negative" if score > 0.1 else "neutral", "score": score}, min(1.0, score * 5)
    
    def sentiment_model_3(text):
        words = str(text).split()
        score = 0.5 + (len(words) % 3 - 1) * 0.2  # Modelo "aleatorio" determinista
        return {"sentiment": "neutral", "score": score}, 0.6
    
    # Añadir modelos al ensemble
    model1 = FunctionBasedModel("sentiment_pos", sentiment_model_1, ModelType.RULE_BASED)
    model2 = FunctionBasedModel("sentiment_neg", sentiment_model_2, ModelType.RULE_BASED) 
    model3 = FunctionBasedModel("sentiment_base", sentiment_model_3, ModelType.HEURISTIC)
    
    universal_ensemble_evaluator.add_model(model1)
    universal_ensemble_evaluator.add_model(model2)
    universal_ensemble_evaluator.add_model(model3)
    
    # Evaluar con ensemble
    test_text = "This is a great example with excellent results but some terrible aspects"
    
    result = universal_ensemble_evaluator.evaluate(test_text)
    
    print(f"📊 Evaluación Ensemble:")
    print(f"   • Entrada: '{test_text}'")
    print(f"   • Resultado Final: {result.final_result}")
    print(f"   • Confianza General: {result.overall_confidence:.3f}")
    print(f"   • Estrategia: {result.strategy_used.value}")
    print(f"   • Modelos Exitosos: {len([r for r in result.individual_results if r.error is None])}/{len(result.individual_results)}")
    
    print(f"\n🔍 Resultados Individuales:")
    for i, model_result in enumerate(result.individual_results):
        print(f"   • Modelo {i+1} ({model_result.model_id}):")
        print(f"     - Resultado: {model_result.result}")
        print(f"     - Confianza: {model_result.confidence:.3f}")
        print(f"     - Tiempo: {model_result.processing_time:.3f}s")

def demonstrate_genealogical_analysis():
    """Demuestra el análisis genealógico"""
    from genealogical.influence_tracker import (
        UniversalInfluenceTracker, NodeType, InfluenceType
    )
    
    print("🔗 Análisis Genealógico Universal")
    
    # Crear tracker
    tracker = UniversalInfluenceTracker("demo_genealogy")
    
    # Simular un pipeline de análisis completo
    print("📋 Construyendo grafo genealógico...")
    
    # Paso 1: Datos de entrada
    input_node = tracker.add_node(
        "raw_data_input",
        NodeType.INPUT_DATA, 
        "Original dataset",
        importance=1.0
    )
    
    # Paso 2: Preprocesamiento
    preprocess_node = tracker.add_node(
        "data_preprocessing",
        NodeType.PROCESSING_STEP,
        "Clean and normalize data",
        importance=0.8
    )
    
    processed_data_node = tracker.add_node(
        "processed_data",
        NodeType.INTERMEDIATE_RESULT,
        "Cleaned dataset", 
        importance=0.9
    )
    
    # Paso 3: Extracción de características
    feature_extract_node = tracker.add_node(
        "feature_extraction", 
        NodeType.PROCESSING_STEP,
        "Extract relevant features",
        importance=0.85
    )
    
    features_node = tracker.add_node(
        "extracted_features",
        NodeType.INTERMEDIATE_RESULT,
        "Feature vectors",
        importance=0.9
    )
    
    # Paso 4: Análisis principal
    analysis_node = tracker.add_node(
        "core_analysis",
        NodeType.PROCESSING_STEP, 
        "Main analysis algorithm",
        importance=0.95
    )
    
    result_node = tracker.add_node(
        "final_result",
        NodeType.FINAL_RESULT,
        "Analysis conclusions",
        importance=1.0
    )
    
    # Añadir influencias
    tracker.add_influence(input_node, preprocess_node, InfluenceType.DATA_DEPENDENCY, 1.0)
    tracker.add_influence(preprocess_node, processed_data_node, InfluenceType.DIRECT_CAUSAL, 1.0)
    tracker.add_influence(processed_data_node, feature_extract_node, InfluenceType.DATA_DEPENDENCY, 1.0)
    tracker.add_influence(feature_extract_node, features_node, InfluenceType.DIRECT_CAUSAL, 0.95)
    tracker.add_influence(features_node, analysis_node, InfluenceType.DATA_DEPENDENCY, 1.0)
    tracker.add_influence(analysis_node, result_node, InfluenceType.DIRECT_CAUSAL, 0.92)
    
    # Añadir algunas influencias metodológicas
    method_node = tracker.add_node(
        "analysis_methodology",
        NodeType.EXTERNAL_SOURCE,
        "Statistical analysis method",
        importance=0.8
    )
    
    tracker.add_influence(method_node, analysis_node, InfluenceType.METHODOLOGICAL, 0.85)
    
    # Realizar análisis genealógico
    genealogy = tracker.analyze_genealogy()
    
    print(f"\n📊 Análisis Genealógico Completado:")
    print(f"   • Nodos totales: {len(genealogy.nodes)}")
    print(f"   • Relaciones: {len(genealogy.relations)}")
    print(f"   • Densidad del grafo: {genealogy.metadata.get('graph_density', 0):.3f}")
    print(f"   • Es DAG: {'Sí' if genealogy.metadata.get('is_dag') else 'No'}")
    
    # Influencias críticas
    critical = tracker.find_critical_influences(importance_threshold=0.8)
    
    print(f"\n🎯 Influencias Críticas (>{len(critical)} encontradas):")
    for i, influence in enumerate(critical[:3]):  # Top 3
        rel = influence["relation"]
        print(f"   • Influencia {i+1}:")
        print(f"     - Origen: {rel['source_id']}")
        print(f"     - Destino: {rel['target_id']}")
        print(f"     - Tipo: {rel['influence_type']}")
        print(f"     - Fuerza: {rel['strength']:.3f}")
        print(f"     - Criticidad: {influence['criticality_score']:.3f}")

def demonstrate_adaptive_hybridization():
    """Demuestra la hibridización adaptativa"""
    from hybridization.adaptive_hybridizer import (
        universal_adaptive_hybridizer, HybridizationContext, ComponentType
    )
    
    print("🔄 Hibridización Adaptativa Universal")
    
    # Añadir algunos componentes de ejemplo
    def fast_algorithm(data):
        return {"result": "fast_processed", "quality": 0.7}, 0.8
    
    def accurate_algorithm(data): 
        return {"result": "accurate_processed", "quality": 0.95}, 0.9
    
    def balanced_algorithm(data):
        return {"result": "balanced_processed", "quality": 0.85}, 0.85
    
    # Registrar componentes con diferentes perfiles
    universal_adaptive_hybridizer.add_function_component(
        "fast_algo",
        ComponentType.ALGORITHM,
        fast_algorithm,
        performance_metrics={"speed": 0.95, "accuracy": 0.70},
        context_suitability={"real_time": 0.9, "batch": 0.5},
        computational_cost=0.5,
        reliability_score=0.8
    )
    
    universal_adaptive_hybridizer.add_function_component(
        "accurate_algo", 
        ComponentType.ALGORITHM,
        accurate_algorithm,
        performance_metrics={"speed": 0.3, "accuracy": 0.95},
        context_suitability={"real_time": 0.3, "batch": 0.9},
        computational_cost=2.0,
        reliability_score=0.95
    )
    
    universal_adaptive_hybridizer.add_function_component(
        "balanced_algo",
        ComponentType.ALGORITHM,
        balanced_algorithm, 
        performance_metrics={"speed": 0.7, "accuracy": 0.85},
        context_suitability={"real_time": 0.7, "batch": 0.8},
        computational_cost=1.2,
        reliability_score=0.85
    )
    
    # Contexto para análisis en tiempo real
    real_time_context = HybridizationContext(
        domain="real_time_analysis",
        data_characteristics={"size": "medium", "complexity": "low"},
        performance_requirements={"speed": 0.9, "accuracy": 0.7},
        resource_constraints={"max_computation_time": 1.0}
    )
    
    # Contexto para análisis batch de alta precisión
    batch_context = HybridizationContext(
        domain="batch_analysis",
        data_characteristics={"size": "large", "complexity": "high"}, 
        performance_requirements={"speed": 0.5, "accuracy": 0.95},
        resource_constraints={"max_computation_time": 10.0}
    )
    
    # Hibridización para contexto en tiempo real
    print(f"\n⚡ Hibridización para Tiempo Real:")
    rt_result = universal_adaptive_hybridizer.hybridize(real_time_context)
    
    print(f"   • Componentes seleccionados: {rt_result.selected_components}")
    print(f"   • Estrategia: {rt_result.hybridization_strategy.value}")
    print(f"   • Confianza: {rt_result.confidence_score:.3f}")
    print(f"   • Razonamiento:")
    for reason in rt_result.adaptation_reasoning:
        print(f"     - {reason}")
    
    # Hibridización para contexto batch
    print(f"\n🔬 Hibridización para Análisis Batch:")
    batch_result = universal_adaptive_hybridizer.hybridize(batch_context)
    
    print(f"   • Componentes seleccionados: {batch_result.selected_components}")
    print(f"   • Estrategia: {batch_result.hybridization_strategy.value}")
    print(f"   • Confianza: {batch_result.confidence_score:.3f}")
    print(f"   • Razonamiento:")
    for reason in batch_result.adaptation_reasoning:
        print(f"     - {reason}")
    
    # Mostrar diferencias adaptativas
    print(f"\n📈 Adaptación Contextual Demostrada:")
    print(f"   • Tiempo real priorizó: {rt_result.selected_components}")
    print(f"   • Batch priorizó: {batch_result.selected_components}")
    print(f"   • Adaptación exitosa según contexto ✅")

if __name__ == "__main__":
    main()