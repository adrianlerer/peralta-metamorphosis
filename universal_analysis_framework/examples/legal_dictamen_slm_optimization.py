"""
Integración Práctica: Optimización SLM para Análisis de Dictámenes Legales
Ejemplo real de implementación del paper NVIDIA para sistemas de dictámenes jurídicos.

Este ejemplo demuestra cómo aplicar la optimización SLM vs LLM específicamente
para el análisis automatizado de documentos legales y generación de dictámenes.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
import os

# Importar optimizador SLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domains.slm_agentic_optimizer import (
    SLMAgenticOptimizer, AgenticTask, TaskComplexity, AgenticMode,
    OptimizationResult
)

@dataclass
class DictamenTask:
    """Tarea específica de análisis legal para dictámenes"""
    document_type: str  # "contract", "regulation", "case_law", "statute"
    legal_domain: str   # "civil", "commercial", "administrative", "constitutional"
    complexity_level: str  # "routine", "standard", "complex", "exceptional"
    urgency: str       # "standard", "urgent", "emergency"
    confidentiality: str  # "public", "confidential", "restricted"
    expected_length: int  # Páginas esperadas del dictamen
    research_depth: str   # "basic", "comprehensive", "exhaustive"

def create_legal_analysis_tasks() -> List[AgenticTask]:
    """Crea tareas representativas de análisis legal para dictámenes"""
    
    tasks = []
    
    # Tarea 1: Revisión de contratos comerciales rutinaria
    tasks.append(AgenticTask(
        task_id="contract_compliance_review",
        description="Revisar contratos comerciales para cumplimiento normativo y identificar cláusulas problemáticas",
        complexity=TaskComplexity.SIMPLE,
        agentic_mode=AgenticMode.CODE_AGENCY,
        frequency=25.0,  # 25 contratos por hora en peak
        latency_requirement=5.0,  # 5 minutos máximo
        accuracy_requirement=0.95,  # Alta precisión requerida
        cost_sensitivity=0.9,  # Muy sensible al costo (alto volumen)
        formatting_strictness=True,  # Formato legal específico requerido
        interaction_type="document_analysis_tool_calling",
        context_window_needed=8000,  # Contratos típicos
        specialized_domain="commercial_law",
        historical_performance={
            "accuracy_rate": 0.93,
            "avg_processing_time": 4.2,
            "client_satisfaction": 0.88
        }
    ))
    
    # Tarea 2: Análisis constitucional complejo
    tasks.append(AgenticTask(
        task_id="constitutional_law_analysis",
        description="Analizar constitucionalidad de normas y preparar dictámenes de fondo",
        complexity=TaskComplexity.COMPLEX,
        agentic_mode=AgenticMode.LANGUAGE_MODEL_AGENCY,
        frequency=1.5,  # 1-2 por hora (trabajo especializado)
        latency_requirement=120.0,  # 2 horas máximo
        accuracy_requirement=0.98,  # Precisión crítica
        cost_sensitivity=0.3,  # Menos sensible (alto valor)
        formatting_strictness=False,  # Más flexibilidad en formato
        interaction_type="deep_reasoning_legal_research",
        context_window_needed=32000,  # Documentos extensos
        specialized_domain="constitutional_law",
        historical_performance={
            "accuracy_rate": 0.95,
            "avg_processing_time": 85.5,
            "judicial_acceptance": 0.92
        }
    ))
    
    # Tarea 3: Due diligence automatizado
    tasks.append(AgenticTask(
        task_id="due_diligence_automation",
        description="Automatizar procesos de due diligence corporativo y identificación de riesgos",
        complexity=TaskComplexity.MODERATE,
        agentic_mode=AgenticMode.CODE_AGENCY,
        frequency=8.0,  # 8 por hora
        latency_requirement=15.0,  # 15 minutos
        accuracy_requirement=0.90,
        cost_sensitivity=0.7,
        formatting_strictness=True,  # Reportes estructurados
        interaction_type="data_extraction_classification",
        context_window_needed=12000,
        specialized_domain="corporate_law",
        historical_performance={
            "risk_detection_rate": 0.87,
            "false_positive_rate": 0.12,
            "process_efficiency": 0.91
        }
    ))
    
    # Tarea 4: Investigación jurisprudencial
    tasks.append(AgenticTask(
        task_id="jurisprudence_research",
        description="Buscar y analizar precedentes jurisprudenciales relevantes para casos específicos",
        complexity=TaskComplexity.CONVERSATIONAL,
        agentic_mode=AgenticMode.LANGUAGE_MODEL_AGENCY,
        frequency=5.0,  # 5 por hora
        latency_requirement=30.0,  # 30 minutos
        accuracy_requirement=0.88,
        cost_sensitivity=0.6,
        formatting_strictness=False,
        interaction_type="research_reasoning_synthesis",
        context_window_needed=16000,
        specialized_domain="case_law_research"
    ))
    
    # Tarea 5: Redacción de dictámenes simples
    tasks.append(AgenticTask(
        task_id="simple_opinion_drafting",
        description="Redactar dictámenes legales para consultas rutinarias y procedimientos estándar",
        complexity=TaskComplexity.SIMPLE,
        agentic_mode=AgenticMode.CODE_AGENCY,
        frequency=15.0,  # 15 por hora
        latency_requirement=10.0,  # 10 minutos
        accuracy_requirement=0.92,
        cost_sensitivity=0.8,
        formatting_strictness=True,  # Formato oficial requerido
        interaction_type="structured_document_generation",
        context_window_needed=4000,
        specialized_domain="general_legal_opinions"
    ))
    
    return tasks

def demonstrate_legal_slm_optimization():
    """Demuestra optimización SLM para análisis legal"""
    
    print("⚖️  OPTIMIZACIÓN SLM PARA DICTÁMENES LEGALES")
    print("=" * 80)
    print("Aplicando NVIDIA Research para sistemas jurídicos automatizados")
    print("=" * 80)
    
    # Crear optimizador especializado
    optimizer = SLMAgenticOptimizer()
    
    # Obtener tareas legales
    legal_tasks = create_legal_analysis_tasks()
    
    # Análisis de cada tarea legal
    optimization_results = []
    
    for task in legal_tasks:
        print(f"\n📋 ANÁLISIS: {task.task_id}")
        print(f"   • Descripción: {task.description}")
        print(f"   • Dominio Legal: {task.specialized_domain}")
        print(f"   • Complejidad: {task.complexity.value}")
        print(f"   • Frecuencia: {task.frequency} casos/hora")
        print(f"   • Req. Precisión: {task.accuracy_requirement:.1%}")
        print(f"   • Sensibilidad Costo: {task.cost_sensitivity:.1%}")
        
        # Realizar optimización
        result = optimizer.analyze(task)
        optimization_results.append((task, result))
        
        print(f"\n🎯 RECOMENDACIÓN SLM:")
        print(f"   • Confianza: {result.confidence:.3f}")
        
        if not result.abstained and result.result:
            opt_result = result.result
            print(f"   • Modelo Recomendado: {opt_result.recommended_model}")
            print(f"   • Categoría: {opt_result.model_size_category.value.upper()}")
            print(f"   • Ahorro Costo: {opt_result.cost_savings:.1%}")
            print(f"   • Impacto Performance: {opt_result.performance_impact:+.2f}")
            
            # Análisis específico para contexto legal
            legal_insights = analyze_legal_implications(task, opt_result)
            print(f"\n⚖️  IMPLICACIONES LEGALES:")
            for insight in legal_insights:
                print(f"   • {insight}")
        
        elif result.abstained:
            print(f"   • ⚠️  Abstención: Requiere análisis manual especializado")
            print(f"   • Razón: {result.metadata.abstention_reasons[0] if result.metadata.abstention_reasons else 'Baja confianza'}")
    
    # Resumen ejecutivo para dirección legal
    print(f"\n" + "="*80)
    print(f"📊 RESUMEN EJECUTIVO - OPTIMIZACIÓN SLM PARA ÁREA LEGAL")
    print(f"="*80)
    
    generate_legal_executive_summary(optimization_results)
    
    # Recomendaciones de implementación
    print(f"\n🚀 PLAN DE IMPLEMENTACIÓN PARA ÁREA LEGAL:")
    generate_legal_implementation_plan(optimization_results)
    
    return optimization_results

def analyze_legal_implications(task: AgenticTask, result: OptimizationResult) -> List[str]:
    """Analiza implicaciones específicas para el contexto legal"""
    
    implications = []
    
    # Análisis de precisión legal
    if result.model_size_category.value == "slm":
        if task.accuracy_requirement > 0.95:
            implications.append("Requiere validación adicional para casos de alta precisión")
        
        if task.specialized_domain in ["constitutional_law", "criminal_law"]:
            implications.append("Recomendado para revisión humana en casos críticos")
        
        if result.cost_savings > 0.7:
            implications.append(f"Ahorro significativo permite reinversión en especialización")
    
    # Análisis de compliance
    if task.formatting_strictness:
        implications.append("Formato compatible con estándares judiciales")
    
    # Análisis de escalabilidad
    if task.frequency > 10:
        implications.append("Candidato ideal para automatización masiva")
    
    # Análisis de riesgo
    if task.complexity == TaskComplexity.COMPLEX and result.model_size_category.value == "slm":
        implications.append("Considerar modelo híbrido SLM+LLM para casos complejos")
    
    return implications

def generate_legal_executive_summary(results: List[tuple]):
    """Genera resumen ejecutivo para directores legales"""
    
    total_tasks = len(results)
    slm_recommendations = sum(1 for _, result in results 
                             if not result.abstained and result.result and 
                             result.result.model_size_category.value == "slm")
    
    total_cost_savings = 0
    high_confidence_decisions = 0
    
    for task, result in results:
        if not result.abstained and result.result:
            total_cost_savings += result.result.cost_savings
            if result.confidence > 0.8:
                high_confidence_decisions += 1
    
    avg_cost_savings = total_cost_savings / len([r for _, r in results if not r.abstained]) if results else 0
    
    print(f"   • Tareas Analizadas: {total_tasks}")
    print(f"   • Recomendaciones SLM: {slm_recommendations} ({slm_recommendations/total_tasks:.1%})")
    print(f"   • Ahorro Promedio Proyectado: {avg_cost_savings:.1%}")
    print(f"   • Decisiones Alta Confianza: {high_confidence_decisions} ({high_confidence_decisions/total_tasks:.1%})")
    
    # Análisis por dominio legal
    domain_analysis = {}
    for task, result in results:
        domain = task.specialized_domain or "general"
        if domain not in domain_analysis:
            domain_analysis[domain] = {"tasks": 0, "slm_suitable": 0, "avg_savings": []}
        
        domain_analysis[domain]["tasks"] += 1
        if not result.abstained and result.result and result.result.model_size_category.value == "slm":
            domain_analysis[domain]["slm_suitable"] += 1
            domain_analysis[domain]["avg_savings"].append(result.result.cost_savings)
    
    print(f"\n📊 ANÁLISIS POR ÁREA LEGAL:")
    for domain, stats in domain_analysis.items():
        suitability = stats["slm_suitable"] / stats["tasks"] if stats["tasks"] > 0 else 0
        avg_savings = sum(stats["avg_savings"]) / len(stats["avg_savings"]) if stats["avg_savings"] else 0
        print(f"   • {domain.replace('_', ' ').title()}: {suitability:.1%} aptitud SLM, {avg_savings:.1%} ahorro")

def generate_legal_implementation_plan(results: List[tuple]):
    """Genera plan de implementación específico para contexto legal"""
    
    implementation_phases = []
    
    # Fase 1: Tareas de bajo riesgo y alto volumen
    phase1_tasks = []
    for task, result in results:
        if (not result.abstained and result.result and 
            result.result.model_size_category.value == "slm" and
            task.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE] and
            task.frequency > 10):
            phase1_tasks.append(task.task_id)
    
    if phase1_tasks:
        print(f"\n   📅 FASE 1 (0-3 meses) - Automatización de Rutinas:")
        for task_id in phase1_tasks:
            print(f"      • {task_id.replace('_', ' ').title()}")
        print(f"      • Riesgo: Bajo | Impacto: Alto | ROI Esperado: 6-12 meses")
    
    # Fase 2: Tareas de complejidad media con alta confianza
    phase2_tasks = []
    for task, result in results:
        if (not result.abstained and result.result and
            result.confidence > 0.75 and
            task.complexity == TaskComplexity.MODERATE and
            task.task_id not in phase1_tasks):
            phase2_tasks.append(task.task_id)
    
    if phase2_tasks:
        print(f"\n   📅 FASE 2 (3-8 meses) - Expansión a Casos Moderados:")
        for task_id in phase2_tasks:
            print(f"      • {task_id.replace('_', ' ').title()}")
        print(f"      • Riesgo: Medio | Requiere: Validación humana especializada")
    
    # Fase 3: Casos complejos con supervisión
    phase3_tasks = []
    for task, result in results:
        if (task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.CONVERSATIONAL] and
            not result.abstained):
            phase3_tasks.append(task.task_id)
    
    if phase3_tasks:
        print(f"\n   📅 FASE 3 (6-12 meses) - Asistencia en Casos Complejos:")
        for task_id in phase3_tasks:
            print(f"      • {task_id.replace('_', ' ').title()}")
        print(f"      • Estrategia: Modelo híbrido SLM+humano especialista")
    
    # Consideraciones específicas legales
    print(f"\n   ⚖️  CONSIDERACIONES LEGALES CRÍTICAS:")
    print(f"      • Auditoría completa de decisiones automatizadas")
    print(f"      • Trazabilidad de razonamiento para casos judiciales")
    print(f"      • Protocolos de escalación para casos atípicos")
    print(f"      • Validación por abogados senior en implementación")
    print(f"      • Compliance con normativas de protección de datos")
    print(f"      • Backup LLM para casos de alta complejidad")

if __name__ == "__main__":
    # Ejecutar demostración legal
    results = demonstrate_legal_slm_optimization()
    
    print(f"\n" + "="*80)
    print(f"✅ ANÁLISIS COMPLETADO - SISTEMA LISTO PARA ÁREA LEGAL")
    print(f"="*80)
    print(f"El sistema de optimización SLM está configurado específicamente")
    print(f"para las necesidades de análisis legal y generación de dictámenes.")
    print(f"")
    print(f"Beneficios implementados:")
    print(f"• Reducción significativa de costos operativos")
    print(f"• Automatización de tareas rutinarias legales")
    print(f"• Mantenimiento de estándares de precisión jurídica")
    print(f"• Escalabilidad para volúmenes altos de documentos")
    print(f"• Integración con workflows legales existentes")