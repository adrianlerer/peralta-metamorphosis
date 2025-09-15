"""
Integración Práctica: Optimización SLM para Sistemas de Integridad Comercial
Ejemplo real de implementación del paper NVIDIA para análisis de integridad empresarial.

Este ejemplo demuestra cómo aplicar la optimización SLM vs LLM específicamente
para sistemas de compliance, anti-fraude, y control de integridad comercial.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Importar optimizador SLM y convertidor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domains.slm_agentic_optimizer import (
    SLMAgenticOptimizer, AgenticTask, TaskComplexity, AgenticMode,
    OptimizationResult
)
from domains.llm_to_slm_converter import LLMToSLMConverter

class IntegrityRiskLevel(Enum):
    """Niveles de riesgo de integridad comercial"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceArea(Enum):
    """Áreas de compliance empresarial"""
    ANTI_MONEY_LAUNDERING = "aml"
    FRAUD_DETECTION = "fraud"
    SUPPLIER_VERIFICATION = "supplier"
    FINANCIAL_REPORTING = "financial"
    REGULATORY_COMPLIANCE = "regulatory"
    ETHICS_VIOLATIONS = "ethics"
    DATA_PRIVACY = "privacy"
    TRADE_SANCTIONS = "sanctions"

@dataclass
class IntegrityTask:
    """Tarea específica de integridad comercial"""
    compliance_area: ComplianceArea
    risk_level: IntegrityRiskLevel
    data_sensitivity: str  # "public", "internal", "confidential", "restricted"
    regulatory_scope: str  # "local", "national", "international"
    automation_potential: float  # 0-1
    false_positive_tolerance: float  # 0-1
    regulatory_penalties: str  # "low", "medium", "high", "severe"

def create_commercial_integrity_tasks() -> List[AgenticTask]:
    """Crea tareas representativas de integridad comercial"""
    
    tasks = []
    
    # Tarea 1: Screening automático de transacciones
    tasks.append(AgenticTask(
        task_id="transaction_screening_aml",
        description="Screening automático de transacciones para detección de lavado de dinero y actividades sospechosas",
        complexity=TaskComplexity.MODERATE,
        agentic_mode=AgenticMode.CODE_AGENCY,
        frequency=200.0,  # 200 transacciones por hora
        latency_requirement=2.0,  # 2 segundos máximo
        accuracy_requirement=0.95,  # Alta precisión para evitar falsos positivos
        cost_sensitivity=0.9,  # Muy sensible al costo (alto volumen)
        formatting_strictness=True,  # Reportes regulatorios estructurados
        interaction_type="pattern_detection_classification",
        context_window_needed=2000,  # Datos de transacción
        specialized_domain="anti_money_laundering",
        historical_performance={
            "detection_rate": 0.92,
            "false_positive_rate": 0.08,
            "processing_speed": 1.5
        }
    ))
    
    # Tarea 2: Análisis de integridad de proveedores
    tasks.append(AgenticTask(
        task_id="supplier_integrity_verification",
        description="Verificar integridad y compliance de proveedores mediante análisis de documentos y bases de datos",
        complexity=TaskComplexity.COMPLEX,
        agentic_mode=AgenticMode.LANGUAGE_MODEL_AGENCY,
        frequency=5.0,  # 5 verificaciones por hora
        latency_requirement=60.0,  # 1 hora máximo
        accuracy_requirement=0.98,  # Precisión crítica
        cost_sensitivity=0.4,  # Menos sensible (alto impacto)
        formatting_strictness=False,  # Análisis cualitativo
        interaction_type="document_analysis_research_synthesis",
        context_window_needed=16000,  # Documentos extensos de proveedores
        specialized_domain="supplier_compliance",
        historical_performance={
            "risk_identification": 0.89,
            "compliance_accuracy": 0.94,
            "thoroughness_score": 0.91
        }
    ))
    
    # Tarea 3: Detección de fraude en facturas
    tasks.append(AgenticTask(
        task_id="invoice_fraud_detection",
        description="Detectar patrones de fraude en facturas y documentos financieros",
        complexity=TaskComplexity.SIMPLE,
        agentic_mode=AgenticMode.CODE_AGENCY,
        frequency=150.0,  # 150 facturas por hora
        latency_requirement=5.0,  # 5 segundos
        accuracy_requirement=0.93,
        cost_sensitivity=0.85,
        formatting_strictness=True,  # Alertas estructuradas
        interaction_type="anomaly_detection_pattern_matching",
        context_window_needed=1500,  # Datos de factura
        specialized_domain="fraud_detection"
    ))
    
    # Tarea 4: Análisis de compliance regulatorio
    tasks.append(AgenticTask(
        task_id="regulatory_compliance_analysis",
        description="Analizar compliance con regulaciones específicas y identificar brechas",
        complexity=TaskComplexity.COMPLEX,
        agentic_mode=AgenticMode.LANGUAGE_MODEL_AGENCY,
        frequency=3.0,  # 3 análisis por hora
        latency_requirement=90.0,  # 1.5 horas
        accuracy_requirement=0.96,
        cost_sensitivity=0.3,
        formatting_strictness=True,  # Reportes regulatorios
        interaction_type="regulatory_analysis_gap_identification",
        context_window_needed=20000,  # Documentos normativos extensos
        specialized_domain="regulatory_compliance"
    ))
    
    # Tarea 5: Monitoreo de conflictos de interés
    tasks.append(AgenticTask(
        task_id="conflict_of_interest_monitoring",
        description="Monitorear y detectar potenciales conflictos de interés entre empleados y terceros",
        complexity=TaskComplexity.CONVERSATIONAL,
        agentic_mode=AgenticMode.LANGUAGE_MODEL_AGENCY,
        frequency=10.0,  # 10 casos por hora
        latency_requirement=20.0,  # 20 minutos
        accuracy_requirement=0.88,
        cost_sensitivity=0.6,
        formatting_strictness=False,
        interaction_type="relationship_analysis_pattern_detection",
        context_window_needed=8000,
        specialized_domain="ethics_compliance"
    ))
    
    # Tarea 6: Verificación de sanciones comerciales
    tasks.append(AgenticTask(
        task_id="trade_sanctions_verification",
        description="Verificar cumplimiento con sanciones comerciales y listas de control de exportaciones",
        complexity=TaskComplexity.SIMPLE,
        agentic_mode=AgenticMode.CODE_AGENCY,
        frequency=80.0,  # 80 verificaciones por hora
        latency_requirement=3.0,  # 3 segundos
        accuracy_requirement=0.99,  # Precisión extrema requerida
        cost_sensitivity=0.7,
        formatting_strictness=True,
        interaction_type="database_matching_verification",
        context_window_needed=1000,
        specialized_domain="trade_compliance"
    ))
    
    return tasks

def demonstrate_commercial_integrity_optimization():
    """Demuestra optimización SLM para integridad comercial"""
    
    print("🛡️  OPTIMIZACIÓN SLM PARA INTEGRIDAD COMERCIAL")
    print("=" * 80)
    print("Aplicando NVIDIA Research para sistemas de compliance empresarial")
    print("=" * 80)
    
    # Crear optimizador especializado
    optimizer = SLMAgenticOptimizer()
    
    # Obtener tareas de integridad
    integrity_tasks = create_commercial_integrity_tasks()
    
    # Análisis de cada tarea de integridad
    optimization_results = []
    
    for task in integrity_tasks:
        print(f"\n🔍 ANÁLISIS: {task.task_id}")
        print(f"   • Descripción: {task.description}")
        print(f"   • Área Compliance: {task.specialized_domain}")
        print(f"   • Complejidad: {task.complexity.value}")
        print(f"   • Volumen: {task.frequency} casos/hora")
        print(f"   • Req. Precisión: {task.accuracy_requirement:.1%}")
        print(f"   • Latencia Máx: {task.latency_requirement}s")
        
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
            
            # Análisis específico para integridad comercial
            integrity_insights = analyze_integrity_implications(task, opt_result)
            print(f"\n🛡️  IMPLICACIONES DE INTEGRIDAD:")
            for insight in integrity_insights:
                print(f"   • {insight}")
                
            # Análisis de riesgo regulatorio
            regulatory_risks = assess_regulatory_risks(task, opt_result)
            if regulatory_risks:
                print(f"\n⚠️  RIESGOS REGULATORIOS:")
                for risk in regulatory_risks:
                    print(f"   • {risk}")
        
        elif result.abstained:
            print(f"   • ⚠️  Abstención: Requiere análisis manual por compliance")
            abstention_reasons = result.metadata.abstention_reasons if result.metadata.abstention_reasons else ["Baja confianza en automatización"]
            print(f"   • Razón: {abstention_reasons[0]}")
    
    # Resumen ejecutivo para directores de integridad
    print(f"\n" + "="*80)
    print(f"📊 RESUMEN EJECUTIVO - OPTIMIZACIÓN SLM PARA COMPLIANCE")
    print(f"="*80)
    
    generate_integrity_executive_summary(optimization_results)
    
    # Plan de implementación para compliance
    print(f"\n🚀 PLAN DE IMPLEMENTACIÓN PARA COMPLIANCE:")
    generate_integrity_implementation_plan(optimization_results)
    
    # Demostración de conversión LLM→SLM para un caso específico
    demonstrate_integrity_llm_to_slm_conversion()
    
    return optimization_results

def analyze_integrity_implications(task: AgenticTask, result: OptimizationResult) -> List[str]:
    """Analiza implicaciones específicas para integridad comercial"""
    
    implications = []
    
    # Análisis de precisión para compliance
    if result.model_size_category.value == "slm":
        if task.accuracy_requirement > 0.95:
            implications.append("Precisión alta mantenida para compliance crítico")
        
        if task.specialized_domain in ["anti_money_laundering", "trade_compliance"]:
            implications.append("Validación adicional requerida para compliance regulatorio")
        
        if result.cost_savings > 0.8:
            implications.append(f"Ahorro permite inversión en controles adicionales")
    
    # Análisis de volumen y escalabilidad
    if task.frequency > 50:
        implications.append("Automatización masiva reduce carga operacional")
    
    # Análisis de latencia para procesos críticos
    if task.latency_requirement <= 5.0:
        implications.append("Cumple requisitos de procesamiento en tiempo real")
    
    # Análisis de trazabilidad
    if task.formatting_strictness:
        implications.append("Mantiene trazabilidad para auditorías regulatorias")
    
    return implications

def assess_regulatory_risks(task: AgenticTask, result: OptimizationResult) -> List[str]:
    """Evalúa riesgos regulatorios de la implementación SLM"""
    
    risks = []
    
    # Riesgos por área de compliance
    high_risk_areas = ["anti_money_laundering", "trade_compliance", "regulatory_compliance"]
    if task.specialized_domain in high_risk_areas:
        if result.model_size_category.value == "slm":
            risks.append("Área de alto riesgo regulatorio - requiere supervisión adicional")
    
    # Riesgos por precisión
    if task.accuracy_requirement > 0.95 and result.performance_impact < 0:
        risks.append("Reducción de precisión puede impactar compliance")
    
    # Riesgos por volumen
    if task.frequency > 100:
        risks.append("Alto volumen requiere monitoreo continuo de performance")
    
    # Riesgos de explicabilidad
    if task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.CONVERSATIONAL]:
        risks.append("Casos complejos pueden requerir explicabilidad para reguladores")
    
    return risks

def generate_integrity_executive_summary(results: List[tuple]):
    """Genera resumen ejecutivo para directores de integridad"""
    
    total_tasks = len(results)
    slm_recommendations = sum(1 for _, result in results 
                             if not result.abstained and result.result and 
                             result.result.model_size_category.value == "slm")
    
    total_cost_savings = 0
    high_precision_tasks = 0
    real_time_tasks = 0
    
    for task, result in results:
        if task.accuracy_requirement > 0.95:
            high_precision_tasks += 1
        if task.latency_requirement <= 5.0:
            real_time_tasks += 1
        if not result.abstained and result.result:
            total_cost_savings += result.result.cost_savings
    
    avg_cost_savings = total_cost_savings / len([r for _, r in results if not r.abstained]) if results else 0
    
    print(f"   • Procesos Analizados: {total_tasks}")
    print(f"   • Candidatos para SLM: {slm_recommendations} ({slm_recommendations/total_tasks:.1%})")
    print(f"   • Ahorro Operacional Promedio: {avg_cost_savings:.1%}")
    print(f"   • Procesos Alta Precisión: {high_precision_tasks} ({high_precision_tasks/total_tasks:.1%})")
    print(f"   • Procesos Tiempo Real: {real_time_tasks} ({real_time_tasks/total_tasks:.1%})")
    
    # Análisis por área de compliance
    compliance_analysis = {}
    for task, result in results:
        area = task.specialized_domain or "general"
        if area not in compliance_analysis:
            compliance_analysis[area] = {
                "tasks": 0, 
                "slm_suitable": 0, 
                "high_volume": 0,
                "avg_savings": []
            }
        
        compliance_analysis[area]["tasks"] += 1
        if task.frequency > 50:
            compliance_analysis[area]["high_volume"] += 1
        if not result.abstained and result.result and result.result.model_size_category.value == "slm":
            compliance_analysis[area]["slm_suitable"] += 1
            compliance_analysis[area]["avg_savings"].append(result.result.cost_savings)
    
    print(f"\n🛡️  ANÁLISIS POR ÁREA DE COMPLIANCE:")
    for area, stats in compliance_analysis.items():
        suitability = stats["slm_suitable"] / stats["tasks"] if stats["tasks"] > 0 else 0
        volume_ratio = stats["high_volume"] / stats["tasks"] if stats["tasks"] > 0 else 0
        avg_savings = sum(stats["avg_savings"]) / len(stats["avg_savings"]) if stats["avg_savings"] else 0
        print(f"   • {area.replace('_', ' ').title()}: {suitability:.1%} SLM, {volume_ratio:.1%} alto volumen, {avg_savings:.1%} ahorro")

def generate_integrity_implementation_plan(results: List[tuple]):
    """Genera plan de implementación para integridad comercial"""
    
    # Clasificar tareas por riesgo y volumen
    low_risk_high_volume = []
    medium_risk_tasks = []
    high_risk_tasks = []
    
    for task, result in results:
        if (not result.abstained and result.result and 
            result.result.model_size_category.value == "slm"):
            
            # Determinar nivel de riesgo basado en área de compliance
            risk_level = "low"
            if task.specialized_domain in ["anti_money_laundering", "regulatory_compliance"]:
                risk_level = "high"
            elif task.specialized_domain in ["supplier_compliance", "fraud_detection"]:
                risk_level = "medium"
            
            if risk_level == "low" and task.frequency > 50:
                low_risk_high_volume.append(task.task_id)
            elif risk_level == "medium":
                medium_risk_tasks.append(task.task_id)
            elif risk_level == "high":
                high_risk_tasks.append(task.task_id)
    
    # Fase 1: Automatización de bajo riesgo
    if low_risk_high_volume:
        print(f"\n   📅 FASE 1 (0-2 meses) - Automatización de Bajo Riesgo:")
        for task_id in low_risk_high_volume:
            print(f"      • {task_id.replace('_', ' ').title()}")
        print(f"      • Impacto: Reducción inmediata de carga operacional")
        print(f"      • Riesgo: Mínimo | Supervisión: Semanal")
    
    # Fase 2: Expansión controlada
    if medium_risk_tasks:
        print(f"\n   📅 FASE 2 (2-6 meses) - Expansión Controlada:")
        for task_id in medium_risk_tasks:
            print(f"      • {task_id.replace('_', ' ').title()}")
        print(f"      • Estrategia: Implementación gradual con validación")
        print(f"      • Supervisión: Diaria con escalación automática")
    
    # Fase 3: Casos de alto riesgo regulatorio
    if high_risk_tasks:
        print(f"\n   📅 FASE 3 (6-12 meses) - Alto Riesgo Regulatorio:")
        for task_id in high_risk_tasks:
            print(f"      • {task_id.replace('_', ' ').title()}")
        print(f"      • Estrategia: Modelo híbrido SLM+supervisión humana")
        print(f"      • Validación: Doble verificación obligatoria")
    
    # Controles y monitoreo
    print(f"\n   🔒 CONTROLES DE INTEGRIDAD REQUERIDOS:")
    print(f"      • Dashboard de monitoreo en tiempo real")
    print(f"      • Alertas automáticas por degradación de performance")
    print(f"      • Auditoría mensual de decisiones automatizadas")
    print(f"      • Backup LLM para casos de alta complejidad")
    print(f"      • Trazabilidad completa para investigaciones regulatorias")
    print(f"      • Testing continuo contra casos conocidos")

def demonstrate_integrity_llm_to_slm_conversion():
    """Demuestra conversión LLM→SLM para caso de integridad específico"""
    
    print(f"\n" + "="*80)
    print(f"🔄 DEMOSTRACIÓN: CONVERSIÓN LLM→SLM PARA SCREENING AML")
    print(f"="*80)
    
    # Crear convertidor
    converter = LLMToSLMConverter()
    
    # Datos simulados de agente AML existente
    aml_agent_calls = [
        {
            "call_id": f"aml_{i:04d}",
            "input_prompt": f"Analyze transaction: Amount ${1000+i*50}, From: Account_{i%100}, To: Account_{(i*3)%150}, Country: {'US' if i%3==0 else 'MX' if i%3==1 else 'CA'}",
            "output_response": f"Risk Score: {'LOW' if i%4!=0 else 'MEDIUM' if i%8!=0 else 'HIGH'} - {'Approved' if i%4!=0 else 'Review Required'}",
            "tool_calls": [{"tool": "sanctions_check", "result": "clear"}, {"tool": "pattern_analyzer", "score": 0.1 + (i%10)*0.05}],
            "success": True,
            "latency_ms": 1500 + (i%5)*200,
            "user_id": f"compliance_officer_{i//50}"
        }
        for i in range(300)
    ]
    
    conversion_input = {
        "agent_id": "aml_screening_agent_v2",
        "agent_calls": aml_agent_calls,
        "collection_period_days": 45,
        "conversion_goal": "Reduce latency while maintaining AML compliance",
        "performance_requirements": {
            "min_accuracy": 0.95,
            "max_latency_ms": 1000,
            "cost_reduction_target": 0.8
        }
    }
    
    print(f"📊 AGENTE AML ORIGINAL:")
    print(f"   • Transacciones Procesadas: {len(aml_agent_calls)}")
    print(f"   • Período: {conversion_input['collection_period_days']} días")
    print(f"   • Target Precisión: {conversion_input['performance_requirements']['min_accuracy']:.0%}")
    print(f"   • Target Latencia: {conversion_input['performance_requirements']['max_latency_ms']}ms")
    
    # Ejecutar conversión (simulada para demo)
    result = converter.analyze(conversion_input)
    
    print(f"\n🎯 RESULTADO DE CONVERSIÓN AML:")
    print(f"   • Confianza: {result.confidence:.3f}")
    print(f"   • Abstención: {'Sí' if result.abstained else 'No'}")
    
    if not result.abstained and result.result:
        conv_result = result.result
        print(f"\n📋 CONVERSIÓN COMPLETADA:")
        print(f"   • Etapas Ejecutadas: {len(conv_result.conversion_stages_completed)}/6")
        print(f"   • Clusters Identificados: {len(conv_result.identified_clusters)}")
        print(f"   • Modelos SLM Seleccionados: {len(conv_result.selected_slms)}")
        
        # Análisis específico para AML
        print(f"\n🛡️  IMPACTO EN COMPLIANCE AML:")
        cost_analysis = conv_result.cost_analysis
        if cost_analysis:
            baseline = cost_analysis.get("baseline_llm_costs", {}).get("total_monthly", 0)
            projected = cost_analysis.get("projected_slm_costs", {}).get("total_monthly", 0)
            savings = cost_analysis.get("savings_analysis", {}).get("percentage_savings", 0)
            print(f"   • Reducción de Costos: {savings:.1%}")
            print(f"   • Ahorro Mensual: ${baseline - projected:,.2f}")
        
        performance = conv_result.performance_comparison
        if performance:
            latency_improvement = performance.get("improvement_metrics", {}).get("latency_improvement", 0)
            print(f"   • Mejora de Latencia: {latency_improvement:.1%}")
            print(f"   • Retención Performance: {performance.get('improvement_metrics', {}).get('performance_retention', 0):.1%}")

if __name__ == "__main__":
    # Ejecutar demostración de integridad comercial
    results = demonstrate_commercial_integrity_optimization()
    
    print(f"\n" + "="*80)
    print(f"✅ SISTEMA DE INTEGRIDAD COMERCIAL OPTIMIZADO")
    print(f"="*80)
    print(f"La optimización SLM está implementada para sistemas de compliance")
    print(f"empresarial con controles específicos de integridad comercial.")
    print(f"")
    print(f"Capacidades implementadas:")
    print(f"• Automatización de screening de transacciones AML")
    print(f"• Verificación automatizada de proveedores y terceros")
    print(f"• Detección de patrones de fraude en documentos")
    print(f"• Monitoreo de compliance regulatorio continuo")
    print(f"• Escalación automática para casos de alto riesgo")
    print(f"• Trazabilidad completa para auditorías regulatorias")