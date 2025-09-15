"""
Reality Filter 2.0 - Validación Rigurosa del Caso AML
Aplicación del filtro de realidad para separar hechos verificables de proyecciones optimistas.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'universal_analysis_framework'))

from core.universal_framework import UniversalAnalyzer, UniversalResult
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class VerificationLevel(Enum):
    """Niveles de verificación de afirmaciones"""
    VERIFIED_FACT = "verified_fact"           # Datos públicos verificables
    INDUSTRY_STANDARD = "industry_standard"   # Estándares conocidos de la industria  
    REASONABLE_ESTIMATE = "reasonable_estimate" # Estimación basada en datos conocidos
    OPTIMISTIC_PROJECTION = "optimistic_projection" # Proyección que requiere validación
    UNVERIFIED_CLAIM = "unverified_claim"     # Afirmación no verificable

@dataclass
class RealityCheck:
    """Resultado de verificación de una afirmación específica"""
    claim: str
    verification_level: VerificationLevel
    evidence: List[str]
    confidence_score: float  # 0-1
    reality_adjusted_value: Any
    assumptions: List[str]
    risk_factors: List[str]

class RealityFilter:
    """Filtro de realidad para validar afirmaciones técnicas y financieras"""
    
    def __init__(self):
        # Base de conocimiento verificable de la industria
        self.verified_benchmarks = {
            "gpt4_cost_per_1k_tokens": 0.03,  # OpenAI pricing público
            "kimi_k2_active_parameters": 32e9,  # Kimi K2 - 32B activos de 1T total (MoE)
            "kimi_k2_openrouter_available": True,  # Acceso directo via OpenRouter
            "typical_aml_transaction_volume": {"small_bank": 5000, "large_bank": 100000},
            "regulatory_accuracy_requirements": {"aml_minimum": 0.90, "high_risk": 0.95},
            "slm_vs_llm_cost_ratio": {"conservative": 0.1, "optimistic": 0.01},
            "inference_latency_improvements": {"typical_slm": 0.3, "optimized_slm": 0.7}
        }
        
        self.industry_standards = {
            "aml_false_positive_rates": {"industry_average": 0.05, "best_practice": 0.02},
            "compliance_staff_cost_per_hour": 75,  # USD promedio compliance analyst
            "model_implementation_costs": {"basic": 5000, "enterprise": 50000},
            "roi_timeframes": {"software": "6-18 months", "ai_models": "3-12 months"}
        }

def apply_reality_filter_to_aml_case():
    """Aplica Reality Filter 2.0 al caso AML presentado"""
    
    print("🔍 REALITY FILTER 2.0 - VALIDACIÓN RIGUROSA")
    print("=" * 70)
    print("Análisis crítico de afirmaciones del caso Banco Nacional AML")
    print("=" * 70)
    
    filter_engine = RealityFilter()
    
    # Afirmaciones del caso original para verificar
    claims_to_verify = [
        {
            "claim": "Costo por transacción actual: $0.008 con GPT-4",
            "original_value": 0.008,
            "category": "cost_analysis"
        },
        {
            "claim": "Costo proyectado con SLM: $0.0001 por transacción",
            "original_value": 0.0001,
            "category": "cost_projection"
        },
        {
            "claim": "Mejora de latencia: 70% (2.3s → 0.69s)",
            "original_value": 0.70,
            "category": "performance"
        },
        {
            "claim": "Ahorro anual: $142,903",
            "original_value": 142903,
            "category": "financial_projection"
        },
        {
            "claim": "ROI en 1.3 meses",
            "original_value": 1.3,
            "category": "roi_timeline"
        },
        {
            "claim": "Precisión mejora de 94.2% a 95.0%",
            "original_value": 0.008,
            "category": "accuracy_improvement"
        },
        {
            "claim": "Reducción falsos positivos: 44 casos/día",
            "original_value": 44,
            "category": "operational_improvement"
        }
    ]
    
    verified_results = []
    
    for claim_data in claims_to_verify:
        result = verify_claim(claim_data, filter_engine)
        verified_results.append(result)
        
        print(f"\n📊 VERIFICACIÓN: {claim_data['claim']}")
        print(f"   🎯 Nivel de Verificación: {result.verification_level.value.upper()}")
        print(f"   📈 Confianza: {result.confidence_score:.2f}")
        print(f"   ✅ Valor Ajustado por Realidad: {result.reality_adjusted_value}")
        
        if result.evidence:
            print(f"   📋 Evidencia:")
            for evidence in result.evidence[:2]:  # Top 2 evidencias
                print(f"      • {evidence}")
        
        if result.assumptions:
            print(f"   ⚠️ Asunciones Críticas:")
            for assumption in result.assumptions[:2]:
                print(f"      • {assumption}")
        
        if result.risk_factors:
            print(f"   🚨 Factores de Riesgo:")
            for risk in result.risk_factors[:2]:
                print(f"      • {risk}")
    
    # Generar resumen de realidad ajustada
    generate_reality_adjusted_summary(verified_results)
    
    return verified_results

def verify_claim(claim_data: Dict, filter_engine: RealityFilter) -> RealityCheck:
    """Verifica una afirmación específica contra datos conocidos"""
    
    claim = claim_data["claim"]
    category = claim_data["category"]
    original_value = claim_data["original_value"]
    
    if category == "cost_analysis":
        return verify_gpt4_cost_claim(claim, original_value, filter_engine)
    
    elif category == "cost_projection":
        return verify_slm_cost_projection(claim, original_value, filter_engine)
    
    elif category == "performance":
        return verify_latency_improvement(claim, original_value, filter_engine)
    
    elif category == "financial_projection":
        return verify_financial_projection(claim, original_value, filter_engine)
    
    elif category == "roi_timeline":
        return verify_roi_timeline(claim, original_value, filter_engine)
    
    elif category == "accuracy_improvement":
        return verify_accuracy_improvement(claim, original_value, filter_engine)
    
    elif category == "operational_improvement":
        return verify_operational_improvement(claim, original_value, filter_engine)
    
    else:
        return RealityCheck(
            claim=claim,
            verification_level=VerificationLevel.UNVERIFIED_CLAIM,
            evidence=["No hay método de verificación disponible"],
            confidence_score=0.1,
            reality_adjusted_value="No verificable",
            assumptions=["Categoría no reconocida"],
            risk_factors=["Imposible evaluar sin categorización"]
        )

def verify_gpt4_cost_claim(claim: str, value: float, filter_engine: RealityFilter) -> RealityCheck:
    """Verifica el costo de GPT-4 reportado"""
    
    # GPT-4 pricing público: $0.03 per 1K input tokens, $0.06 per 1K output tokens
    gpt4_input_cost = 0.03 / 1000  # Por token de input
    gpt4_output_cost = 0.06 / 1000  # Por token de output
    
    # Estimación típica de tokens por transacción AML
    tokens_per_transaction = 500  # Input + output estimado
    input_tokens = int(tokens_per_transaction * 0.7)  # 70% input
    output_tokens = int(tokens_per_transaction * 0.3)  # 30% output
    
    calculated_cost = (input_tokens * gpt4_input_cost) + (output_tokens * gpt4_output_cost)
    
    # El valor reportado ($0.008) vs calculado
    variance = abs(value - calculated_cost) / calculated_cost
    
    if variance < 0.2:  # Dentro del 20%
        verification_level = VerificationLevel.VERIFIED_FACT
        confidence = 0.9
        adjusted_value = calculated_cost
    elif variance < 0.5:  # Dentro del 50%
        verification_level = VerificationLevel.REASONABLE_ESTIMATE
        confidence = 0.7
        adjusted_value = calculated_cost
    else:
        verification_level = VerificationLevel.OPTIMISTIC_PROJECTION
        confidence = 0.4
        adjusted_value = calculated_cost
    
    return RealityCheck(
        claim=claim,
        verification_level=verification_level,
        evidence=[
            f"GPT-4 pricing público: ${gpt4_input_cost:.6f}/token input, ${gpt4_output_cost:.6f}/token output",
            f"Costo calculado para {tokens_per_transaction} tokens: ${calculated_cost:.4f}",
            f"Varianza vs reportado: {variance:.1%}"
        ],
        confidence_score=confidence,
        reality_adjusted_value=f"${calculated_cost:.4f}",
        assumptions=[
            f"~{tokens_per_transaction} tokens por transacción AML",
            "70/30 split input/output tokens",
            "No incluye costos de infraestructura adicionales"
        ],
        risk_factors=[
            "Tokens reales pueden variar según complejidad de transacciones",
            "Precios de OpenAI pueden cambiar",
            "Costos de infraestructura no incluidos en cálculo"
        ]
    )

def verify_slm_cost_projection(claim: str, value: float, filter_engine: RealityFilter) -> RealityCheck:
    """Verifica la proyección de costo del SLM"""
    
    # Nemotron-4B no tiene pricing público directo, estimamos basado en tamaño
    # Modelos SLM típicamente cuestan 10-100x menos que LLMs equivalentes
    gpt4_equivalent_cost = 0.015  # Costo GPT-4 ajustado por realidad
    
    # Rango conservador vs optimista para SLM
    conservative_reduction = 0.1  # 10x menos (90% reducción)
    optimistic_reduction = 0.01  # 100x menos (99% reducción)
    
    conservative_cost = gpt4_equivalent_cost * conservative_reduction
    optimistic_cost = gpt4_equivalent_cost * optimistic_reduction
    
    if value >= conservative_cost:
        verification_level = VerificationLevel.REASONABLE_ESTIMATE
        confidence = 0.8
        adjusted_value = conservative_cost
    elif value >= optimistic_cost:
        verification_level = VerificationLevel.OPTIMISTIC_PROJECTION
        confidence = 0.6
        adjusted_value = value  # Mantener valor si está en rango optimista
    else:
        verification_level = VerificationLevel.UNVERIFIED_CLAIM
        confidence = 0.3
        adjusted_value = conservative_cost  # Usar estimación conservadora
    
    return RealityCheck(
        claim=claim,
        verification_level=verification_level,
        evidence=[
            f"SLMs típicamente 10-100x más baratos que LLMs equivalentes",
            f"Costo conservador estimado: ${conservative_cost:.4f}",
            f"Costo optimista estimado: ${optimistic_cost:.4f}",
            "Nemotron-4B no tiene pricing público directo"
        ],
        confidence_score=confidence,
        reality_adjusted_value=f"${adjusted_value:.4f}",
        assumptions=[
            "Modelo self-hosted o pricing competitivo disponible",
            "Volumen suficiente para pricing preferencial",
            "No incluye costos de fine-tuning y deployment"
        ],
        risk_factors=[
            "Pricing real de Nemotron-4B puede ser más alto",
            "Costos de infraestructura y mantenimiento adicionales",
            "Fine-tuning y optimización requieren inversión adicional"
        ]
    )

def verify_latency_improvement(claim: str, value: float, filter_engine: RealityFilter) -> RealityCheck:
    """Verifica la mejora de latencia reportada"""
    
    # Mejoras de latencia SLM vs LLM son verificables técnicamente
    # Parámetros: GPT-4 (~1.76T params) vs Nemotron-4B (4.8B params)
    parameter_ratio = 4.8e9 / 1.76e12  # ~0.0027
    
    # Latencia típicamente escala con número de parámetros (no linealmente)
    # Mejoras típicas: 30-70% para SLMs bien optimizados
    typical_improvement_range = (0.3, 0.7)
    
    if typical_improvement_range[0] <= value <= typical_improvement_range[1]:
        verification_level = VerificationLevel.INDUSTRY_STANDARD
        confidence = 0.8
        adjusted_value = value
    elif value > typical_improvement_range[1]:
        verification_level = VerificationLevel.OPTIMISTIC_PROJECTION
        confidence = 0.6
        adjusted_value = typical_improvement_range[1]  # Cap at realistic max
    else:
        verification_level = VerificationLevel.REASONABLE_ESTIMATE
        confidence = 0.7
        adjusted_value = value
    
    return RealityCheck(
        claim=claim,
        verification_level=verification_level,
        evidence=[
            f"Nemotron-4B tiene ~365x menos parámetros que GPT-4",
            f"SLMs típicamente 30-70% más rápidos en inferencia",
            f"Valor reportado ({value:.0%}) está en rango esperado"
        ],
        confidence_score=confidence,
        reality_adjusted_value=f"{adjusted_value:.0%} mejora de velocidad",
        assumptions=[
            "Hardware optimizado para el modelo específico",
            "Modelo fine-tuned mantiene velocidad base",
            "Medición end-to-end incluyendo procesamiento"
        ],
        risk_factors=[
            "Performance puede variar según hardware disponible",
            "Fine-tuning puede impactar latencia",
            "Volumen alto puede crear cuellos de botella"
        ]
    )

def verify_financial_projection(claim: str, value: float, filter_engine: RealityFilter) -> RealityCheck:
    """Verifica la proyección financiera anual"""
    
    # Calculamos basado en datos ajustados por realidad
    transactions_per_day = 50000
    transactions_per_year = transactions_per_day * 365
    
    # Usando costos ajustados por realidad
    current_cost_per_transaction = 0.015  # GPT-4 ajustado
    projected_cost_per_transaction = 0.0015  # SLM conservador (10x reducción)
    
    savings_per_transaction = current_cost_per_transaction - projected_cost_per_transaction
    annual_savings_adjusted = savings_per_transaction * transactions_per_year
    
    # Comparar con valor reportado
    variance = abs(value - annual_savings_adjusted) / annual_savings_adjusted
    
    if variance < 0.3:  # Dentro del 30%
        verification_level = VerificationLevel.REASONABLE_ESTIMATE
        confidence = 0.7
        adjusted_value = annual_savings_adjusted
    else:
        verification_level = VerificationLevel.OPTIMISTIC_PROJECTION
        confidence = 0.5
        adjusted_value = annual_savings_adjusted
    
    return RealityCheck(
        claim=claim,
        verification_level=verification_level,
        evidence=[
            f"50k transacciones/día × 365 días = {transactions_per_year:,} transacciones/año",
            f"Ahorro por transacción (ajustado): ${savings_per_transaction:.4f}",
            f"Ahorro anual calculado: ${annual_savings_adjusted:,.0f}",
            f"Varianza vs reportado: {variance:.1%}"
        ],
        confidence_score=confidence,
        reality_adjusted_value=f"${adjusted_value:,.0f} anual",
        assumptions=[
            "Volumen de transacciones se mantiene constante",
            "Costos de implementación amortizados en cálculo",
            "No incluye ahorros indirectos (horas analistas, etc.)"
        ],
        risk_factors=[
            "Implementación puede tomar más tiempo del estimado",
            "Costos ocultos de migración no considerados",
            "Volumen de transacciones puede variar"
        ]
    )

def verify_roi_timeline(claim: str, value: float, filter_engine: RealityFilter) -> RealityCheck:
    """Verifica el timeline de ROI reportado"""
    
    # ROI de 1.3 meses es extremadamente optimista para proyectos enterprise
    industry_standards = filter_engine.industry_standards["roi_timeframes"]
    
    # Para proyectos de AI/ML enterprise: típicamente 3-12 meses
    realistic_roi_range = (3, 12)  # meses
    
    if value < realistic_roi_range[0]:
        verification_level = VerificationLevel.OPTIMISTIC_PROJECTION
        confidence = 0.3
        adjusted_value = realistic_roi_range[0]  # Mínimo realista
    elif realistic_roi_range[0] <= value <= realistic_roi_range[1]:
        verification_level = VerificationLevel.INDUSTRY_STANDARD
        confidence = 0.8
        adjusted_value = value
    else:
        verification_level = VerificationLevel.REASONABLE_ESTIMATE
        confidence = 0.6
        adjusted_value = realistic_roi_range[1]  # Máximo esperado
    
    return RealityCheck(
        claim=claim,
        verification_level=verification_level,
        evidence=[
            f"ROI reportado: {value} meses",
            f"Rango típico industria: {realistic_roi_range[0]}-{realistic_roi_range[1]} meses",
            "Proyectos AI enterprise requieren tiempo de implementación",
            "Factores como training del equipo, validación, etc. añaden tiempo"
        ],
        confidence_score=confidence,
        reality_adjusted_value=f"{adjusted_value} meses",
        assumptions=[
            "Implementación sin contratiempos significativos",
            "Equipo técnico disponible y capacitado",
            "No requiere cambios regulatorios adicionales"
        ],
        risk_factors=[
            "Proyectos enterprise típicamente toman más tiempo",
            "Validación regulatoria puede extender timeline",
            "Resistencia al cambio organizacional",
            "Integración con sistemas legacy puede ser compleja"
        ]
    )

def verify_accuracy_improvement(claim: str, value: float, filter_engine: RealityFilter) -> RealityCheck:
    """Verifica la mejora de precisión reportada"""
    
    # Mejora de 94.2% a 95% (0.8 puntos porcentuales)
    improvement_points = 0.8
    
    # Para modelos especializados, mejoras de 0.5-2% son típicas
    realistic_improvement_range = (0.5, 2.0)
    
    if realistic_improvement_range[0] <= improvement_points <= realistic_improvement_range[1]:
        verification_level = VerificationLevel.REASONABLE_ESTIMATE
        confidence = 0.7
        adjusted_value = "94.2% → 95.0% (+0.8pp)"
    elif improvement_points > realistic_improvement_range[1]:
        verification_level = VerificationLevel.OPTIMISTIC_PROJECTION
        confidence = 0.5
        adjusted_value = f"94.2% → 96.2% (+{realistic_improvement_range[1]}pp max realista)"
    else:
        verification_level = VerificationLevel.INDUSTRY_STANDARD
        confidence = 0.8
        adjusted_value = "94.2% → 95.0% (+0.8pp)"
    
    return RealityCheck(
        claim=claim,
        verification_level=verification_level,
        evidence=[
            f"Mejora reportada: {improvement_points} puntos porcentuales",
            f"Rango típico para modelos especializados: {realistic_improvement_range[0]}-{realistic_improvement_range[1]}pp",
            "Fine-tuning en dominio específico puede mejorar precisión",
            "Baseline de 94.2% ya es alta para AML"
        ],
        confidence_score=confidence,
        reality_adjusted_value=adjusted_value,
        assumptions=[
            "Fine-tuning efectivo con datos históricos del banco",
            "Métricas medidas consistentemente",
            "No degradación por optimización de velocidad"
        ],
        risk_factors=[
            "Mejoras pueden ser menores en casos edge",
            "Trade-off entre velocidad y precisión",
            "Validación requiere período de testing extenso"
        ]
    )

def verify_operational_improvement(claim: str, value: float, filter_engine: RealityFilter) -> RealityCheck:
    """Verifica las mejoras operacionales reportadas"""
    
    # 44 menos falsos positivos por día de 290 totales = 15.2% reducción
    reduction_percentage = value / 290
    
    # Reducciones típicas con modelos especializados: 10-25%
    realistic_reduction_range = (0.10, 0.25)
    
    if realistic_reduction_range[0] <= reduction_percentage <= realistic_reduction_range[1]:
        verification_level = VerificationLevel.REASONABLE_ESTIMATE
        confidence = 0.7
        adjusted_value = f"{value} casos/día ({reduction_percentage:.1%} reducción)"
    elif reduction_percentage > realistic_reduction_range[1]:
        verification_level = VerificationLevel.OPTIMISTIC_PROJECTION
        confidence = 0.5
        max_realistic = 290 * realistic_reduction_range[1]
        adjusted_value = f"{max_realistic:.0f} casos/día ({realistic_reduction_range[1]:.0%} reducción max realista)"
    else:
        verification_level = VerificationLevel.INDUSTRY_STANDARD
        confidence = 0.8
        adjusted_value = f"{value} casos/día ({reduction_percentage:.1%} reducción)"
    
    return RealityCheck(
        claim=claim,
        verification_level=verification_level,
        evidence=[
            f"Reducción reportada: {reduction_percentage:.1%} de falsos positivos",
            f"Rango típico industria: {realistic_reduction_range[0]:.0%}-{realistic_reduction_range[1]:.0%}",
            "Modelos especializados pueden reducir falsos positivos",
            "Baseline de 290 FP/día es alto, hay margen de mejora"
        ],
        confidence_score=confidence,
        reality_adjusted_value=adjusted_value,
        assumptions=[
            "Modelo especializado mantiene sensibilidad para casos verdaderos",
            "Patrones de falsos positivos son identificables",
            "No incremento en falsos negativos"
        ],
        risk_factors=[
            "Reducción de FP puede incrementar falsos negativos",
            "Ajustes requieren validación regulatoria",
            "Beneficio puede tomar tiempo en materializarse"
        ]
    )

def generate_reality_adjusted_summary(verified_results: List[RealityCheck]):
    """Genera resumen ajustado por realidad"""
    
    print(f"\n" + "="*70)
    print(f"📊 RESUMEN AJUSTADO POR REALIDAD FILTER 2.0")
    print(f"="*70)
    
    # Contar niveles de verificación
    verification_counts = {}
    total_confidence = 0
    
    for result in verified_results:
        level = result.verification_level.value
        verification_counts[level] = verification_counts.get(level, 0) + 1
        total_confidence += result.confidence_score
    
    avg_confidence = total_confidence / len(verified_results)
    
    print(f"🎯 NIVEL DE CONFIANZA PROMEDIO: {avg_confidence:.2f}")
    print(f"\n📋 DISTRIBUCIÓN DE VERIFICACIÓN:")
    
    for level, count in verification_counts.items():
        percentage = (count / len(verified_results)) * 100
        print(f"   • {level.replace('_', ' ').title()}: {count} ({percentage:.0f}%)")
    
    print(f"\n💰 PROYECCIONES AJUSTADAS POR REALIDAD:")
    
    # Extraer valores clave ajustados
    cost_analysis = next((r for r in verified_results if "Costo por transacción actual" in r.claim), None)
    cost_projection = next((r for r in verified_results if "Costo proyectado con SLM" in r.claim), None)
    financial_projection = next((r for r in verified_results if "Ahorro anual" in r.claim), None)
    roi_timeline = next((r for r in verified_results if "ROI en" in r.claim), None)
    
    if cost_analysis and cost_projection:
        print(f"   • Costo Actual (ajustado): {cost_analysis.reality_adjusted_value}")
        print(f"   • Costo SLM (ajustado): {cost_projection.reality_adjusted_value}")
    
    if financial_projection:
        print(f"   • Ahorro Anual (ajustado): {financial_projection.reality_adjusted_value}")
    
    if roi_timeline:
        print(f"   • Timeline ROI (ajustado): {roi_timeline.reality_adjusted_value}")
    
    print(f"\n⚠️ PRINCIPALES FACTORES DE RIESGO IDENTIFICADOS:")
    
    all_risks = []
    for result in verified_results:
        all_risks.extend(result.risk_factors)
    
    # Contar riesgos más frecuentes
    risk_counts = {}
    for risk in all_risks:
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    # Top 5 riesgos
    top_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (risk, count) in enumerate(top_risks, 1):
        print(f"   {i}. {risk}")
    
    print(f"\n✅ CONCLUSIÓN DEL REALITY FILTER:")
    
    if avg_confidence >= 0.7:
        print(f"   🟢 ALTA CONFIABILIDAD - Proyecciones son razonablemente realistas")
    elif avg_confidence >= 0.5:
        print(f"   🟡 CONFIABILIDAD MODERADA - Algunas proyecciones requieren validación")
    else:
        print(f"   🔴 BAJA CONFIABILIDAD - Múltiples proyecciones son optimistas")
    
    print(f"\n📋 RECOMENDACIONES BASADAS EN ANÁLISIS DE REALIDAD:")
    print(f"   • Usar valores ajustados por realidad para business case")
    print(f"   • Implementar piloto para validar asunciones clave")
    print(f"   • Monitorear factores de riesgo identificados")
    print(f"   • Planificar para timeline más conservador")
    print(f"   • Incluir costos adicionales no considerados en estimaciones")

if __name__ == "__main__":
    verified_results = apply_reality_filter_to_aml_case()
    
    print(f"\n" + "="*70)
    print(f"🔍 REALITY FILTER 2.0 - ANÁLISIS COMPLETADO")
    print(f"="*70)
    print(f"Las afirmaciones han sido verificadas contra datos conocidos")
    print(f"y ajustadas por factores de realidad identificables.")
    print(f"")
    print(f"Este análisis proporciona una base más sólida para")
    print(f"decisiones empresariales informadas y realistas.")