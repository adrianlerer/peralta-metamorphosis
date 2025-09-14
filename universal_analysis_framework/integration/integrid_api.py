"""
IntegridAI Suite Integration API
API de integración para el Universal Analysis Framework con la suite IntegridAI.

Permite la reutilización de los meta-principios universales en otros desarrollos
y proporciona interfaces estándar para integración empresarial.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Imports del framework universal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.universal_framework import (
    UniversalAnalyzer, UniversalResult, universal_registry
)
from mathematical.abstention_framework import (
    BoundCalculationMethod, RiskLevel, universal_math_framework,
    universal_uncertainty_quantifier
)
from ensemble.multi_model_evaluator import (
    universal_ensemble_evaluator, EnsembleStrategy
)
from genealogical.influence_tracker import (
    universal_influence_tracker, InfluenceType
)
from hybridization.adaptive_hybridizer import (
    universal_adaptive_hybridizer, HybridizationStrategy
)

# Modelos Pydantic para la API
class AnalysisRequest(BaseModel):
    """Solicitud de análisis universal"""
    domain: str = Field(..., description="Dominio del análisis (e.g., 'text_analysis', 'financial_analysis')")
    input_data: Any = Field(..., description="Datos de entrada para el análisis")
    confidence_threshold: Optional[float] = Field(0.80, ge=0.0, le=1.0, description="Umbral de confianza")
    enable_abstention: Optional[bool] = Field(True, description="Habilitar abstención matemática")
    ensemble_strategy: Optional[str] = Field("confidence_weighted", description="Estrategia ensemble")
    hybridization_strategy: Optional[str] = Field("context_adaptive", description="Estrategia de hibridización")
    
class AnalysisResponse(BaseModel):
    """Respuesta del análisis universal"""
    success: bool
    analysis_id: str
    domain: str
    result: Optional[Any] = None
    confidence: float
    abstained: bool
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None

class MathematicalBoundRequest(BaseModel):
    """Solicitud de cálculo de límites matemáticos"""
    data: List[float] = Field(..., description="Datos numéricos para calcular límites")
    methods: List[str] = Field(default=["bootstrap_percentile"], description="Métodos de cálculo")
    confidence_level: Optional[float] = Field(0.95, ge=0.01, le=0.99, description="Nivel de confianza")

class EnsembleEvaluationRequest(BaseModel):
    """Solicitud de evaluación ensemble"""
    input_data: Any = Field(..., description="Datos para evaluación ensemble")
    model_ids: Optional[List[str]] = Field(None, description="IDs de modelos específicos (opcional)")
    strategy: Optional[str] = Field("confidence_weighted", description="Estrategia de combinación")
    weights: Optional[Dict[str, float]] = Field(None, description="Pesos específicos por modelo")

class HybridizationRequest(BaseModel):
    """Solicitud de hibridización adaptativa"""
    domain: str = Field(..., description="Dominio del análisis")
    data_characteristics: Dict[str, Any] = Field(default_factory=dict)
    performance_requirements: Dict[str, float] = Field(default_factory=dict)
    resource_constraints: Dict[str, Any] = Field(default_factory=dict)
    strategy: Optional[str] = Field("context_adaptive", description="Estrategia de hibridización")

class ComponentRegistrationRequest(BaseModel):
    """Solicitud de registro de componente"""
    component_id: str = Field(..., description="ID único del componente")
    component_type: str = Field(..., description="Tipo del componente")
    domain: str = Field(..., description="Dominio aplicable")
    performance_metrics: Optional[Dict[str, float]] = Field(default_factory=dict)
    context_suitability: Optional[Dict[str, float]] = Field(default_factory=dict)
    computational_cost: Optional[float] = Field(1.0, ge=0.1)
    reliability_score: Optional[float] = Field(1.0, ge=0.0, le=1.0)

# Configuración de la API
class IntegridAPIConfig:
    """Configuración de la API IntegridAI"""
    def __init__(self):
        self.title = "Universal Analysis Framework API"
        self.version = "1.0.0"
        self.description = """
        API de integración para el Universal Analysis Framework aplicando los 8 meta-principios universales:
        
        1. **Marco de abstención matemática**: Abstención inteligente con garantías estadísticas
        2. **Límites de confianza**: Cálculo de intervalos de confianza con múltiples métodos
        3. **Análisis genealógico**: Rastreo de influencias y dependencias
        4. **Pipeline multi-etapa**: Validación en múltiples etapas
        5. **Evaluación ensemble**: Combinación de múltiples modelos/métodos
        6. **Cuantificación de incertidumbre**: Medición rigurosa de incertidumbre
        7. **Salida estructurada**: Metadatos completos y trazabilidad
        8. **Hibridización adaptativa**: Combinación dinámica según contexto
        
        Compatible con IntegridAI Suite para reutilización en múltiples desarrollos.
        """
        self.contact = {
            "name": "Universal Framework Team",
            "email": "framework@integrid.ai"
        }

# Inicializar FastAPI
config = IntegridAPIConfig()
app = FastAPI(
    title=config.title,
    version=config.version,
    description=config.description,
    contact=config.contact
)

# Configurar CORS
app.add_middleware(
    CORsMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger
logger = logging.getLogger("IntegridAPI")

# Executor para tareas asíncronas
executor = ThreadPoolExecutor(max_workers=4)

# Variables globales para tracking
analysis_history = []
active_analyses = {}

@app.on_event("startup")
async def startup_event():
    """Inicialización de la API"""
    logger.info("🚀 Iniciando Universal Analysis Framework API")
    
    # Registrar analizadores de ejemplo
    try:
        from domains.text_analysis_example import SimpleTextAnalyzer
        text_analyzer = SimpleTextAnalyzer()
        universal_registry.register_analyzer("text_analysis", text_analyzer)
        logger.info("✅ Analizador de texto registrado")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo cargar analizador de texto: {e}")
    
    try:
        from domains.financial_analysis_example import FinancialAnalyzer
        financial_analyzer = FinancialAnalyzer()
        universal_registry.register_analyzer("financial_analysis", financial_analyzer)
        logger.info("✅ Analizador financiero registrado")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo cargar analizador financiero: {e}")
    
    logger.info("🎯 API lista para recibir solicitudes")

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "service": "Universal Analysis Framework API",
        "version": config.version,
        "status": "operational",
        "meta_principles": [
            "mathematical_abstention",
            "confidence_bounds", 
            "genealogical_analysis",
            "multi_stage_pipeline",
            "ensemble_evaluation",
            "uncertainty_quantification",
            "structured_output",
            "adaptive_hybridization"
        ],
        "available_domains": universal_registry.list_domains(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    return {
        "status": "healthy",
        "framework_components": {
            "universal_registry": len(universal_registry.list_domains()),
            "math_framework": "operational",
            "ensemble_evaluator": "operational", 
            "influence_tracker": "operational",
            "adaptive_hybridizer": "operational"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Endpoint principal para análisis universal aplicando los 8 meta-principios
    """
    start_time = datetime.now()
    analysis_id = f"analysis_{int(start_time.timestamp())}"
    
    try:
        # Verificar si el dominio está registrado
        analyzer = universal_registry.get_analyzer(request.domain)
        if not analyzer:
            raise HTTPException(
                status_code=404,
                detail=f"Dominio '{request.domain}' no encontrado. Dominios disponibles: {universal_registry.list_domains()}"
            )
        
        # Configurar analizador según solicitud
        analyzer.confidence_threshold = request.confidence_threshold
        analyzer.enable_abstention = request.enable_abstention
        
        # Ejecutar análisis en hilo separado
        def run_analysis():
            return analyzer.analyze(request.input_data)
        
        # Ejecutar análisis
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_analysis)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Registrar análisis en historial
        analysis_record = {
            "analysis_id": analysis_id,
            "domain": request.domain,
            "timestamp": start_time.isoformat(),
            "processing_time": processing_time,
            "confidence": result.confidence,
            "abstained": result.abstained,
            "success": True
        }
        
        # Añadir a historial en background
        background_tasks.add_task(record_analysis, analysis_record)
        
        return AnalysisResponse(
            success=True,
            analysis_id=analysis_id,
            domain=request.domain,
            result=result.result,
            confidence=result.confidence,
            abstained=result.abstained,
            processing_time=processing_time,
            metadata=result.to_dict()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error en análisis {analysis_id}: {str(e)}")
        
        # Registrar error en historial
        error_record = {
            "analysis_id": analysis_id,
            "domain": request.domain,
            "timestamp": start_time.isoformat(),
            "processing_time": processing_time,
            "error": str(e),
            "success": False
        }
        background_tasks.add_task(record_analysis, error_record)
        
        return AnalysisResponse(
            success=False,
            analysis_id=analysis_id,
            domain=request.domain,
            confidence=0.0,
            abstained=True,
            processing_time=processing_time,
            error_message=str(e)
        )

@app.post("/mathematical/bounds")
async def calculate_bounds(request: MathematicalBoundRequest):
    """
    Calcula límites de confianza usando múltiples métodos matemáticos
    """
    try:
        # Mapear nombres de métodos a enum
        method_mapping = {
            "hoeffding": BoundCalculationMethod.HOEFFDING,
            "bootstrap_percentile": BoundCalculationMethod.BOOTSTRAP_PERCENTILE,
            "bootstrap_bca": BoundCalculationMethod.BOOTSTRAP_BCa,
            "clopper_pearson": BoundCalculationMethod.CLOPPER_PEARSON,
            "wilson_score": BoundCalculationMethod.WILSON_SCORE,
            "bayesian_credible": BoundCalculationMethod.BAYESIAN_CREDIBLE,
            "ensemble_variance": BoundCalculationMethod.ENSEMBLE_VARIANCE
        }
        
        methods = []
        for method_name in request.methods:
            if method_name in method_mapping:
                methods.append(method_mapping[method_name])
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Método '{method_name}' no reconocido. Métodos disponibles: {list(method_mapping.keys())}"
                )
        
        # Calcular límites
        bounds = universal_math_framework.calculate_multiple_bounds(
            request.data, methods, request.confidence_level
        )
        
        # Decisión de abstención
        abstention_decision = universal_math_framework.make_abstention_decision(
            bounds, confidence_threshold=0.80, width_threshold=0.3
        )
        
        return {
            "success": True,
            "bounds": {name: bound.to_dict() for name, bound in bounds.items()},
            "abstention_analysis": abstention_decision.to_dict(),
            "summary": {
                "methods_used": len(bounds),
                "confidence_level": request.confidence_level,
                "data_points": len(request.data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculando límites: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ensemble/evaluate")
async def ensemble_evaluate(request: EnsembleEvaluationRequest):
    """
    Realiza evaluación ensemble con múltiples modelos
    """
    try:
        # Mapear estrategia
        strategy_mapping = {
            "simple_average": EnsembleStrategy.SIMPLE_AVERAGE,
            "weighted_average": EnsembleStrategy.WEIGHTED_AVERAGE,
            "majority_vote": EnsembleStrategy.MAJORITY_VOTE,
            "confidence_weighted": EnsembleStrategy.CONFIDENCE_WEIGHTED,
            "dynamic_selection": EnsembleStrategy.DYNAMIC_SELECTION
        }
        
        strategy = strategy_mapping.get(
            request.strategy, EnsembleStrategy.CONFIDENCE_WEIGHTED
        )
        
        # Ejecutar evaluación ensemble
        result = universal_ensemble_evaluator.evaluate(
            request.input_data,
            strategy=strategy,
            model_weights=request.weights
        )
        
        return {
            "success": True,
            "result": result.to_dict(),
            "summary": {
                "strategy_used": result.strategy_used.value,
                "models_evaluated": len(result.individual_results),
                "successful_models": len([r for r in result.individual_results if r.error is None]),
                "overall_confidence": result.overall_confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Error en evaluación ensemble: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybridization/analyze")
async def hybridization_analyze(request: HybridizationRequest):
    """
    Realiza hibridización adaptativa de componentes
    """
    try:
        from hybridization.adaptive_hybridizer import HybridizationContext, HybridizationStrategy
        
        # Mapear estrategia
        strategy_mapping = {
            "context_adaptive": HybridizationStrategy.CONTEXT_ADAPTIVE,
            "performance_based": HybridizationStrategy.PERFORMANCE_BASED,
            "confidence_driven": HybridizationStrategy.CONFIDENCE_DRIVEN,
            "data_dependent": HybridizationStrategy.DATA_DEPENDENT
        }
        
        strategy = strategy_mapping.get(
            request.strategy, HybridizationStrategy.CONTEXT_ADAPTIVE
        )
        
        # Crear contexto de hibridización
        context = HybridizationContext(
            domain=request.domain,
            data_characteristics=request.data_characteristics,
            performance_requirements=request.performance_requirements,
            resource_constraints=request.resource_constraints
        )
        
        # Ejecutar hibridización
        result = universal_adaptive_hybridizer.hybridize(context, strategy)
        
        return {
            "success": True,
            "hybridization_result": result.to_dict(),
            "summary": {
                "selected_components": len(result.selected_components),
                "strategy_used": result.hybridization_strategy.value,
                "confidence_score": result.confidence_score,
                "adaptation_reasoning": len(result.adaptation_reasoning)
            }
        }
        
    except Exception as e:
        logger.error(f"Error en hibridización: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uncertainty/quantify")
async def quantify_uncertainty(
    predictions: List[Any],
    ground_truth: Optional[List[Any]] = None,
    uncertainty_type: str = "epistemic"
):
    """
    Cuantifica incertidumbre en predicciones
    """
    try:
        # Cuantificar incertidumbre
        uncertainty_metrics = universal_uncertainty_quantifier.quantify_prediction_uncertainty(
            predictions, ground_truth, uncertainty_type
        )
        
        return {
            "success": True,
            "uncertainty_metrics": uncertainty_metrics,
            "summary": {
                "predictions_analyzed": len(predictions),
                "has_ground_truth": ground_truth is not None,
                "uncertainty_type": uncertainty_type
            }
        }
        
    except Exception as e:
        logger.error(f"Error cuantificando incertidumbre: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/domains")
async def list_domains():
    """
    Lista dominios de análisis disponibles
    """
    domains = universal_registry.list_domains()
    return {
        "success": True,
        "domains": domains,
        "count": len(domains)
    }

@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Obtiene información de un análisis específico
    """
    # Buscar en historial
    analysis = next((a for a in analysis_history if a.get("analysis_id") == analysis_id), None)
    
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Análisis {analysis_id} no encontrado")
    
    return {
        "success": True,
        "analysis": analysis
    }

@app.get("/history")
async def get_analysis_history(limit: int = 50, domain: Optional[str] = None):
    """
    Obtiene historial de análisis
    """
    filtered_history = analysis_history
    
    if domain:
        filtered_history = [a for a in filtered_history if a.get("domain") == domain]
    
    # Limitar resultados
    limited_history = filtered_history[-limit:]
    
    return {
        "success": True,
        "history": limited_history,
        "total_records": len(analysis_history),
        "filtered_records": len(filtered_history),
        "returned_records": len(limited_history)
    }

@app.get("/stats")
async def get_statistics():
    """
    Obtiene estadísticas de uso de la API
    """
    if not analysis_history:
        return {
            "success": True,
            "message": "No hay datos estadísticos disponibles",
            "stats": {}
        }
    
    # Calcular estadísticas
    total_analyses = len(analysis_history)
    successful_analyses = len([a for a in analysis_history if a.get("success", False)])
    
    # Por dominio
    domain_stats = {}
    for analysis in analysis_history:
        domain = analysis.get("domain", "unknown")
        if domain not in domain_stats:
            domain_stats[domain] = {"count": 0, "success_rate": 0}
        domain_stats[domain]["count"] += 1
        if analysis.get("success", False):
            domain_stats[domain]["success_rate"] += 1
    
    # Calcular tasas de éxito por dominio
    for domain, stats in domain_stats.items():
        if stats["count"] > 0:
            stats["success_rate"] = stats["success_rate"] / stats["count"]
    
    # Tiempos de procesamiento
    processing_times = [a.get("processing_time", 0) for a in analysis_history if a.get("processing_time")]
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    return {
        "success": True,
        "stats": {
            "total_analyses": total_analyses,
            "successful_analyses": successful_analyses,
            "success_rate": successful_analyses / total_analyses if total_analyses > 0 else 0,
            "domain_statistics": domain_stats,
            "average_processing_time": avg_processing_time,
            "available_domains": universal_registry.list_domains()
        }
    }

# Funciones auxiliares
async def record_analysis(analysis_record: Dict[str, Any]):
    """Registra análisis en el historial"""
    analysis_history.append(analysis_record)
    
    # Mantener solo últimos 1000 registros
    if len(analysis_history) > 1000:
        analysis_history.pop(0)

# Ejecutar servidor de desarrollo
if __name__ == "__main__":
    uvicorn.run(
        "integrid_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )