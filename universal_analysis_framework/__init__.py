"""
Universal Analysis Framework
Implementación de los 8 meta-principios universales aplicables a cualquier dominio de análisis.

Meta-Principios:
1. Marco de abstención matemática (Mathematical Abstention Framework)
2. Límites de confianza en decisiones (Confidence Bounds in Decisions)  
3. Análisis genealógico de influencias (Genealogical Analysis of Influences)
4. Pipeline multi-etapa con validación (Multi-stage Pipeline with Validation)
5. Evaluación ensemble multi-modelo (Multi-model Ensemble Evaluation)
6. Cuantificación de incertidumbre (Uncertainty Quantification)
7. Salida estructurada con metadatos (Structured Output with Metadata)
8. Hibridación adaptativa (Adaptive Hybridization)
"""

__version__ = "1.0.0"
__author__ = "Universal Framework Team"
__email__ = "framework@integrid.ai"

# Core imports
from .core.universal_framework import (
    UniversalAnalyzer,
    UniversalResult, 
    UniversalMetadata,
    universal_registry,
    AnalysisStage,
    ConfidenceLevel
)

# Mathematical framework
from .mathematical.abstention_framework import (
    UniversalMathematicalFramework,
    BoundCalculationMethod,
    RiskLevel,
    MathematicalBound,
    AbstractionDecision,
    universal_math_framework,
    universal_uncertainty_quantifier
)

# Ensemble evaluation  
from .ensemble.multi_model_evaluator import (
    UniversalEnsembleEvaluator,
    UniversalModel,
    EnsembleStrategy,
    ModelType,
    EnsembleResult,
    universal_ensemble_evaluator
)

# Genealogical analysis
from .genealogical.influence_tracker import (
    UniversalInfluenceTracker,
    InfluenceType,
    NodeType,
    InfluenceRelation,
    GenealogyNode,
    GenealogyAnalysis
)

# Adaptive hybridization
from .hybridization.adaptive_hybridizer import (
    UniversalAdaptiveHybridizer,
    HybridizationStrategy,
    ComponentType,
    HybridizationContext,
    HybridizationResult,
    universal_adaptive_hybridizer
)

__all__ = [
    # Core framework
    "UniversalAnalyzer",
    "UniversalResult", 
    "UniversalMetadata",
    "universal_registry",
    "AnalysisStage",
    "ConfidenceLevel",
    
    # Mathematical abstention
    "UniversalMathematicalFramework",
    "BoundCalculationMethod",
    "RiskLevel", 
    "MathematicalBound",
    "AbstractionDecision",
    "universal_math_framework",
    "universal_uncertainty_quantifier",
    
    # Ensemble evaluation
    "UniversalEnsembleEvaluator",
    "UniversalModel",
    "EnsembleStrategy",
    "ModelType",
    "EnsembleResult", 
    "universal_ensemble_evaluator",
    
    # Genealogical analysis
    "UniversalInfluenceTracker",
    "InfluenceType",
    "NodeType",
    "InfluenceRelation",
    "GenealogyNode", 
    "GenealogyAnalysis",
    
    # Adaptive hybridization
    "UniversalAdaptiveHybridizer",
    "HybridizationStrategy",
    "ComponentType",
    "HybridizationContext",
    "HybridizationResult",
    "universal_adaptive_hybridizer",
]