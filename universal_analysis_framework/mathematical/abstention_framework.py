"""
Universal Mathematical Abstention Framework
Sistema matemático de abstención aplicable a cualquier dominio de análisis.

Implementa múltiples métodos de cálculo de límites con garantías estadísticas rigurosas.
Basado en el framework desarrollado para LexCertainty Enterprise pero generalizado
para cualquier tipo de análisis.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import scipy.stats as stats
try:
    from scipy.stats import bootstrap
except ImportError:
    # Fallback for older scipy versions
    bootstrap = None
import logging
from datetime import datetime
import json

class BoundCalculationMethod(Enum):
    """Métodos disponibles para cálculo de límites de confianza"""
    HOEFFDING = "hoeffding"
    BOOTSTRAP_PERCENTILE = "bootstrap_percentile"
    BOOTSTRAP_BCa = "bootstrap_bca"  # Bias-Corrected and accelerated
    CLOPPER_PEARSON = "clopper_pearson"
    WILSON_SCORE = "wilson_score"
    BAYESIAN_CREDIBLE = "bayesian_credible"
    ENSEMBLE_VARIANCE = "ensemble_variance"

class RiskLevel(Enum):
    """Niveles de riesgo para abstención"""
    VERY_LOW = 0.01    # 99% confianza
    LOW = 0.05         # 95% confianza  
    MEDIUM = 0.10      # 90% confianza
    HIGH = 0.20        # 80% confianza
    VERY_HIGH = 0.30   # 70% confianza

@dataclass
class MathematicalBound:
    """Representa un límite matemático con metadatos"""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: BoundCalculationMethod
    sample_size: Optional[int] = None
    bootstrap_iterations: Optional[int] = None
    calculation_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def width(self) -> float:
        """Ancho del intervalo de confianza"""
        return self.upper_bound - self.lower_bound
    
    @property
    def center(self) -> float:
        """Centro del intervalo"""
        return (self.lower_bound + self.upper_bound) / 2
    
    def contains(self, value: float) -> bool:
        """Verifica si un valor está dentro del límite"""
        return self.lower_bound <= value <= self.upper_bound
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "width": self.width,
            "center": self.center,
            "confidence_level": self.confidence_level,
            "method": self.method.value,
            "sample_size": self.sample_size,
            "bootstrap_iterations": self.bootstrap_iterations,
            "calculation_time": self.calculation_time,
            "metadata": self.metadata
        }

@dataclass
class AbstractionDecision:
    """Decisión de abstención con justificación matemática"""
    should_abstain: bool
    confidence_score: float
    risk_assessment: RiskLevel
    violated_bounds: List[str] = field(default_factory=list)
    mathematical_justification: str = ""
    bounds_summary: Dict[str, MathematicalBound] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "should_abstain": self.should_abstain,
            "confidence_score": self.confidence_score,
            "risk_assessment": self.risk_assessment.value,
            "violated_bounds": self.violated_bounds,
            "mathematical_justification": self.mathematical_justification,
            "bounds_summary": {k: v.to_dict() for k, v in self.bounds_summary.items()}
        }

class UniversalMathematicalFramework:
    """
    Framework matemático universal para abstención inteligente
    Aplicable a cualquier dominio de análisis
    """
    
    def __init__(
        self,
        default_confidence_level: float = 0.95,
        bootstrap_iterations: int = 1000,
        random_seed: Optional[int] = None
    ):
        self.default_confidence_level = default_confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        self.random_seed = random_seed
        self.logger = logging.getLogger("UniversalMathematicalFramework")
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def calculate_bound(
        self,
        data: Union[List[float], np.ndarray],
        method: BoundCalculationMethod,
        confidence_level: Optional[float] = None,
        **kwargs
    ) -> MathematicalBound:
        """
        Calcula límite de confianza usando el método especificado
        
        Args:
            data: Datos numéricos para calcular límites
            method: Método de cálculo
            confidence_level: Nivel de confianza (default: self.default_confidence_level)
            **kwargs: Parámetros específicos del método
        """
        start_time = datetime.now()
        confidence_level = confidence_level or self.default_confidence_level
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if len(data) == 0:
            raise ValueError("Los datos no pueden estar vacíos")
        
        try:
            if method == BoundCalculationMethod.HOEFFDING:
                bound = self._calculate_hoeffding_bound(data, confidence_level, **kwargs)
            elif method == BoundCalculationMethod.BOOTSTRAP_PERCENTILE:
                bound = self._calculate_bootstrap_percentile(data, confidence_level, **kwargs)
            elif method == BoundCalculationMethod.BOOTSTRAP_BCa:
                bound = self._calculate_bootstrap_bca(data, confidence_level, **kwargs)
            elif method == BoundCalculationMethod.CLOPPER_PEARSON:
                bound = self._calculate_clopper_pearson(data, confidence_level, **kwargs)
            elif method == BoundCalculationMethod.WILSON_SCORE:
                bound = self._calculate_wilson_score(data, confidence_level, **kwargs)
            elif method == BoundCalculationMethod.BAYESIAN_CREDIBLE:
                bound = self._calculate_bayesian_credible(data, confidence_level, **kwargs)
            elif method == BoundCalculationMethod.ENSEMBLE_VARIANCE:
                bound = self._calculate_ensemble_variance(data, confidence_level, **kwargs)
            else:
                raise ValueError(f"Método no soportado: {method}")
            
            # Agregar metadatos de cálculo
            calculation_time = (datetime.now() - start_time).total_seconds()
            bound.calculation_time = calculation_time
            bound.sample_size = len(data)
            
            return bound
            
        except Exception as e:
            self.logger.error(f"Error calculando límite con {method.value}: {str(e)}")
            raise
    
    def _calculate_hoeffding_bound(
        self, 
        data: np.ndarray, 
        confidence_level: float, 
        **kwargs
    ) -> MathematicalBound:
        """Calcula límites usando desigualdad de Hoeffding"""
        n = len(data)
        alpha = 1 - confidence_level
        
        # Asumimos datos en [0,1] para Hoeffding clásico
        data_range = kwargs.get('data_range', 1.0)
        
        mean = np.mean(data)
        epsilon = np.sqrt(-np.log(alpha/2) * data_range**2 / (2 * n))
        
        lower = max(0.0, mean - epsilon)
        upper = min(1.0, mean + epsilon)
        
        return MathematicalBound(
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            method=BoundCalculationMethod.HOEFFDING,
            metadata={"epsilon": epsilon, "data_range": data_range}
        )
    
    def _calculate_bootstrap_percentile(
        self, 
        data: np.ndarray, 
        confidence_level: float, 
        **kwargs
    ) -> MathematicalBound:
        """Calcula límites usando bootstrap percentil"""
        iterations = kwargs.get('bootstrap_iterations', self.bootstrap_iterations)
        statistic_func = kwargs.get('statistic', np.mean)
        
        bootstrap_samples = []
        for _ in range(iterations):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_samples.append(statistic_func(sample))
        
        bootstrap_samples = np.array(bootstrap_samples)
        alpha = 1 - confidence_level
        
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_samples, lower_percentile)
        upper = np.percentile(bootstrap_samples, upper_percentile)
        
        return MathematicalBound(
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            method=BoundCalculationMethod.BOOTSTRAP_PERCENTILE,
            bootstrap_iterations=iterations,
            metadata={"bootstrap_mean": np.mean(bootstrap_samples), "bootstrap_std": np.std(bootstrap_samples)}
        )
    
    def _calculate_bootstrap_bca(
        self, 
        data: np.ndarray, 
        confidence_level: float, 
        **kwargs
    ) -> MathematicalBound:
        """Calcula límites usando Bootstrap BCa (Bias-Corrected and accelerated)"""
        statistic_func = kwargs.get('statistic', np.mean)
        
        if bootstrap is None:
            # Fallback to percentile bootstrap if BCa is not available
            return self._calculate_bootstrap_percentile(data, confidence_level, **kwargs)
        
        try:
            # Usar scipy.bootstrap para BCa
            rng = np.random.default_rng(self.random_seed)
            
            def bootstrap_statistic(x, axis):
                return statistic_func(x, axis=axis)
            
            res = bootstrap(
                (data,), 
                bootstrap_statistic, 
                n_resamples=self.bootstrap_iterations,
                confidence_level=confidence_level,
                method='BCa',
                random_state=rng
            )
            
            return MathematicalBound(
                lower_bound=res.confidence_interval.low,
                upper_bound=res.confidence_interval.high,
                confidence_level=confidence_level,
                method=BoundCalculationMethod.BOOTSTRAP_BCa,
                bootstrap_iterations=self.bootstrap_iterations,
                metadata={"bootstrap_std": res.bootstrap_distribution.std()}
            )
        except Exception as e:
            # Fallback to percentile bootstrap
            self.logger.warning(f"BCa bootstrap failed, using percentile: {e}")
            return self._calculate_bootstrap_percentile(data, confidence_level, **kwargs)
    
    def _calculate_clopper_pearson(
        self, 
        data: np.ndarray, 
        confidence_level: float, 
        **kwargs
    ) -> MathematicalBound:
        """Calcula límites usando Clopper-Pearson (para proporciones)"""
        # Asumimos datos binarios (0,1) para proporciones
        successes = np.sum(data)
        n = len(data)
        alpha = 1 - confidence_level
        
        if successes == 0:
            lower = 0.0
        else:
            lower = stats.beta.ppf(alpha/2, successes, n - successes + 1)
        
        if successes == n:
            upper = 1.0
        else:
            upper = stats.beta.ppf(1 - alpha/2, successes + 1, n - successes)
        
        return MathematicalBound(
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            method=BoundCalculationMethod.CLOPPER_PEARSON,
            metadata={"successes": int(successes), "trials": n, "proportion": successes/n}
        )
    
    def _calculate_wilson_score(
        self, 
        data: np.ndarray, 
        confidence_level: float, 
        **kwargs
    ) -> MathematicalBound:
        """Calcula límites usando Wilson Score (para proporciones)"""
        successes = np.sum(data)
        n = len(data)
        p = successes / n
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha/2)
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return MathematicalBound(
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            method=BoundCalculationMethod.WILSON_SCORE,
            metadata={"z_score": z, "center": center, "margin": margin}
        )
    
    def _calculate_bayesian_credible(
        self, 
        data: np.ndarray, 
        confidence_level: float, 
        **kwargs
    ) -> MathematicalBound:
        """Calcula intervalo creíble bayesiano"""
        # Usando prior Beta para proporciones
        alpha_prior = kwargs.get('alpha_prior', 1)
        beta_prior = kwargs.get('beta_prior', 1)
        
        successes = np.sum(data)
        n = len(data)
        
        # Posterior Beta
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + n - successes
        
        alpha = 1 - confidence_level
        lower = stats.beta.ppf(alpha/2, alpha_post, beta_post)
        upper = stats.beta.ppf(1 - alpha/2, alpha_post, beta_post)
        
        return MathematicalBound(
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            method=BoundCalculationMethod.BAYESIAN_CREDIBLE,
            metadata={
                "alpha_prior": alpha_prior, 
                "beta_prior": beta_prior,
                "alpha_posterior": alpha_post,
                "beta_posterior": beta_post,
                "posterior_mean": alpha_post / (alpha_post + beta_post)
            }
        )
    
    def _calculate_ensemble_variance(
        self, 
        data: np.ndarray, 
        confidence_level: float, 
        **kwargs
    ) -> MathematicalBound:
        """Calcula límites basados en varianza de ensemble"""
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # t-distribution para muestra pequeña
        alpha = 1 - confidence_level
        t_val = stats.t.ppf(1 - alpha/2, n - 1)
        margin = t_val * std / np.sqrt(n)
        
        lower = mean - margin
        upper = mean + margin
        
        return MathematicalBound(
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            method=BoundCalculationMethod.ENSEMBLE_VARIANCE,
            metadata={
                "mean": mean, 
                "std": std, 
                "t_value": t_val, 
                "margin": margin,
                "degrees_freedom": n - 1
            }
        )
    
    def calculate_multiple_bounds(
        self,
        data: Union[List[float], np.ndarray],
        methods: List[BoundCalculationMethod],
        confidence_level: Optional[float] = None
    ) -> Dict[str, MathematicalBound]:
        """Calcula límites usando múltiples métodos"""
        bounds = {}
        
        for method in methods:
            try:
                bound = self.calculate_bound(data, method, confidence_level)
                bounds[method.value] = bound
            except Exception as e:
                self.logger.warning(f"Error calculando {method.value}: {str(e)}")
        
        return bounds
    
    def make_abstention_decision(
        self,
        bounds: Dict[str, MathematicalBound],
        confidence_threshold: float = 0.80,
        width_threshold: float = 0.3,
        risk_level: RiskLevel = RiskLevel.LOW
    ) -> AbstractionDecision:
        """
        Toma decisión de abstención basada en límites matemáticos
        
        Args:
            bounds: Límites calculados por diferentes métodos
            confidence_threshold: Umbral mínimo de confianza
            width_threshold: Ancho máximo aceptable del intervalo
            risk_level: Nivel de riesgo aceptable
        """
        violated_bounds = []
        justifications = []
        
        # Evaluar cada límite
        for method_name, bound in bounds.items():
            # Verificar ancho del intervalo
            if bound.width > width_threshold:
                violated_bounds.append(f"{method_name}_width")
                justifications.append(
                    f"Intervalo {method_name} muy amplio: {bound.width:.4f} > {width_threshold}"
                )
            
            # Verificar nivel de confianza
            if bound.confidence_level < confidence_threshold:
                violated_bounds.append(f"{method_name}_confidence")
                justifications.append(
                    f"Confianza {method_name} insuficiente: {bound.confidence_level:.4f} < {confidence_threshold}"
                )
        
        # Calcular consenso entre métodos
        if len(bounds) > 1:
            centers = [bound.center for bound in bounds.values()]
            center_std = np.std(centers)
            
            if center_std > 0.1:  # Alta variabilidad entre métodos
                violated_bounds.append("method_consensus")
                justifications.append(f"Alta variabilidad entre métodos: std={center_std:.4f}")
        
        # Calcular score de confianza general
        if bounds:
            confidence_scores = []
            for bound in bounds.values():
                # Score basado en confianza y ancho del intervalo
                width_penalty = min(1.0, bound.width / width_threshold)
                confidence_score = bound.confidence_level * (1 - width_penalty * 0.5)
                confidence_scores.append(confidence_score)
            
            overall_confidence = np.mean(confidence_scores)
        else:
            overall_confidence = 0.0
        
        # Decisión final
        should_abstain = len(violated_bounds) > 0 or overall_confidence < confidence_threshold
        
        # Evaluación de riesgo
        if overall_confidence >= 0.95:
            assessed_risk = RiskLevel.VERY_LOW
        elif overall_confidence >= 0.90:
            assessed_risk = RiskLevel.LOW
        elif overall_confidence >= 0.80:
            assessed_risk = RiskLevel.MEDIUM
        elif overall_confidence >= 0.70:
            assessed_risk = RiskLevel.HIGH
        else:
            assessed_risk = RiskLevel.VERY_HIGH
        
        return AbstractionDecision(
            should_abstain=should_abstain,
            confidence_score=overall_confidence,
            risk_assessment=assessed_risk,
            violated_bounds=violated_bounds,
            mathematical_justification="; ".join(justifications),
            bounds_summary=bounds
        )

class UniversalUncertaintyQuantifier:
    """Cuantificador de incertidumbre universal aplicable a cualquier dominio"""
    
    def __init__(self, framework: UniversalMathematicalFramework):
        self.framework = framework
        self.logger = logging.getLogger("UniversalUncertaintyQuantifier")
    
    def quantify_prediction_uncertainty(
        self,
        predictions: List[Any],
        ground_truth: Optional[List[Any]] = None,
        uncertainty_type: str = "epistemic"
    ) -> Dict[str, Any]:
        """
        Cuantifica incertidumbre en predicciones de cualquier tipo
        
        Args:
            predictions: Lista de predicciones (pueden ser numéricas, categóricas, etc.)
            ground_truth: Valores reales (opcional, para uncertainty aleatórica)
            uncertainty_type: 'epistemic' (model uncertainty) o 'aleatoric' (data uncertainty)
        """
        uncertainty_metrics = {}
        
        try:
            if self._are_numeric_predictions(predictions):
                uncertainty_metrics.update(
                    self._quantify_numeric_uncertainty(predictions, ground_truth, uncertainty_type)
                )
            elif self._are_categorical_predictions(predictions):
                uncertainty_metrics.update(
                    self._quantify_categorical_uncertainty(predictions, ground_truth, uncertainty_type)
                )
            else:
                uncertainty_metrics.update(
                    self._quantify_generic_uncertainty(predictions, ground_truth, uncertainty_type)
                )
            
            # Metadatos generales
            uncertainty_metrics.update({
                "sample_size": len(predictions),
                "uncertainty_type": uncertainty_type,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error cuantificando incertidumbre: {str(e)}")
            uncertainty_metrics["error"] = str(e)
        
        return uncertainty_metrics
    
    def _are_numeric_predictions(self, predictions: List[Any]) -> bool:
        """Verifica si las predicciones son numéricas"""
        try:
            [float(p) for p in predictions[:5]]  # Test con primeros 5
            return True
        except (ValueError, TypeError):
            return False
    
    def _are_categorical_predictions(self, predictions: List[Any]) -> bool:
        """Verifica si las predicciones son categóricas"""
        return isinstance(predictions[0], (str, int)) and len(set(predictions)) < len(predictions) * 0.5
    
    def _quantify_numeric_uncertainty(
        self, 
        predictions: List[float], 
        ground_truth: Optional[List[float]], 
        uncertainty_type: str
    ) -> Dict[str, Any]:
        """Cuantifica incertidumbre para predicciones numéricas"""
        numeric_preds = np.array([float(p) for p in predictions])
        
        metrics = {
            "prediction_variance": np.var(numeric_preds),
            "prediction_std": np.std(numeric_preds),
            "prediction_range": np.max(numeric_preds) - np.min(numeric_preds),
            "coefficient_variation": np.std(numeric_preds) / np.mean(numeric_preds) if np.mean(numeric_preds) != 0 else np.inf
        }
        
        # Calcular múltiples límites de confianza
        methods = [
            BoundCalculationMethod.BOOTSTRAP_PERCENTILE,
            BoundCalculationMethod.ENSEMBLE_VARIANCE
        ]
        
        bounds = self.framework.calculate_multiple_bounds(numeric_preds, methods)
        metrics["confidence_bounds"] = {k: v.to_dict() for k, v in bounds.items()}
        
        # Si hay ground truth, calcular uncertainty aleatórica
        if ground_truth is not None and uncertainty_type == "aleatoric":
            gt_array = np.array([float(gt) for gt in ground_truth])
            if len(gt_array) == len(numeric_preds):
                residuals = numeric_preds - gt_array
                metrics.update({
                    "residual_variance": np.var(residuals),
                    "residual_std": np.std(residuals),
                    "mae": np.mean(np.abs(residuals)),
                    "rmse": np.sqrt(np.mean(residuals**2))
                })
        
        return metrics
    
    def _quantify_categorical_uncertainty(
        self, 
        predictions: List[Any], 
        ground_truth: Optional[List[Any]], 
        uncertainty_type: str
    ) -> Dict[str, Any]:
        """Cuantifica incertidumbre para predicciones categóricas"""
        from collections import Counter
        
        pred_counts = Counter(predictions)
        total_preds = len(predictions)
        
        # Entropy como medida de incertidumbre
        probabilities = np.array(list(pred_counts.values())) / total_preds
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Gini impurity
        gini = 1 - np.sum(probabilities**2)
        
        metrics = {
            "entropy": entropy,
            "gini_impurity": gini,
            "num_unique_predictions": len(pred_counts),
            "prediction_distribution": dict(pred_counts),
            "max_probability": np.max(probabilities),
            "prediction_confidence": 1 - entropy / np.log2(len(pred_counts)) if len(pred_counts) > 1 else 1.0
        }
        
        # Si hay ground truth, calcular métricas de precisión
        if ground_truth is not None:
            correct_predictions = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
            accuracy = correct_predictions / len(predictions)
            metrics.update({
                "accuracy": accuracy,
                "error_rate": 1 - accuracy,
                "correct_predictions": correct_predictions
            })
        
        return metrics
    
    def _quantify_generic_uncertainty(
        self, 
        predictions: List[Any], 
        ground_truth: Optional[List[Any]], 
        uncertainty_type: str
    ) -> Dict[str, Any]:
        """Cuantifica incertidumbre para predicciones genéricas"""
        from collections import Counter
        
        # Análisis básico de diversidad
        unique_predictions = len(set([str(p) for p in predictions]))
        total_predictions = len(predictions)
        
        # Diversidad normalizada
        diversity = unique_predictions / total_predictions
        
        # Análisis de frecuencias
        str_predictions = [str(p) for p in predictions]
        freq_counter = Counter(str_predictions)
        most_common_freq = freq_counter.most_common(1)[0][1] / total_predictions
        
        metrics = {
            "diversity": diversity,
            "unique_predictions": unique_predictions,
            "total_predictions": total_predictions,
            "most_common_frequency": most_common_freq,
            "prediction_consistency": 1 - diversity,  # Inverso de diversidad
            "frequency_distribution": dict(freq_counter)
        }
        
        # Comparación con ground truth si disponible
        if ground_truth is not None:
            str_gt = [str(gt) for gt in ground_truth]
            exact_matches = sum(1 for p, gt in zip(str_predictions, str_gt) if p == gt)
            match_rate = exact_matches / len(predictions)
            metrics.update({
                "exact_match_rate": match_rate,
                "exact_matches": exact_matches
            })
        
        return metrics

# Instancia global del framework matemático
universal_math_framework = UniversalMathematicalFramework()
universal_uncertainty_quantifier = UniversalUncertaintyQuantifier(universal_math_framework)