"""
Ejemplo de implementación del Universal Analysis Framework para Análisis Financiero
Demuestra cómo los 8 meta-principios universales se aplican al análisis de inversiones y riesgos financieros.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import random

# Imports del framework universal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.universal_framework import UniversalAnalyzer, UniversalResult, universal_registry
from mathematical.abstention_framework import (
    BoundCalculationMethod, RiskLevel, universal_math_framework
)
from ensemble.multi_model_evaluator import (
    UniversalModel, ModelType, universal_ensemble_evaluator
)
from genealogical.influence_tracker import (
    UniversalInfluenceTracker, NodeType, InfluenceType
)
from hybridization.adaptive_hybridizer import (
    HybridizationContext, ComponentType, universal_adaptive_hybridizer
)

@dataclass
class FinancialData:
    """Datos financieros de entrada"""
    symbol: str
    prices: List[float]  # Precios históricos
    volumes: List[float]  # Volúmenes de trading
    market_cap: float
    pe_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    
class FinancialAnalysisResult:
    """Resultado específico del análisis financiero"""
    def __init__(self):
        self.investment_recommendation: str = "HOLD"  # BUY, SELL, HOLD
        self.risk_score: float = 0.5  # 0-1, donde 1 es muy riesgo
        self.expected_return: float = 0.0  # Retorno esperado anualizado
        self.volatility: float = 0.0  # Volatilidad calculada
        self.technical_indicators: Dict[str, float] = {}
        self.fundamental_metrics: Dict[str, float] = {}
        self.risk_metrics: Dict[str, float] = {}
        self.price_targets: Dict[str, float] = {}  # optimistic, realistic, pessimistic

class FinancialAnalyzer(UniversalAnalyzer[FinancialData, FinancialAnalysisResult]):
    """Analizador financiero que implementa los 8 meta-principios universales"""
    
    def __init__(self):
        super().__init__(
            domain="financial_analysis",
            confidence_threshold=0.80,  # Mayor umbral para decisiones financieras
            enable_abstention=True,
            bootstrap_iterations=1000,  # Más iteraciones para análisis financiero
            ensemble_models=["technical_model", "fundamental_model", "risk_model", "sentiment_model"]
        )
        
        # Configurar influence tracker
        self.influence_tracker = UniversalInfluenceTracker("financial_analysis")
        
        # Configurar componentes de hibridización
        self._setup_financial_components()
    
    def _setup_financial_components(self):
        """Configura componentes específicos para análisis financiero"""
        
        # Componente de análisis técnico
        def technical_analysis(data: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
            prices = data.get("prices", [])
            volumes = data.get("volumes", [])
            
            if len(prices) < 20:
                return {}, 0.3  # Baja confianza con pocos datos
            
            # Calcular indicadores técnicos
            sma_20 = np.mean(prices[-20:])  # Media móvil simple 20 períodos
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
            
            current_price = prices[-1]
            price_change = (current_price - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            
            # RSI simplificado
            gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
            losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
            
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains) if gains else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses) if losses else 1e-6
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Volatilidad
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            volatility = np.std(returns) * np.sqrt(252)  # Anualizada
            
            indicators = {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "price_vs_sma20": (current_price - sma_20) / sma_20,
                "rsi": rsi,
                "volatility": volatility,
                "price_momentum": price_change,
                "volume_trend": np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
            }
            
            # Confianza basada en cantidad de datos y consistencia
            confidence = min(1.0, len(prices) / 100) * 0.85  # Base de confianza técnica
            
            return indicators, confidence
        
        universal_adaptive_hybridizer.add_function_component(
            component_id="technical_analyzer",
            component_type=ComponentType.ALGORITHM,
            function=technical_analysis,
            performance_metrics={"accuracy": 0.68, "sharpe_ratio": 1.2},
            context_suitability={"financial_analysis": 0.95, "technical_trading": 0.98},
            computational_cost=1.5,
            reliability_score=0.75
        )
        
        # Componente de análisis fundamental
        def fundamental_analysis(data: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
            pe_ratio = data.get("pe_ratio")
            debt_to_equity = data.get("debt_to_equity")
            roe = data.get("roe")
            revenue_growth = data.get("revenue_growth")
            profit_margin = data.get("profit_margin")
            
            fundamentals = {}
            available_metrics = 0
            
            # Evaluar P/E ratio
            if pe_ratio is not None:
                if pe_ratio < 15:
                    pe_score = 0.8  # Valoración atractiva
                elif pe_ratio < 25:
                    pe_score = 0.6  # Valoración razonable
                else:
                    pe_score = 0.3  # Posible sobrevaloración
                
                fundamentals["pe_evaluation"] = pe_score
                fundamentals["pe_ratio"] = pe_ratio
                available_metrics += 1
            
            # Evaluar endeudamiento
            if debt_to_equity is not None:
                debt_score = max(0.0, 1.0 - debt_to_equity / 2.0)  # Penalizar alto endeudamiento
                fundamentals["debt_score"] = debt_score
                fundamentals["debt_to_equity"] = debt_to_equity
                available_metrics += 1
            
            # Evaluar rentabilidad
            if roe is not None:
                roe_score = min(1.0, roe / 0.20)  # ROE de 20% como referencia excelente
                fundamentals["roe_score"] = roe_score
                fundamentals["roe"] = roe
                available_metrics += 1
            
            # Evaluar crecimiento
            if revenue_growth is not None:
                growth_score = min(1.0, max(0.0, revenue_growth / 0.15))  # 15% como crecimiento excelente
                fundamentals["growth_score"] = growth_score
                fundamentals["revenue_growth"] = revenue_growth
                available_metrics += 1
            
            # Evaluar márgenes
            if profit_margin is not None:
                margin_score = min(1.0, profit_margin / 0.20)  # 20% como margen excelente
                fundamentals["margin_score"] = margin_score
                fundamentals["profit_margin"] = profit_margin
                available_metrics += 1
            
            # Score fundamental general
            if available_metrics > 0:
                score_components = [v for k, v in fundamentals.items() if k.endswith("_score")]
                fundamentals["overall_fundamental_score"] = np.mean(score_components) if score_components else 0.5
            
            # Confianza basada en disponibilidad de métricas
            confidence = (available_metrics / 5.0) * 0.9  # Máximo 90% si tenemos todas las métricas
            
            return fundamentals, confidence
        
        universal_adaptive_hybridizer.add_function_component(
            component_id="fundamental_analyzer",
            component_type=ComponentType.ALGORITHM,
            function=fundamental_analysis,
            performance_metrics={"accuracy": 0.72, "information_ratio": 0.85},
            context_suitability={"financial_analysis": 0.90, "value_investing": 0.95},
            computational_cost=1.0,
            reliability_score=0.85
        )
        
        # Componente de análisis de riesgo
        def risk_analysis(data: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
            prices = data.get("prices", [])
            market_cap = data.get("market_cap", 0)
            
            if len(prices) < 10:
                return {"risk_score": 0.8}, 0.4  # Alto riesgo por falta de datos
            
            # Calcular métricas de riesgo
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            
            # Volatilidad (riesgo principal)
            volatility = np.std(returns) * np.sqrt(252)
            
            # Value at Risk (VaR) simplificado
            var_95 = np.percentile(returns, 5)  # VaR al 95%
            
            # Maximum Drawdown
            peak = prices[0]
            max_drawdown = 0
            for price in prices:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Riesgo de liquidez basado en market cap
            if market_cap > 10e9:  # > 10B
                liquidity_risk = 0.1
            elif market_cap > 1e9:  # 1B - 10B
                liquidity_risk = 0.3
            else:
                liquidity_risk = 0.6
            
            # Score de riesgo compuesto
            volatility_risk = min(1.0, volatility / 0.5)  # Normalizar volatilidad
            var_risk = min(1.0, abs(var_95) * 10)  # VaR risk
            drawdown_risk = min(1.0, max_drawdown * 2)
            
            overall_risk = np.mean([volatility_risk, var_risk, drawdown_risk, liquidity_risk])
            
            risk_metrics = {
                "volatility": volatility,
                "var_95": var_95,
                "max_drawdown": max_drawdown,
                "liquidity_risk": liquidity_risk,
                "overall_risk_score": overall_risk,
                "volatility_risk": volatility_risk,
                "var_risk": var_risk,
                "drawdown_risk": drawdown_risk
            }
            
            # Alta confianza en métricas de riesgo (son objetivas)
            confidence = 0.9 if len(prices) >= 50 else 0.7
            
            return risk_metrics, confidence
        
        universal_adaptive_hybridizer.add_function_component(
            component_id="risk_analyzer",
            component_type=ComponentType.ALGORITHM,
            function=risk_analysis,
            performance_metrics={"accuracy": 0.85, "precision": 0.82},
            context_suitability={"financial_analysis": 0.92, "risk_management": 0.98},
            computational_cost=1.2,
            reliability_score=0.90
        )
    
    def preprocess_input(self, input_data: FinancialData) -> Dict[str, Any]:
        """Preprocesa los datos financieros"""
        # Validaciones básicas
        if not input_data.prices or len(input_data.prices) < 5:
            raise ValueError("Se requieren al menos 5 puntos de precio histórico")
        
        if not input_data.volumes:
            input_data.volumes = [1.0] * len(input_data.prices)  # Volumen por defecto
        
        # Normalización y limpieza
        processed_data = {
            "symbol": input_data.symbol,
            "prices": input_data.prices,
            "volumes": input_data.volumes,
            "market_cap": input_data.market_cap,
            "pe_ratio": input_data.pe_ratio,
            "debt_to_equity": input_data.debt_to_equity,
            "roe": input_data.roe,
            "revenue_growth": input_data.revenue_growth,
            "profit_margin": input_data.profit_margin,
            "data_points": len(input_data.prices),
            "price_range": max(input_data.prices) - min(input_data.prices),
            "current_price": input_data.prices[-1]
        }
        
        # Rastrear preprocesamiento
        _, process_id, output_id = self.influence_tracker.track_processing_step(
            "financial_data_preprocessing",
            input_data,
            processed_data,
            "data_validation_and_normalization"
        )
        
        processed_data["preprocessing_node_id"] = output_id
        return processed_data
    
    def extract_features(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae características financieras relevantes"""
        prices = preprocessed_data["prices"]
        volumes = preprocessed_data["volumes"]
        
        # Características de precios
        current_price = prices[-1]
        price_features = {
            "current_price": current_price,
            "price_change_1d": (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0,
            "price_change_5d": (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0,
            "price_change_20d": (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0,
            "price_min_max_ratio": min(prices) / max(prices),
            "price_trend": np.polyfit(range(len(prices)), prices, 1)[0] if len(prices) > 2 else 0
        }
        
        # Características de volumen
        volume_features = {
            "avg_volume": np.mean(volumes),
            "volume_volatility": np.std(volumes) / max(np.mean(volumes), 1),
            "recent_volume_trend": np.mean(volumes[-5:]) / np.mean(volumes) if len(volumes) > 5 else 1.0
        }
        
        # Características fundamentales
        fundamental_features = {
            "has_pe": preprocessed_data["pe_ratio"] is not None,
            "has_debt_ratio": preprocessed_data["debt_to_equity"] is not None,
            "has_roe": preprocessed_data["roe"] is not None,
            "has_growth": preprocessed_data["revenue_growth"] is not None,
            "has_margin": preprocessed_data["profit_margin"] is not None,
            "fundamental_completeness": sum([
                preprocessed_data["pe_ratio"] is not None,
                preprocessed_data["debt_to_equity"] is not None,
                preprocessed_data["roe"] is not None,
                preprocessed_data["revenue_growth"] is not None,
                preprocessed_data["profit_margin"] is not None
            ]) / 5.0
        }
        
        # Características de mercado
        market_features = {
            "market_cap": preprocessed_data["market_cap"],
            "market_cap_category": (
                "large" if preprocessed_data["market_cap"] > 10e9 else
                "mid" if preprocessed_data["market_cap"] > 1e9 else "small"
            ),
            "data_quality": min(1.0, len(prices) / 100)  # Calidad basada en cantidad de datos
        }
        
        # Combinar todas las características
        features = {
            **price_features,
            **volume_features,
            **fundamental_features,
            **market_features,
            **preprocessed_data  # Incluir datos originales
        }
        
        # Rastrear extracción de características
        _, process_id, output_id = self.influence_tracker.track_processing_step(
            "financial_feature_extraction",
            preprocessed_data,
            features,
            "technical_and_fundamental_feature_engineering"
        )
        
        features["feature_extraction_node_id"] = output_id
        return features
    
    def perform_core_analysis(self, features: Dict[str, Any]) -> FinancialAnalysisResult:
        """Realiza análisis financiero principal usando hibridización"""
        
        result = FinancialAnalysisResult()
        
        # Configurar contexto de hibridización
        context = HybridizationContext(
            domain="financial_analysis",
            data_characteristics={
                "data_points": features["data_points"],
                "market_cap": features["market_cap"],
                "fundamental_completeness": features["fundamental_completeness"]
            },
            performance_requirements={
                "accuracy": 0.80,
                "risk_adjusted_return": 1.5
            },
            quality_requirements={
                "confidence": 0.85,
                "robustness": 0.80
            }
        )
        
        # Hibridización para análisis completo
        hybridization_result = universal_adaptive_hybridizer.hybridize(context)
        
        # Inicializar métricas
        technical_scores = {}
        fundamental_scores = {}
        risk_scores = {}
        
        # Ejecutar análisis según componentes seleccionados
        if "technical_analyzer" in hybridization_result.selected_components:
            # Análisis técnico
            prices = features["prices"]
            
            # Indicadores técnicos simplificados
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
            current_price = features["current_price"]
            
            rsi = 50 + features["price_change_1d"] * 100  # RSI simplificado
            rsi = max(0, min(100, rsi))
            
            technical_scores = {
                "sma_20": sma_20,
                "price_vs_sma": (current_price - sma_20) / sma_20,
                "rsi": rsi,
                "momentum": features["price_trend"],
                "volatility": np.std([(prices[i]/prices[i-1]-1) for i in range(1, len(prices))]) * np.sqrt(252)
            }
            
            result.technical_indicators = technical_scores
            result.volatility = technical_scores["volatility"]
        
        if "fundamental_analyzer" in hybridization_result.selected_components:
            # Análisis fundamental
            if features["pe_ratio"]:
                pe_score = 1.0 - min(1.0, features["pe_ratio"] / 30.0)  # Normalizar P/E
                fundamental_scores["pe_score"] = pe_score
            
            if features["roe"]:
                roe_score = min(1.0, features["roe"] / 0.20)  # 20% ROE como excelente
                fundamental_scores["roe_score"] = roe_score
            
            if features["debt_to_equity"]:
                debt_score = max(0.0, 1.0 - features["debt_to_equity"] / 2.0)
                fundamental_scores["debt_score"] = debt_score
            
            result.fundamental_metrics = fundamental_scores
        
        if "risk_analyzer" in hybridization_result.selected_components:
            # Análisis de riesgo
            prices = features["prices"]
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            
            volatility = np.std(returns) * np.sqrt(252)
            var_95 = np.percentile(returns, 5) if returns else -0.05
            
            # Riesgo basado en capitalización
            if features["market_cap"] > 10e9:
                liquidity_risk = 0.1
            elif features["market_cap"] > 1e9:
                liquidity_risk = 0.3
            else:
                liquidity_risk = 0.6
            
            risk_scores = {
                "volatility": volatility,
                "var_95": var_95,
                "liquidity_risk": liquidity_risk,
                "overall_risk": np.mean([volatility / 0.5, abs(var_95) * 10, liquidity_risk])
            }
            
            result.risk_metrics = risk_scores
            result.risk_score = risk_scores["overall_risk"]
        
        # Generar recomendación basada en análisis combinado
        recommendation_score = 0.0
        factors = 0
        
        # Factor técnico
        if technical_scores:
            tech_signal = 0.0
            if technical_scores.get("price_vs_sma", 0) > 0.05:  # Precio 5% sobre SMA
                tech_signal += 0.3
            if technical_scores.get("rsi", 50) < 30:  # Oversold
                tech_signal += 0.4
            elif technical_scores.get("rsi", 50) > 70:  # Overbought
                tech_signal -= 0.4
            if technical_scores.get("momentum", 0) > 0:
                tech_signal += 0.2
            
            recommendation_score += tech_signal
            factors += 1
        
        # Factor fundamental
        if fundamental_scores:
            fund_signal = np.mean(list(fundamental_scores.values())) if fundamental_scores else 0.5
            recommendation_score += fund_signal - 0.5  # Centrar en 0
            factors += 1
        
        # Factor de riesgo (penaliza alto riesgo)
        if risk_scores:
            risk_penalty = -risk_scores.get("overall_risk", 0.5) + 0.5
            recommendation_score += risk_penalty * 0.5
            factors += 1
        
        # Normalizar recomendación
        if factors > 0:
            recommendation_score /= factors
        
        # Determinar recomendación
        if recommendation_score > 0.2:
            result.investment_recommendation = "BUY"
        elif recommendation_score < -0.2:
            result.investment_recommendation = "SELL"
        else:
            result.investment_recommendation = "HOLD"
        
        # Calcular retorno esperado
        result.expected_return = max(-0.50, min(0.50, recommendation_score * 0.3))
        
        # Objetivos de precio
        current_price = features["current_price"]
        result.price_targets = {
            "pessimistic": current_price * (1 + result.expected_return - 0.1),
            "realistic": current_price * (1 + result.expected_return),
            "optimistic": current_price * (1 + result.expected_return + 0.1)
        }
        
        # Rastrear análisis principal
        analysis_node = self.influence_tracker.add_node(
            "financial_core_analysis",
            NodeType.PROCESSING_STEP,
            "comprehensive_financial_analysis",
            processing_stage="core_analysis"
        )
        
        result_node = self.influence_tracker.add_node(
            "financial_analysis_result",
            NodeType.FINAL_RESULT,
            result,
            processing_stage="investment_recommendation"
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
            0.92
        )
        
        return result
    
    def calculate_confidence_metrics(self, result: FinancialAnalysisResult, features: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de confianza específicas del análisis financiero"""
        metrics = {}
        
        # Confianza en datos (cantidad y calidad)
        data_confidence = min(1.0, features["data_points"] / 50)  # 50+ puntos para confianza máxima
        metrics["data_confidence"] = data_confidence
        
        # Confianza en análisis técnico
        if result.technical_indicators:
            volatility = result.technical_indicators.get("volatility", 0.5)
            # Menor volatilidad = mayor confianza en análisis técnico
            tech_confidence = max(0.3, 1.0 - volatility)
            metrics["technical_confidence"] = tech_confidence
        
        # Confianza en análisis fundamental
        fundamental_completeness = features.get("fundamental_completeness", 0.0)
        metrics["fundamental_confidence"] = fundamental_completeness * 0.9
        
        # Confianza en evaluación de riesgo
        risk_confidence = 0.85 if len(features["prices"]) >= 30 else 0.65
        metrics["risk_confidence"] = risk_confidence
        
        # Confianza en recomendación basada en market cap (más datos para empresas grandes)
        if features["market_cap"] > 10e9:
            recommendation_confidence = 0.8
        elif features["market_cap"] > 1e9:
            recommendation_confidence = 0.7
        else:
            recommendation_confidence = 0.6
        
        metrics["recommendation_confidence"] = recommendation_confidence
        
        # Confianza general considerando volatilidad del mercado
        market_stability = 1.0 - min(1.0, result.volatility / 0.5) if result.volatility else 0.8
        metrics["market_stability_confidence"] = market_stability
        
        # Score de confianza en price targets
        price_spread = (result.price_targets.get("optimistic", 0) - 
                       result.price_targets.get("pessimistic", 0)) / max(features["current_price"], 1)
        target_confidence = max(0.4, 1.0 - price_spread)
        metrics["price_target_confidence"] = target_confidence
        
        return metrics
    
    def perform_genealogical_analysis(self, input_data: FinancialData, result: FinancialAnalysisResult) -> Dict[str, Any]:
        """Analiza las influencias genealógicas del análisis financiero"""
        
        # Añadir fuentes de datos externas
        market_data_node = self.influence_tracker.add_node(
            "market_data_source",
            NodeType.EXTERNAL_SOURCE,
            f"Financial data for {input_data.symbol}",
            importance=1.0
        )
        
        # Añadir nodos de diferentes tipos de análisis como fuentes metodológicas
        if result.technical_indicators:
            technical_source = self.influence_tracker.add_node(
                "technical_analysis_methodology",
                NodeType.EXTERNAL_SOURCE,
                "Technical analysis indicators (SMA, RSI, momentum)",
                importance=0.8
            )
        
        if result.fundamental_metrics:
            fundamental_source = self.influence_tracker.add_node(
                "fundamental_analysis_methodology", 
                NodeType.EXTERNAL_SOURCE,
                "Fundamental analysis metrics (P/E, ROE, debt ratios)",
                importance=0.9
            )
        
        if result.risk_metrics:
            risk_source = self.influence_tracker.add_node(
                "risk_analysis_methodology",
                NodeType.EXTERNAL_SOURCE,
                "Risk assessment methods (VaR, volatility, drawdown)",
                importance=0.95  # Alta importancia para decisiones financieras
            )
        
        # Realizar análisis genealógico completo
        genealogy_analysis = self.influence_tracker.analyze_genealogy()
        
        # Encontrar influencias críticas (umbral más alto para finanzas)
        critical_influences = self.influence_tracker.find_critical_influences(importance_threshold=0.85)
        
        return {
            "genealogy_summary": {
                "total_nodes": len(genealogy_analysis.nodes),
                "total_relations": len(genealogy_analysis.relations),
                "critical_influences_count": len(critical_influences),
                "methodology_nodes": len([n for n in genealogy_analysis.nodes.values() 
                                        if n.node_type == NodeType.EXTERNAL_SOURCE])
            },
            "influence_metrics": genealogy_analysis.influence_metrics,
            "critical_influences": critical_influences[:3],  # Top 3 más críticas
            "data_lineage": genealogy_analysis.ancestry_paths,
            "methodology_impact": {
                node_id: metrics.get("pagerank", 0.0) 
                for node_id, metrics in genealogy_analysis.centrality_metrics.items()
                if genealogy_analysis.nodes[node_id].node_type == NodeType.EXTERNAL_SOURCE
            }
        }
    
    def _evaluate_with_model(self, model_name: str, features: Dict[str, Any], core_result: FinancialAnalysisResult) -> Dict[str, Any]:
        """Evaluaciones ensemble específicas para análisis financiero"""
        
        if model_name == "technical_model":
            # Modelo técnico alternativo
            prices = features["prices"]
            
            # Bandas de Bollinger simplificadas
            sma = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
            std = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
            
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std
            current_price = features["current_price"]
            
            # Señal basada en bandas
            if current_price < lower_band:
                signal = "BUY"
                confidence = 0.75
            elif current_price > upper_band:
                signal = "SELL"
                confidence = 0.75
            else:
                signal = "HOLD"
                confidence = 0.60
            
            return {
                "technical_signal": signal,
                "bollinger_position": (current_price - sma) / (upper_band - lower_band),
                "confidence": confidence
            }
        
        elif model_name == "fundamental_model":
            # Modelo fundamental alternativo
            pe_ratio = features.get("pe_ratio")
            roe = features.get("roe") 
            
            if pe_ratio and roe:
                # PEG ratio simplificado (P/E / Growth)
                growth_proxy = roe  # Usar ROE como proxy de crecimiento
                peg = pe_ratio / (growth_proxy * 100) if growth_proxy > 0 else float('inf')
                
                if peg < 1.0:
                    signal = "BUY"
                    confidence = 0.80
                elif peg > 2.0:
                    signal = "SELL"  
                    confidence = 0.70
                else:
                    signal = "HOLD"
                    confidence = 0.60
                
                return {
                    "fundamental_signal": signal,
                    "peg_ratio": peg,
                    "confidence": confidence
                }
            
            return {"fundamental_signal": "HOLD", "confidence": 0.30}
        
        elif model_name == "risk_model":
            # Modelo de riesgo alternativo
            risk_score = core_result.risk_score
            expected_return = abs(core_result.expected_return)
            
            # Ratio Sharpe simplificado
            sharpe_proxy = expected_return / max(risk_score, 0.01)
            
            if sharpe_proxy > 1.5:
                risk_signal = "LOW_RISK"
                confidence = 0.85
            elif sharpe_proxy > 0.8:
                risk_signal = "MEDIUM_RISK"
                confidence = 0.75
            else:
                risk_signal = "HIGH_RISK"
                confidence = 0.70
            
            return {
                "risk_assessment": risk_signal,
                "sharpe_proxy": sharpe_proxy,
                "confidence": confidence
            }
        
        elif model_name == "sentiment_model":
            # Modelo de sentimiento de mercado (simulado)
            
            # Simular sentimiento basado en momentum y volatilidad
            momentum = features.get("price_trend", 0)
            volatility = core_result.volatility
            
            if momentum > 0 and volatility < 0.3:
                sentiment = "POSITIVE"
                confidence = 0.70
            elif momentum < 0 and volatility > 0.4:
                sentiment = "NEGATIVE" 
                confidence = 0.65
            else:
                sentiment = "NEUTRAL"
                confidence = 0.55
            
            return {
                "market_sentiment": sentiment,
                "sentiment_strength": abs(momentum) / max(volatility, 0.01),
                "confidence": confidence
            }
        
        else:
            return {"model_evaluation": f"Unknown financial model {model_name}"}

# Función de demostración
def demonstrate_financial_analysis():
    """Demuestra el uso del analizador financiero universal"""
    
    # Crear analizador
    analyzer = FinancialAnalyzer()
    
    # Registrar en registry universal
    universal_registry.register_analyzer("financial_analysis", analyzer)
    
    # Datos financieros de ejemplo (simulados)
    np.random.seed(42)  # Para reproducibilidad
    
    # Simular precios históricos con tendencia y volatilidad
    days = 100
    initial_price = 100.0
    prices = [initial_price]
    
    for i in range(1, days):
        # Simulación de random walk con drift
        daily_return = np.random.normal(0.0005, 0.02)  # 0.05% drift, 2% volatilidad diaria
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Simular volúmenes
    volumes = [np.random.normal(1000000, 200000) for _ in range(days)]
    volumes = [max(100000, v) for v in volumes]  # Mínimo volumen
    
    sample_data = FinancialData(
        symbol="SAMPLE",
        prices=prices,
        volumes=volumes,
        market_cap=5.2e9,  # 5.2B market cap
        pe_ratio=18.5,
        debt_to_equity=0.4,
        roe=0.15,  # 15% ROE
        revenue_growth=0.08,  # 8% growth
        profit_margin=0.12  # 12% margin
    )
    
    print("💰 DEMOSTRACIÓN: Análisis Financiero Universal")
    print("=" * 60)
    
    # Realizar análisis
    result = analyzer.analyze(sample_data)
    
    print(f"📈 Resultado del Análisis Financiero:")
    print(f"   • Símbolo: {sample_data.symbol}")
    print(f"   • Precio Actual: ${sample_data.prices[-1]:.2f}")
    print(f"   • Confianza General: {result.confidence:.3f}")
    print(f"   • Se Abstuvo: {'Sí' if result.abstained else 'No'}")
    
    if not result.abstained and result.result:
        fin_result = result.result
        print(f"   • Recomendación: {fin_result.investment_recommendation}")
        print(f"   • Retorno Esperado: {fin_result.expected_return:.1%}")
        print(f"   • Score de Riesgo: {fin_result.risk_score:.3f}")
        print(f"   • Volatilidad Anualizada: {fin_result.volatility:.1%}")
        
        print(f"\n🎯 Objetivos de Precio:")
        for target_type, price in fin_result.price_targets.items():
            print(f"   • {target_type.title()}: ${price:.2f}")
    
    print(f"\n📊 Métricas de Confianza:")
    for metric, value in result.metadata.confidence_metrics.items():
        print(f"   • {metric}: {value:.3f}")
    
    print(f"\n🔗 Análisis Genealógico:")
    genealogy_data = result.metadata.genealogy_data
    if genealogy_data:
        summary = genealogy_data.get("genealogy_summary", {})
        print(f"   • Nodos de Análisis: {summary.get('total_nodes', 0)}")
        print(f"   • Relaciones de Influencia: {summary.get('total_relations', 0)}")
        print(f"   • Metodologías Aplicadas: {summary.get('methodology_nodes', 0)}")
        print(f"   • Influencias Críticas: {summary.get('critical_influences_count', 0)}")
    
    print(f"\n🎭 Resultados Ensemble:")
    for i, ensemble_result in enumerate(result.metadata.ensemble_results):
        model_name = ensemble_result.get("model", "Unknown")
        status = "✓" if not ensemble_result.get("error") else "✗"
        print(f"   • {model_name}: {status}")
    
    # Mostrar límites de incertidumbre usando framework matemático
    print(f"\n🔢 Análisis Matemático de Incertidumbre:")
    if not result.abstained:
        confidence_values = list(result.metadata.confidence_metrics.values())
        
        # Calcular límites usando múltiples métodos
        bounds = universal_math_framework.calculate_multiple_bounds(
            confidence_values,
            [BoundCalculationMethod.BOOTSTRAP_PERCENTILE, 
             BoundCalculationMethod.HOEFFDING,
             BoundCalculationMethod.ENSEMBLE_VARIANCE]
        )
        
        for method_name, bound in bounds.items():
            print(f"   • {method_name}: [{bound.lower_bound:.3f}, {bound.upper_bound:.3f}] "
                  f"(width: {bound.width:.3f})")
        
        # Decisión de abstención matemática
        abstention_decision = universal_math_framework.make_abstention_decision(
            bounds, confidence_threshold=0.75, width_threshold=0.3
        )
        
        print(f"\n⚖️  Evaluación de Abstención Matemática:")
        print(f"   • Debería Abstenerse: {'Sí' if abstention_decision.should_abstain else 'No'}")
        print(f"   • Score de Confianza: {abstention_decision.confidence_score:.3f}")
        print(f"   • Nivel de Riesgo: {abstention_decision.risk_assessment.value}")
        
        if abstention_decision.mathematical_justification:
            print(f"   • Justificación: {abstention_decision.mathematical_justification}")
    
    if result.abstained:
        print(f"\n⚠️  Razones de Abstención:")
        for reason in result.metadata.abstention_reasons:
            print(f"   • {reason}")
    
    return result

# Modelos específicos para ensemble financiero
class TechnicalAnalysisModel(UniversalModel):
    """Modelo especializado en análisis técnico"""
    
    def __init__(self):
        super().__init__("advanced_technical", ModelType.STATISTICAL)
    
    def predict(self, input_data: Any) -> Tuple[Any, float]:
        if hasattr(input_data, 'prices'):
            prices = input_data.prices
        else:
            prices = input_data.get("prices", [])
        
        if len(prices) < 20:
            return {"signal": "HOLD", "reason": "insufficient_data"}, 0.3
        
        # Análisis técnico más avanzado
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices[-20:])
        sma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else sma_50
        
        current_price = prices[-1]
        
        # Golden Cross / Death Cross
        if sma_50 > sma_200 * 1.02:  # Golden cross + 2% buffer
            signal = "BUY"
            confidence = 0.8
            reason = "golden_cross_pattern"
        elif sma_50 < sma_200 * 0.98:  # Death cross - 2% buffer
            signal = "SELL"
            confidence = 0.75
            reason = "death_cross_pattern"
        else:
            signal = "HOLD"
            confidence = 0.6
            reason = "neutral_moving_averages"
        
        # Ajustar confianza por volatilidad
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        volatility = np.std(returns[-30:]) if len(returns) >= 30 else np.std(returns)
        
        # Menor confianza en mercados muy volátiles
        volatility_adjustment = max(0.5, 1.0 - volatility * 5)
        confidence *= volatility_adjustment
        
        return {
            "signal": signal,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "current_price": current_price,
            "volatility": volatility,
            "reason": reason
        }, confidence
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_type": "advanced_technical_analysis",
            "indicators": ["SMA_50", "SMA_200", "volatility", "moving_average_crossovers"],
            "timeframe": "multi_period",
            "accuracy": 0.72
        }

if __name__ == "__main__":
    # Ejecutar demostración
    result = demonstrate_financial_analysis()
    
    # Ejemplo adicional con modelo técnico específico
    print("\n" + "="*60)
    print("📊 EJEMPLO ENSEMBLE: Modelo Técnico Avanzado")
    
    # Crear datos de prueba simples
    simple_data = {
        "prices": [100 + i + np.sin(i/10) * 5 for i in range(100)]  # Tendencia con oscilación
    }
    
    # Añadir modelo técnico al ensemble
    tech_model = TechnicalAnalysisModel()
    universal_ensemble_evaluator.add_model(tech_model)
    
    # Evaluar con ensemble
    ensemble_result = universal_ensemble_evaluator.evaluate(simple_data)
    
    print(f"📈 Resultado Ensemble Técnico:")
    if ensemble_result.final_result:
        print(f"   • Señal Final: {ensemble_result.final_result}")
    print(f"   • Confianza: {ensemble_result.overall_confidence:.3f}")
    print(f"   • Estrategia: {ensemble_result.strategy_used.value}")
    print(f"   • Modelos Exitosos: {ensemble_result.to_dict()['successful_models']}")
    
    print(f"\n📊 Métricas de Consenso:")
    for metric, value in ensemble_result.consensus_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   • {metric}: {value:.3f}")