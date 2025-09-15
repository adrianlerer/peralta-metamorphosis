#!/usr/bin/env python3
"""
Peralta Comprehensive Demo: Demostración integral de las capacidades mejoradas

Este script demuestra la integración completa de:
1. SLM Agentic Optimization (NVIDIA-inspired) 
2. Neurofeedback Metacognitivo (arXiv:2505.13763)
3. Reality Filter 2.0 con gradientes de confianza
4. Red-teaming automático y validación de integridad  
5. Análisis genealógico político avanzado
6. Enrutamiento inteligente Kimi K2 + Apertus-70B-Instruct

Author: Ignacio Adrián Lerer
Project: Peralta Universal Analysis Framework Enhancement
Integration: Paper arXiv:2505.13763 + NVIDIA SLM + Reality Filter 2.0
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any
import logging
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Peralta enhanced components
from universal_analysis_framework.domains.peralta_enhanced_analyzer import (
    PeraltaEnhancedAnalyzer, 
    PeraltaAnalysisRequest,
    create_demo_analysis_request
)
from universal_analysis_framework.domains.metacognitive_neurofeedback import (
    MetacognitiveNeurofeedback,
    create_demo_dataset
)
from universal_analysis_framework.domains.slm_agentic_optimizer import (
    SLMAgenticOptimizer,
    AgenticTask,
    TaskComplexity,
    AgenticMode
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('peralta_demo.log')
    ]
)
logger = logging.getLogger(__name__)

class PeraltaComprehensiveDemo:
    """
    Demostración integral del sistema Peralta mejorado con todas las capacidades avanzadas.
    """
    
    def __init__(self):
        self.demo_results = {}
        self.execution_metrics = {}
        
        # Demo scenarios
        self.demo_scenarios = {
            'legal_analysis': {
                'text': """
                El marco jurídico argentino establece en el Artículo 75 de la Constitución Nacional 
                las facultades del Congreso para dictar códigos de fondo. La interpretación de la 
                Corte Suprema en el caso "Kot, Samuel" (1958) estableció precedentes sobre el habeas 
                corpus que aún hoy influencian la jurisprudencia constitucional. En el contexto 
                del derecho administrativo, la Ley 19.549 de Procedimientos Administrativos requiere 
                que toda actuación administrativa sea motivada, proporcionada y respetuosa del debido proceso.
                """,
                'analysis_type': 'legal',
                'expected_model': 'apertus_70b_instruct',
                'confidence_level': '[Verificado]'
            },
            
            'political_genealogy': {
                'text': """
                La evolución del peronismo argentino desde 1946 hasta 2023 muestra una genealogía 
                compleja de transformaciones ideológicas. Desde el justicialismo original de Perón 
                y Evita, pasando por la resistencia peronista (1955-1973), la ortodoxia de López Rega, 
                el peronismo renovador de los '80, hasta llegar al kirchnerismo (2003-2023) y la 
                actual tensión con el mileísmo. Cada etapa mantiene elementos genealógicos del 
                peronismo fundacional pero incorpora nuevas síntesis según el contexto histórico.
                """,
                'analysis_type': 'genealogical',
                'expected_model': 'kimi_k2',
                'confidence_level': '[Estimación]'
            },
            
            'corruption_analysis': {
                'text': """
                El análisis de los contratos de obra pública adjudicados durante 2022-2023 muestra 
                cumplimiento general de procedimientos de transparencia. De 1,247 licitaciones 
                analizadas, 1,201 siguieron protocolos estándar de publicación en el Boletín Oficial 
                y recibieron al menos 3 ofertas competitivas. Las 46 adjudicaciones restantes 
                corresponden a contrataciones directas justificadas por emergencia, todas con 
                documentación respaldatoria adecuada y aprobación de autoridad competente.
                """,
                'analysis_type': 'corruption',
                'expected_model': 'apertus_70b_instruct',
                'confidence_level': '[Estimación]'
            },
            
            'political_analysis': {
                'text': """
                La polarización política argentina entre 2019-2023 reproduce antagonismos estructurales 
                de larga duración. La tensión entre Cristina Kirchner (nacional-popular) y Mauricio Macri 
                (liberal-conservador) replica la histórica división peronismo-antiperonismo. La irrupción 
                de Javier Milei en 2021 introduce una terceira fuerza libertaria que desafía tanto 
                al establishment kirchnerista como macrista, generando un nuevo mapa de alianzas y 
                conflictos que trasciende las categorías políticas tradicionales.
                """,
                'analysis_type': 'political', 
                'expected_model': 'kimi_k2',
                'confidence_level': '[Inferencia razonada]'
            }
        }
        
    def print_header(self, title: str):
        """Print formatted section header"""
        print("\n" + "=" * 80)
        print(f"🎯 {title}")
        print("=" * 80)
        
    def print_sub_header(self, title: str):
        """Print formatted subsection header"""
        print(f"\n🔹 {title}")
        print("-" * 60)
        
    async def demo_neurofeedback_standalone(self):
        """Demonstrar sistema neurofeedback metacognitivo independiente"""
        self.print_header("DEMO: Neurofeedback Metacognitivo (arXiv:2505.13763)")
        
        start_time = datetime.now()
        
        # Inicializar sistema neurofeedback
        neurofeedback = MetacognitiveNeurofeedback(
            model_name="gpt-4o-mini",
            openrouter_api_key=None  # Usar simulación para demo
        )
        
        # Crear dataset de evaluación
        demo_dataset = create_demo_dataset()
        print(f"📊 Dataset creado: {len(demo_dataset)} ejemplos")
        print(f"Ejes evaluados: {demo_dataset['axis_name'].unique().tolist()}")
        
        # Ejecutar análisis neurofeedback
        print("\n🧠 Ejecutando análisis neurofeedback completo...")
        results = neurofeedback.run_comprehensive_analysis(demo_dataset)
        
        # Mostrar resultados clave
        summary = results.get('summary_metrics', {})
        if summary:
            self.print_sub_header("Métricas de Neurofeedback")
            print(f"Precisión promedio de reporte: {summary['mean_reporting_accuracy']:.3f}")
            print(f"Cohen's d promedio (control): {summary['mean_control_cohens_d']:.3f}")
            print(f"Precisión promedio de control: {summary['mean_control_precision']:.3f}")
            print(f"Eje más controlable: {summary['most_controllable_axis']}")
            print(f"Eje menos controlable: {summary['least_controllable_axis']}")
        
        # Recomendaciones de seguridad
        safety_recs = results.get('safety_recommendations', {})
        if safety_recs:
            self.print_sub_header("Recomendaciones de Seguridad")
            for rec in safety_recs.get('specific_recommendations', []):
                print(f"- {rec}")
        
        # Generar reporte completo
        report = neurofeedback.generate_analysis_report(results)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        self.execution_metrics['neurofeedback_demo'] = execution_time
        self.demo_results['neurofeedback_standalone'] = results
        
        print(f"\n✅ Demo neurofeedback completado en {execution_time:.2f}s")
        return results
        
    async def demo_slm_agentic_optimization(self):
        """Demonstrar SLM Agentic Optimization"""
        self.print_header("DEMO: SLM Agentic Optimization (NVIDIA-inspired)")
        
        start_time = datetime.now()
        
        # Configurar optimizador SLM
        optimizer = SLMAgenticOptimizer(
            domain="legal_political_analysis"
        )
        
        # Test con diferentes escenarios
        test_scenarios = ['legal_analysis', 'political_genealogy']
        
        for scenario_name in test_scenarios:
            scenario = self.demo_scenarios[scenario_name]
            
            self.print_sub_header(f"Análisis SLM: {scenario_name}")
            print(f"Texto: {scenario['text'][:100]}...")
            print(f"Modelo esperado: {scenario['expected_model']}")
            
            # Ejecutar análisis (simulado)
            result = await optimizer.run_comprehensive_analysis(
                text=scenario['text'],
                analysis_type=scenario['analysis_type']
            )
            
            print(f"✅ Análisis completado - Confianza: {result.get('confidence', 0):.3f}")
            print(f"Modelo utilizado: {result.get('model_used', 'simulado')}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        self.execution_metrics['slm_optimization_demo'] = execution_time
        
        print(f"\n✅ Demo SLM optimization completado en {execution_time:.2f}s")
        
    async def demo_peralta_enhanced_comprehensive(self):
        """Demonstrar sistema Peralta Enhanced completo"""
        self.print_header("DEMO: Peralta Enhanced Analyzer - Sistema Integral")
        
        start_time = datetime.now()
        
        # Inicializar sistema integral
        analyzer = PeraltaEnhancedAnalyzer(enable_advanced_features=True)
        
        print(f"🚀 Sistema Peralta Enhanced inicializado")
        print(f"Componentes activos: SLM Agentic + Neurofeedback + Reality Filter 2.0")
        
        # Analizar cada escenario
        scenario_results = {}
        
        for scenario_name, scenario_config in self.demo_scenarios.items():
            self.print_sub_header(f"Análisis Integral: {scenario_name}")
            
            # Crear solicitud de análisis
            request = PeraltaAnalysisRequest(
                text=scenario_config['text'],
                analysis_type=scenario_config['analysis_type'],
                confidence_level=scenario_config['confidence_level'],
                enable_neurofeedback=True,
                enable_red_teaming=True,
                genealogical_depth=3
            )
            
            print(f"Tipo: {request.analysis_type}")
            print(f"Nivel de confianza: {request.confidence_level}")
            print(f"Texto: {request.text[:100]}...")
            
            # Ejecutar análisis integral
            result = await analyzer.analyze_comprehensive(request)
            
            # Mostrar resultados clave
            print(f"\n📊 Resultados del análisis:")
            print(f"Modelo seleccionado: {result.model_routing_decision['selected_model']}")
            print(f"Score de integridad: {result.integrity_score:.3f}")
            print(f"Gradiente de confianza promedio: {np.mean(list(result.confidence_gradient.values())):.3f}")
            
            # Neurofeedback assessment
            if result.neurofeedback_assessment:
                neurofeedback = result.neurofeedback_assessment
                bias_alerts = len(neurofeedback.get('bias_alerts', []))
                integrity_score = neurofeedback.get('overall_integrity_score', 1.0)
                print(f"Neurofeedback - Alertas de sesgo: {bias_alerts}, Integridad: {integrity_score:.3f}")
            
            # Red-team evaluation
            if result.red_team_evaluation:
                red_team = result.red_team_evaluation
                resistance = red_team.get('resistance_score', 1.0)
                security_level = red_team.get('overall_security_level', 'N/A')
                print(f"Red-team - Resistencia: {resistance:.3f}, Nivel: {security_level}")
            
            # Genealogical analysis
            if result.genealogical_trace:
                genealogical = result.genealogical_trace
                actors = len(genealogical.get('actors_identified', []))
                concepts = len(genealogical.get('concepts_identified', []))
                print(f"Genealogía - Actores: {actors}, Conceptos: {concepts}")
            
            scenario_results[scenario_name] = result
            
            print(f"✅ Análisis {scenario_name} completado\n")
        
        # Generar reporte consolidado
        self.print_sub_header("Reporte Consolidado")
        
        total_analyses = len(scenario_results)
        avg_integrity = np.mean([r.integrity_score for r in scenario_results.values()])
        model_usage = {}
        
        for result in scenario_results.values():
            model = result.model_routing_decision['selected_model']
            model_usage[model] = model_usage.get(model, 0) + 1
        
        print(f"Total de análisis: {total_analyses}")
        print(f"Integridad promedio: {avg_integrity:.3f}")
        print(f"Uso de modelos: {model_usage}")
        
        # Métricas del sistema
        print(f"\nMétricas del sistema:")
        for metric, value in analyzer.metrics.items():
            print(f"  - {metric}: {value}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        self.execution_metrics['peralta_comprehensive_demo'] = execution_time
        self.demo_results['peralta_enhanced'] = scenario_results
        
        print(f"\n✅ Demo Peralta Enhanced completado en {execution_time:.2f}s")
        return scenario_results
        
    def generate_demo_report(self):
        """Generar reporte consolidado de toda la demostración"""
        self.print_header("REPORTE CONSOLIDADO DE DEMOSTRACIÓN")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"**Fecha:** {timestamp}")
        print(f"**Componentes demostrados:** Neurofeedback + SLM Agentic + Peralta Enhanced")
        
        self.print_sub_header("Métricas de Ejecución")
        total_time = sum(self.execution_metrics.values())
        
        for demo_name, exec_time in self.execution_metrics.items():
            print(f"- {demo_name}: {exec_time:.2f}s")
        print(f"- **Total**: {total_time:.2f}s")
        
        self.print_sub_header("Resultados Clave")
        
        # Análisis de neurofeedback
        if 'neurofeedback_standalone' in self.demo_results:
            neurofeedback_results = self.demo_results['neurofeedback_standalone']
            summary = neurofeedback_results.get('summary_metrics', {})
            
            if summary:
                print(f"🧠 **Neurofeedback Metacognitivo:**")
                print(f"   - Precisión promedio: {summary['mean_reporting_accuracy']:.3f}")
                print(f"   - Control Cohen's d: {summary['mean_control_cohens_d']:.3f}")
                print(f"   - Eje más/menos controlable: {summary['most_controllable_axis']} / {summary['least_controllable_axis']}")
        
        # Análisis de Peralta Enhanced
        if 'peralta_enhanced' in self.demo_results:
            peralta_results = self.demo_results['peralta_enhanced']
            
            avg_integrity = np.mean([r.integrity_score for r in peralta_results.values()])
            model_routing = {}
            
            for scenario, result in peralta_results.items():
                model = result.model_routing_decision['selected_model']
                model_routing[model] = model_routing.get(model, 0) + 1
            
            print(f"\n🚀 **Peralta Enhanced Analyzer:**")
            print(f"   - Análisis procesados: {len(peralta_results)}")
            print(f"   - Integridad promedio: {avg_integrity:.3f}")
            print(f"   - Enrutamiento de modelos: {model_routing}")
        
        self.print_sub_header("Innovaciones Implementadas")
        
        innovations = [
            "✅ Neurofeedback metacognitivo basado en arXiv:2505.13763",
            "✅ SLM Agentic Optimization inspirado en NVIDIA",
            "✅ Reality Filter 2.0 con gradientes de confianza",
            "✅ Enrutamiento inteligente Kimi K2 + Apertus-70B-Instruct",
            "✅ Red-teaming automático integrado",
            "✅ Análisis genealógico político avanzado",
            "✅ Detección de sesgos internos en tiempo real",
            "✅ Hibridación adaptativa según contexto de análisis"
        ]
        
        for innovation in innovations:
            print(f"  {innovation}")
        
        self.print_sub_header("Próximos Pasos")
        
        next_steps = [
            "🔄 Integrar con APIs reales de OpenRouter (Kimi K2, Apertus-70B)",
            "📊 Implementar métricas de performance en producción", 
            "🛡️ Expandir capacidades de red-teaming y detección de evasión",
            "🌳 Profundizar análisis genealógico con datos históricos reales",
            "⚖️ Validar sistema con casos legales y políticos reales argentinos",
            "🎯 Optimizar enrutamiento de modelos basado en feedback de usage"
        ]
        
        for step in next_steps:
            print(f"  {step}")
        
        print(f"\n📄 **Reporte generado:** peralta_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    def save_demo_results(self):
        """Guardar resultados de la demostración"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"peralta_comprehensive_demo_results_{timestamp}.json"
        
        # Preparar datos para serialización
        serializable_results = {}
        
        # Convert results to serializable format
        for key, value in self.demo_results.items():
            if key == 'peralta_enhanced':
                # Handle PeraltaAnalysisResult objects
                serializable_value = {}
                for scenario_name, result in value.items():
                    serializable_value[scenario_name] = {
                        'primary_analysis': result.primary_analysis,
                        'neurofeedback_assessment': result.neurofeedback_assessment,
                        'red_team_evaluation': result.red_team_evaluation,
                        'confidence_gradient': result.confidence_gradient,
                        'genealogical_trace': result.genealogical_trace,
                        'model_routing_decision': result.model_routing_decision,
                        'integrity_score': result.integrity_score,
                        'recommendations': result.recommendations
                    }
                serializable_results[key] = serializable_value
            else:
                serializable_results[key] = value
        
        # Create complete report
        complete_report = {
            'timestamp': datetime.now().isoformat(),
            'demo_metadata': {
                'components_tested': ['neurofeedback', 'slm_agentic', 'peralta_enhanced'],
                'scenarios_executed': list(self.demo_scenarios.keys()),
                'total_execution_time': sum(self.execution_metrics.values()),
                'paper_implementations': ['arXiv:2505.13763', 'NVIDIA SLM Agentic', 'Reality Filter 2.0']
            },
            'execution_metrics': self.execution_metrics,
            'demo_results': serializable_results,
            'innovations_demonstrated': [
                'metacognitive_neurofeedback',
                'slm_agentic_optimization', 
                'reality_filter_2_0',
                'intelligent_model_routing',
                'automated_red_teaming',
                'political_genealogical_analysis'
            ]
        }
        
        # Save to file
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados completos guardados en: {results_file}")
        return results_file

async def main():
    """
    Función principal para ejecutar la demostración integral del sistema Peralta Enhanced.
    """
    print("🚀 PERALTA COMPREHENSIVE DEMO - Sistema Integral Mejorado")
    print("=" * 80)
    print("Integración completa de:")
    print("- 🧠 Neurofeedback Metacognitivo (arXiv:2505.13763)")
    print("- ⚡ SLM Agentic Optimization (NVIDIA-inspired)")  
    print("- 🔍 Reality Filter 2.0 con gradientes de confianza")
    print("- 🛡️ Red-teaming automático integrado")
    print("- 🌳 Análisis genealógico político avanzado")
    print("- 🎯 Enrutamiento inteligente Kimi K2 + Apertus-70B-Instruct")
    print("=" * 80)
    
    # Inicializar demostrador
    demo = PeraltaComprehensiveDemo()
    
    try:
        # 1. Demo Neurofeedback standalone
        await demo.demo_neurofeedback_standalone()
        
        # 2. Demo SLM Agentic Optimization
        await demo.demo_slm_agentic_optimization()
        
        # 3. Demo Peralta Enhanced integral
        await demo.demo_peralta_enhanced_comprehensive()
        
        # 4. Generar reporte consolidado
        demo.generate_demo_report()
        
        # 5. Guardar resultados
        results_file = demo.save_demo_results()
        
        print(f"\n🎉 DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        print(f"📊 Resultados guardados en: {results_file}")
        print(f"⏱️ Tiempo total: {sum(demo.execution_metrics.values()):.2f}s")
        
    except Exception as e:
        logger.error(f"Error en demostración: {str(e)}")
        print(f"\n❌ Error en demostración: {str(e)}")
        raise
    
    return demo

if __name__ == "__main__":
    import numpy as np
    
    # Ejecutar demostración
    demo_result = asyncio.run(main())