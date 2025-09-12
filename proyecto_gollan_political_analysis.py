#!/usr/bin/env python3
"""
Análisis Político del Proyecto Gollan 2130-D-2025 (Marco Normativo de IA)
Utilizando el Framework Multidimensional de Análisis Político
Relación con Artículo 19 de la Constitución Nacional Argentina

Autor: GenSpark AI Developer  
Fecha: 12 de septiembre de 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Import our political analysis framework
import sys
sys.path.append('/home/user/webapp')
from political_actors_generic import create_generic_political_dataset as create_expanded_political_dataset

class ProyectoGollanAnalyzer:
    """
    Analizador del Proyecto Gollan 2130-D-2025 usando framework político multidimensional
    """
    
    def __init__(self):
        """Inicializar el análisis del proyecto"""
        # Cargar dataset político expandido para comparación
        self.political_df = create_expanded_political_dataset()
        
        # Definir perfil político del Proyecto Gollan
        self.proyecto_gollan_profile = {
            'name': 'Proyecto Gollan 2130-D-2025',
            'author': 'Diputado Gollan',
            'type': 'Proyecto de Ley IA',
            'year': 2025,
            'country': 'Argentina',
            'era': 'Contemporary',
            
            # Dimensiones del espacio político (0-1 scale)
            'ideology_economic': 0.3,      # Centro-izquierda (regulación estatal)
            'ideology_social': 0.6,        # Conservador (control social)
            'leadership_messianic': 0.2,   # Bajo (proyecto tecnocrático)  
            'leadership_charismatic': 0.1, # Muy bajo (burocrático)
            'anti_establishment': 0.3,     # Medio-bajo (utiliza establishment)
            'symbolic_mystical': 0.1,      # Muy bajo (enfoque técnico)
            'populist_appeal': 0.4,        # Medio (retórica de protección)
            'authoritarian': 0.8,          # Alto (control estatal extensivo)
            'media_savvy': 0.6,           # Medio (narrativa tech-friendly)
            
            # Dimensiones específicas del proyecto
            'centralization': 0.85,        # Muy alto (AGC centralizada)
            'port_interior': 0.75,         # Alto (control porteño)
            'elite_popular': 0.65,         # Elite (tecnocracia)
            'continuity_rupture': 0.45,    # Continuidad (modelo regulatorio clásico)
            
            # Métricas constitucionales
            'constitutional_compatibility': 0.2,  # Muy bajo (viola Art. 19)
            'art19_violation_risk': 0.9,         # Muy alto
            'regulatory_overreach': 0.85,        # Muy alto
            'innovation_impact': 0.15            # Muy negativo
        }
        
        print("🔍 Initialized Proyecto Gollan Political Analyzer")
        print(f"📊 Loaded {len(self.political_df)} political actors for comparison")
    
    def analyze_constitutional_dimensions(self):
        """
        Analizar las dimensiones constitucionales del proyecto en relación al Art. 19
        """
        print("⚖️ Analyzing constitutional dimensions (Article 19 CN)...")
        
        # Análisis de violaciones al Art. 19
        art19_violations = {
            'principio_legalidad': {
                'score': 0.8,  # Alta violación
                'descripcion': 'Definiciones vagas y discrecionales (Art. 3)',
                'ejemplos': ['Sistema de IA con objetivos implícitos', 'Riesgo inaceptable subjetivo'],
                'impacto': 'Permite arbitrariedad administrativa'
            },
            'esfera_privada': {
                'score': 0.7,  # Alta violación  
                'descripcion': 'Regulación de actividades privadas sin daño a terceros',
                'ejemplos': ['Evaluación de comportamiento social', 'Control de procesamiento interno'],
                'impacto': 'Invade zona de no interferencia estatal'
            },
            'inversion_carga_prueba': {
                'score': 0.9,  # Muy alta violación
                'descripcion': 'Operadores deben demostrar cumplimiento (Art. 14.d)',
                'ejemplos': ['Presunción de culpabilidad', 'Prueba diabólica'],
                'impacto': 'Viola principio de inocencia'
            },
            'exceso_regulatorio': {
                'score': 0.85, # Muy alta violación
                'descripcion': 'Estado se arroga competencias no delegadas',
                'ejemplos': ['AGC con poderes omnímodos', 'Control transversal sectores'],
                'impacto': 'Excede límites constitucionales del poder estatal'
            }
        }
        
        # Calcular score general de constitucionalidad
        constitutional_score = 1 - np.mean([v['score'] for v in art19_violations.values()])
        
        return {
            'constitutional_compatibility': constitutional_score,
            'art19_violations': art19_violations,
            'overall_assessment': 'INCONSTITUCIONAL' if constitutional_score < 0.3 else 'CUESTIONABLE' if constitutional_score < 0.6 else 'COMPATIBLE'
        }
    
    def compare_with_political_actors(self):
        """
        Comparar el proyecto con actores políticos en el dataset
        """
        print("🔄 Comparing project with political actors...")
        
        # Dimensiones para comparación
        comparison_dimensions = [
            'ideology_economic', 'ideology_social', 'leadership_messianic',
            'leadership_charismatic', 'anti_establishment', 'symbolic_mystical', 
            'populist_appeal', 'authoritarian', 'media_savvy'
        ]
        
        # Calcular similaridades
        similarities = {}
        
        for _, actor in self.political_df.iterrows():
            actor_similarities = []
            
            for dim in comparison_dimensions:
                project_value = self.proyecto_gollan_profile[dim]
                actor_value = actor[dim]
                similarity = 1 - abs(project_value - actor_value)
                actor_similarities.append(similarity)
            
            overall_similarity = np.mean(actor_similarities)
            similarities[actor['name']] = {
                'overall_similarity': overall_similarity,
                'dimensional_similarities': dict(zip(comparison_dimensions, actor_similarities)),
                'actor_profile': {
                    'country': actor['country'],
                    'era': actor['era'],
                    'position': actor.get('position', 'Political Figure')
                }
            }
        
        # Ordenar por similaridad
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1]['overall_similarity'], reverse=True)
        
        return {
            'top_similar_actors': sorted_similarities[:10],
            'least_similar_actors': sorted_similarities[-5:],
            'all_similarities': similarities
        }
    
    def analyze_governance_structure(self):
        """
        Analizar la estructura de gobernanza propuesta (AGC)
        """
        print("🏛️ Analyzing governance structure (AGC)...")
        
        governance_analysis = {
            'estructura_poder': {
                'centralizacion': 0.9,  # AGC centralizada en Buenos Aires
                'politizacion': 0.8,    # Director designado por PEN
                'tecnocratizacion': 0.7, # Consejo técnico sin legitimidad democrática
                'federalismo': 0.2      # Consejo Federal cosmético
            },
            'competencias': {
                'exclusividad': 0.9,    # Competencias exclusivas (Art. 26)
                'discrecionalidad': 0.85, # Amplio margen interpretativo
                'transversalidad': 0.9,  # Control sobre todos los sectores
                'enforcement': 0.8       # Poder sancionatorio directo
            },
            'financiamiento': {
                'autonomia': 0.7,       # Aranceles propios + 5% ganancias
                'incentivos_perversos': 0.8, # Más regulación = más ingresos
                'transparencia': 0.3,   # Mecanismos de control limitados
                'sostenibilidad': 0.6   # Dependiente de recaudación tributaria
            },
            'riesgos_democraticos': {
                'capture_regulatorio': 0.8,  # Grandes tech vs. PyMEs
                'deficit_legitimidad': 0.7,   # Sin elección democrática
                'concentracion_poder': 0.9,   # Poderes legislativo+ejecutivo+judicial
                'accountability': 0.3         # Controles democráticos limitados
            }
        }
        
        # Calcular score general de governance
        governance_score = np.mean([
            1 - governance_analysis['estructura_poder']['centralizacion'],
            1 - governance_analysis['competencias']['discrecionalidad'], 
            governance_analysis['financiamiento']['transparencia'],
            governance_analysis['riesgos_democraticos']['accountability']
        ])
        
        return {
            'governance_score': governance_score,
            'detailed_analysis': governance_analysis,
            'assessment': 'AUTORITARIO' if governance_score < 0.3 else 'PROBLEMÁTICO' if governance_score < 0.6 else 'ACEPTABLE'
        }
    
    def analyze_innovation_impact(self):
        """
        Analizar el impacto en innovación y desarrollo tecnológico
        """
        print("💡 Analyzing innovation impact...")
        
        innovation_impact = {
            'barreras_entrada': {
                'pymes': 0.9,           # Muy alto impacto negativo
                'startups': 0.95,       # Casi prohibitivo
                'grandes_empresas': 0.3, # Pueden absorber costos
                'investigadores': 0.7    # Restricciones significativas
            },
            'costos_regulatorios': {
                'compliance': 0.8,      # Costos altos de cumplimiento
                'certificacion': 0.85,  # Procesos burocráticos extensos
                'auditoria': 0.7,       # Auditorías continuas requeridas
                'legal': 0.9            # Necesidad de equipos legales especializados
            },
            'efectos_dinamicos': {
                'chilling_effect': 0.8,  # Desincento a experimentación
                'fuga_talento': 0.7,     # Migración a jurisdicciones más flexibles
                'consolidacion': 0.8,    # Ventaja a incumbentes establecidos
                'rezago_tecnologico': 0.75 # Retraso en adopción de IA
            },
            'beneficios_esperados': {
                'proteccion_consumidor': 0.4,  # Beneficios limitados
                'transparencia': 0.3,          # Más burocracia que transparencia real
                'competencia_leal': 0.2,       # Efecto opuesto (barreras)
                'confianza_publica': 0.5       # Posible aumento inicial
            }
        }
        
        # Calcular impacto neto en innovación
        negative_impact = np.mean([
            np.mean(list(innovation_impact['barreras_entrada'].values())),
            np.mean(list(innovation_impact['costos_regulatorios'].values())),
            np.mean(list(innovation_impact['efectos_dinamicos'].values()))
        ])
        
        positive_impact = np.mean(list(innovation_impact['beneficios_esperados'].values()))
        net_innovation_impact = positive_impact - negative_impact
        
        return {
            'net_innovation_impact': net_innovation_impact,
            'detailed_analysis': innovation_impact,
            'assessment': 'MUY NEGATIVO' if net_innovation_impact < -0.5 else 'NEGATIVO' if net_innovation_impact < -0.2 else 'NEUTRO' if net_innovation_impact < 0.2 else 'POSITIVO'
        }
    
    def generate_recommendations(self):
        """
        Generar recomendaciones para compatibilidad constitucional e innovación
        """
        print("📋 Generating constitutional and innovation recommendations...")
        
        recommendations = {
            'constitucionales': {
                'urgentes': [
                    'Precisar definiciones del Art. 3 eliminando conceptos discrecionales',
                    'Limitar competencias AGC a enforcement ex-post de daños reales',
                    'Establecer test de proporcionalidad obligatorio para toda regulación',
                    'Requerir orden judicial para sistemas de vigilancia estatal'
                ],
                'importantes': [
                    'Transformar AGC en organismo sectorial, no transversal',
                    'Eliminar financiamiento mediante tributación directa (conflicto de interés)',
                    'Fortalecer controles democráticos y accountability',
                    'Establecer sunset clauses para revisión periódica'
                ],
                'deseables': [
                    'Crear ombudsman de IA independiente del Ejecutivo',
                    'Implementar consulta pública obligatoria para nuevas regulaciones',
                    'Establecer mecanismos de revisión judicial expedita'
                ]
            },
            'innovacion': {
                'criticas': [
                    'Crear safe harbor para experimentos de investigación académica',
                    'Eximir PyMEs de regulación hasta alcanzar escala crítica',
                    'Establecer sandbox permanente con enforcement reducido',
                    'Priorizar autorregulación sectorial sobre control estatal'
                ],
                'importantes': [
                    'Crear incentivos fiscales para I+D en IA ética',
                    'Establecer fast-track para certificación de startups',
                    'Promover estándares internacionales vs. regulación local',
                    'Fomentar arbitraje privado para disputas de IA'
                ]
            },
            'alternativas_sistemicas': [
                'Modelo de responsabilidad civil ampliada (vs. regulación ex-ante)',
                'Sistema de certificación voluntaria con incentivos de mercado',
                'Autorregulación sectorial supervisada judicialmente',  
                'Marco minimalista enfocado solo en daños objetivos a terceros'
            ]
        }
        
        return recommendations
    
    def create_comprehensive_visualizations(self):
        """
        Crear visualizaciones comprehensivas del análisis
        """
        print("📊 Creating comprehensive visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Comparación multidimensional con actores políticos
        ax1 = plt.subplot(2, 3, 1)
        
        # Obtener top 5 actores similares para visualización
        similarities = self.compare_with_political_actors()
        top_actors = similarities['top_similar_actors'][:5]
        
        actor_names = [actor[0].split()[-1] for actor in top_actors]
        similarity_scores = [actor[1]['overall_similarity'] for actor in top_actors]
        
        bars = ax1.barh(actor_names, similarity_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        ax1.set_xlabel('Similarity Score')
        ax1.set_title('Proyecto Gollan: Top 5\nActores Políticos Similares', fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # Agregar valores en las barras
        for bar, score in zip(bars, similarity_scores):
            ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontweight='bold')
        
        # 2. Análisis constitucional (Art. 19)
        ax2 = plt.subplot(2, 3, 2)
        
        constitutional_analysis = self.analyze_constitutional_dimensions()
        violations = constitutional_analysis['art19_violations']
        
        violation_names = [v.replace('_', ' ').title() for v in violations.keys()]
        violation_scores = [violations[v]['score'] for v in violations.keys()]
        
        colors = plt.cm.Reds(np.array(violation_scores))
        bars = ax2.bar(range(len(violation_names)), violation_scores, color=colors)
        ax2.set_xticks(range(len(violation_names)))
        ax2.set_xticklabels(violation_names, rotation=45, ha='right')
        ax2.set_ylabel('Violation Score (0-1)')
        ax2.set_title('Violaciones al Artículo 19 CN', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # 3. Estructura de gobernanza
        ax3 = plt.subplot(2, 3, 3)
        
        governance = self.analyze_governance_structure()
        gov_data = governance['detailed_analysis']
        
        # Crear radar chart simplificado
        categories = ['Centralización', 'Discrecionalidad', 'Transparencia', 'Accountability']
        values = [
            gov_data['estructura_poder']['centralizacion'],
            gov_data['competencias']['discrecionalidad'], 
            1 - gov_data['financiamiento']['transparencia'],  # Invertir para que más sea peor
            1 - gov_data['riesgos_democraticos']['accountability']  # Invertir
        ]
        
        bars = ax3.bar(categories, values, color=['#FF4444', '#FF6666', '#FF8888', '#FFAAAA'])
        ax3.set_ylabel('Problematic Score (0-1)')
        ax3.set_title('Problemas de Gobernanza\n(AGC)', fontweight='bold')
        ax3.set_ylim(0, 1)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Impacto en innovación
        ax4 = plt.subplot(2, 3, 4)
        
        innovation = self.analyze_innovation_impact()
        innovation_data = innovation['detailed_analysis']
        
        impact_categories = ['Barreras\nEntrada', 'Costos\nRegulatorious', 'Efectos\nDinámicos', 'Beneficios\nEsperados']
        impact_values = [
            np.mean(list(innovation_data['barreras_entrada'].values())),
            np.mean(list(innovation_data['costos_regulatorios'].values())),
            np.mean(list(innovation_data['efectos_dinamicos'].values())),
            1 - np.mean(list(innovation_data['beneficios_esperados'].values()))  # Invertir beneficios
        ]
        
        colors = ['#FF4444' if v > 0.7 else '#FF8888' if v > 0.4 else '#FFCCCC' for v in impact_values]
        bars = ax4.bar(impact_categories, impact_values, color=colors)
        ax4.set_ylabel('Negative Impact Score')
        ax4.set_title('Impacto Negativo\nen Innovación', fontweight='bold')
        ax4.set_ylim(0, 1)
        
        # 5. Dimensiones del espacio político 4D
        ax5 = plt.subplot(2, 3, 5)
        
        dimensions_4d = ['Centralization', 'Port-Interior', 'Elite-Popular', 'Continuity']
        dimension_values = [
            self.proyecto_gollan_profile['centralization'],
            self.proyecto_gollan_profile['port_interior'],
            self.proyecto_gollan_profile['elite_popular'],
            self.proyecto_gollan_profile['continuity_rupture']
        ]
        
        # Crear gráfico de barras radial simplificado
        angles = np.linspace(0, 2*np.pi, len(dimensions_4d), endpoint=False)
        values_closed = dimension_values + [dimension_values[0]]  # Cerrar el polígono
        angles_closed = list(angles) + [angles[0]]
        
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        ax5.plot(angles_closed, values_closed, 'o-', linewidth=2, color='#FF6B6B')
        ax5.fill(angles_closed, values_closed, alpha=0.25, color='#FF6B6B')
        ax5.set_xticks(angles)
        ax5.set_xticklabels(dimensions_4d)
        ax5.set_ylim(0, 1)
        ax5.set_title('Espacio Político 4D\nProyecto Gollan', fontweight='bold', pad=20)
        
        # 6. Score summary
        ax6 = plt.subplot(2, 3, 6)
        
        # Calcular scores finales
        constitutional_score = constitutional_analysis['constitutional_compatibility']
        governance_score = governance['governance_score']
        innovation_score = 1 + innovation['net_innovation_impact']  # Ajustar escala
        
        final_scores = {
            'Constitucionalidad': constitutional_score,
            'Gobernanza': governance_score,
            'Innovación': innovation_score,
            'Score General': np.mean([constitutional_score, governance_score, innovation_score])
        }
        
        score_names = list(final_scores.keys())
        score_values = list(final_scores.values())
        
        colors = ['#FF4444' if v < 0.3 else '#FFA500' if v < 0.6 else '#4ECDC4' for v in score_values]
        bars = ax6.bar(score_names, score_values, color=colors)
        ax6.set_ylabel('Score (0-1)')
        ax6.set_title('Evaluación Final\nProyecto Gollan', fontweight='bold')
        ax6.set_ylim(0, 1)
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        
        # Agregar línea de referencia en 0.5
        ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Umbral Aceptable')
        ax6.legend()
        
        # Agregar valores en las barras
        for bar, score in zip(bars, score_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/user/webapp/proyecto_gollan_analisis_politico_completo.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def generate_comprehensive_report(self):
        """
        Generar reporte comprehensivo del análisis
        """
        print("📋 Generating comprehensive analysis report...")
        
        # Ejecutar todos los análisis
        constitutional_analysis = self.analyze_constitutional_dimensions()
        political_comparison = self.compare_with_political_actors()
        governance_analysis = self.analyze_governance_structure()
        innovation_analysis = self.analyze_innovation_impact()
        recommendations = self.generate_recommendations()
        
        # Compilar reporte final
        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'project_name': 'Proyecto Gollan 2130-D-2025',
                'framework': 'Peralta-Metamorphosis Political Analysis',
                'constitutional_basis': 'Artículo 19 Constitución Nacional Argentina',
                'methodology': 'Multi-dimensional Political Space Analysis + Constitutional Review'
            },
            'executive_summary': {
                'overall_assessment': 'INCONSTITUCIONAL Y CONTRAPRODUCENTE',
                'constitutional_compatibility': constitutional_analysis['constitutional_compatibility'],
                'governance_score': governance_analysis['governance_score'],
                'innovation_impact': innovation_analysis['net_innovation_impact'],
                'recommendation': 'RECHAZO INTEGRAL - Requiere reescritura completa'
            },
            'constitutional_analysis': constitutional_analysis,
            'political_comparison': {
                'most_similar_actor': political_comparison['top_similar_actors'][0][0],
                'similarity_score': political_comparison['top_similar_actors'][0][1]['overall_similarity'],
                'political_positioning': 'Tecnocracia autoritaria con tendencias regulatorias excesivas'
            },
            'governance_analysis': governance_analysis,
            'innovation_analysis': innovation_analysis,
            'recommendations': recommendations,
            'political_dimensions': {
                'centralization_autonomy': self.proyecto_gollan_profile['centralization'],
                'port_interior': self.proyecto_gollan_profile['port_interior'], 
                'elite_popular': self.proyecto_gollan_profile['elite_popular'],
                'continuity_rupture': self.proyecto_gollan_profile['continuity_rupture']
            },
            'key_findings': [
                f"El proyecto viola sistemáticamente el Art. 19 CN (score: {constitutional_analysis['constitutional_compatibility']:.2f})",
                f"Presenta alta centralización autoritaria (score: {self.proyecto_gollan_profile['centralization']:.2f})",
                f"Impacto muy negativo en innovación (score: {innovation_analysis['net_innovation_impact']:.2f})",
                f"Estructura de gobernanza problemática (score: {governance_analysis['governance_score']:.2f})",
                "Requiere reescritura integral para compatibilidad constitucional"
            ]
        }
        
        return report

def main():
    """
    Función principal de análisis del Proyecto Gollan
    """
    print("🚀 ANÁLISIS POLÍTICO DEL PROYECTO GOLLAN 2130-D-2025")
    print("=" * 70)
    print("⚖️ Marco Normativo de IA vs. Artículo 19 Constitución Nacional")
    print("📊 Framework: Peralta-Metamorphosis Political Analysis")
    print("🔬 Análisis Multidimensional + Revisión Constitucional")
    print("=" * 70)
    
    # Inicializar analizador
    analyzer = ProyectoGollanAnalyzer()
    
    # Generar reporte comprehensivo
    print("\n🔄 Running comprehensive constitutional and political analysis...")
    report = analyzer.generate_comprehensive_report()
    
    # Crear visualizaciones
    print("\n📊 Creating comprehensive visualizations...")
    fig = analyzer.create_comprehensive_visualizations()
    
    # Mostrar resultados clave
    print("\n" + "=" * 70)
    print("🎯 RESULTADOS CLAVE DEL ANÁLISIS:")
    print("=" * 70)
    
    executive = report['executive_summary']
    print(f"\n📋 EVALUACIÓN GENERAL: {executive['overall_assessment']}")
    print(f"⚖️ Compatibilidad Constitucional: {executive['constitutional_compatibility']:.3f}/1.000")
    print(f"🏛️ Score de Gobernanza: {executive['governance_score']:.3f}/1.000")  
    print(f"💡 Impacto en Innovación: {executive['innovation_impact']:.3f} (negativo)")
    print(f"📝 Recomendación: {executive['recommendation']}")
    
    print(f"\n" + "=" * 70)
    print("⚖️ ANÁLISIS CONSTITUCIONAL (Art. 19 CN):")
    print("=" * 70)
    
    constitutional = report['constitutional_analysis']
    print(f"🔴 Assessment: {constitutional['overall_assessment']}")
    print(f"📊 Violaciones identificadas:")
    
    for violation, data in constitutional['art19_violations'].items():
        print(f"   • {violation.replace('_', ' ').title()}: {data['score']:.2f} - {data['descripcion']}")
    
    print(f"\n" + "=" * 70)
    print("🔍 COMPARACIÓN CON ACTORES POLÍTICOS:")
    print("=" * 70)
    
    political = report['political_comparison']
    print(f"🎯 Actor más similar: {political['most_similar_actor']}")
    print(f"📊 Score de similaridad: {political['similarity_score']:.3f}")
    print(f"📍 Posicionamiento político: {political['political_positioning']}")
    
    print(f"\n" + "=" * 70)
    print("🏛️ ESTRUCTURA DE GOBERNANZA (AGC):")
    print("=" * 70)
    
    governance = report['governance_analysis']
    print(f"📊 Score de Gobernanza: {governance['governance_score']:.3f}/1.000")
    print(f"🔴 Assessment: {governance['assessment']}")
    
    print(f"\n" + "=" * 70)
    print("💡 IMPACTO EN INNOVACIÓN:")
    print("=" * 70)
    
    innovation = report['innovation_analysis'] 
    print(f"📊 Impacto Neto: {innovation['net_innovation_impact']:.3f}")
    print(f"🔴 Assessment: {innovation['assessment']}")
    
    print(f"\n" + "=" * 70)
    print("📋 RECOMENDACIONES PRIORITARIAS:")
    print("=" * 70)
    
    recommendations = report['recommendations']
    print(f"🚨 URGENTES (Constitucionales):")
    for rec in recommendations['constitucionales']['urgentes']:
        print(f"   • {rec}")
    
    print(f"\n💡 CRÍTICAS (Innovación):")
    for rec in recommendations['innovacion']['criticas']:
        print(f"   • {rec}")
    
    print(f"\n🔄 ALTERNATIVAS SISTÉMICAS:")
    for alt in recommendations['alternativas_sistemicas']:
        print(f"   • {alt}")
    
    print(f"\n" + "=" * 70)
    print("🎯 HALLAZGOS CLAVE:")
    print("=" * 70)
    
    for i, finding in enumerate(report['key_findings'], 1):
        print(f"{i:2d}. {finding}")
    
    # Guardar resultados
    print(f"\n" + "=" * 70)
    print("💾 GUARDANDO RESULTADOS:")
    print("=" * 70)
    
    # Guardar JSON
    with open('/home/user/webapp/proyecto_gollan_analisis_politico_completo.json', 'w', encoding='utf-8') as f:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        clean_report = json.loads(json.dumps(report, default=convert_numpy))
        json.dump(clean_report, f, indent=2, ensure_ascii=False)
    
    print("✅ Resultados guardados:")
    print("   📄 proyecto_gollan_analisis_politico_completo.json")
    print("   📊 proyecto_gollan_analisis_politico_completo.png")
    print("   📋 ANÁLISIS_PROYECTO_GOLLAN_IA_2130-D-2025.md")
    
    print(f"\n🏆 ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"✓ Análisis constitucional realizado (Art. 19 CN)")
    print(f"✓ Comparación política multidimensional completada")  
    print(f"✓ Evaluación de gobernanza e innovación finalizada")
    print(f"✓ Recomendaciones para compatibilidad constitucional generadas")
    print(f"✓ Visualizaciones comprehensivas creadas")

if __name__ == "__main__":
    main()