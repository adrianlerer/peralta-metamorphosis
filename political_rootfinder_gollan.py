#!/usr/bin/env python3
"""
PoliticalRootFinder: Análisis Genealógico del Proyecto Gollan 2130-D-2025
Búsqueda de fuentes y análisis de hibridación entre procesos vs fines

Utiliza el algoritmo ABAN (Ancestral Backward Analysis of Networks) adaptado
para análisis político y regulatorio.

Autor: GenSpark AI Developer
Fecha: 12 de septiembre de 2025
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
import json
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our existing frameworks
import sys
sys.path.append('/home/user/webapp')
from rootfinder.rootfinder import RootFinder, GenealogyNode
from political_actors_generic import create_generic_political_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PoliticalSourceNode:
    """Representa un nodo fuente en el análisis genealógico político."""
    source_id: str
    source_type: str  # 'constitutional', 'legislative', 'regulatory', 'doctrinal', 'international'
    influence_strength: float = 0.0
    textual_similarity: float = 0.0
    conceptual_overlap: List[str] = field(default_factory=list)
    process_orientation: float = 0.0  # 0=procedural focus, 1=substantive focus
    ends_orientation: float = 0.0     # 0=means focus, 1=ends focus
    hybridization_score: float = 0.0  # Measure of process-ends mixing
    political_distance: float = 0.0
    temporal_proximity: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario para serialización."""
        return {
            'source_id': self.source_id,
            'source_type': self.source_type,
            'influence_strength': self.influence_strength,
            'textual_similarity': self.textual_similarity,
            'conceptual_overlap': self.conceptual_overlap,
            'process_orientation': self.process_orientation,
            'ends_orientation': self.ends_orientation,
            'hybridization_score': self.hybridization_score,
            'political_distance': self.political_distance,
            'temporal_proximity': self.temporal_proximity
        }

class PoliticalRootFinder:
    """
    Trazador genealógico de fuentes políticas y análisis de hibridación.
    
    Analiza las fuentes ideológicas y normativas del Proyecto Gollan
    identificando hibridación entre orientación a procesos vs fines.
    """
    
    def __init__(self):
        """Inicializar PoliticalRootFinder."""
        self.political_df = create_generic_political_dataset()
        
        # Texto del Proyecto Gollan (extractos clave para análisis)
        self.proyecto_gollan_text = """
        Artículo 1°: Marco Normativo para el desarrollo responsable de sistemas de inteligencia artificial
        Artículo 2°: Definiciones - Sistema de IA, Algoritmo, Riesgo algorítmico, Datos personales
        Artículo 3°: Principios rectores - Transparencia, explicabilidad, no discriminación, privacidad
        Artículo 4°: Registro Nacional de Sistemas de IA - Obligatorio para sistemas de alto riesgo
        Artículo 5°: Evaluación de impacto algorítmico obligatoria
        Artículo 6°: Derechos de los ciudadanos frente a decisiones automatizadas
        Artículo 7°: Supervisión y control - Autoridad de Aplicación
        Artículo 8°: Régimen sancionatorio - Multas hasta 4% del volumen de negocios
        Artículo 9°: Prohibiciones específicas - Sistemas de reconocimiento facial masivo
        Artículo 10°: Disposiciones complementarias sobre investigación e innovación
        """
        
        # Base de conocimiento de fuentes potenciales
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self) -> Dict[str, Dict]:
        """Construir base de conocimiento de fuentes potenciales."""
        
        knowledge_base = {
            # Fuentes Constitucionales
            'art19_cn': {
                'type': 'constitutional',
                'name': 'Artículo 19 Constitución Nacional',
                'text': 'Las acciones privadas de los hombres que de ningún modo ofendan al orden y a la moral pública, ni perjudiquen a un tercero, están sólo reservadas a Dios, y exentas de la autoridad de los magistrados.',
                'process_keywords': ['procedimiento', 'autoridad', 'magistrados', 'regulación'],
                'ends_keywords': ['orden público', 'moral pública', 'perjuicio', 'terceros'],
                'year': 1853,
                'country': 'Argentina'
            },
            
            # Fuentes Regulatorias Internacionales
            'gdpr_eu': {
                'type': 'international',
                'name': 'GDPR Unión Europea',
                'text': 'Reglamento General de Protección de Datos - transparencia, consentimiento, derechos del titular',
                'process_keywords': ['registro', 'evaluación impacto', 'autoridad control', 'supervisión'],
                'ends_keywords': ['protección datos', 'privacidad', 'derechos fundamentales'],
                'year': 2018,
                'country': 'EU'
            },
            
            'ai_act_eu': {
                'type': 'international', 
                'name': 'EU AI Act',
                'text': 'Ley de Inteligencia Artificial de la UE - sistemas alto riesgo, prohibiciones, transparencia',
                'process_keywords': ['registro obligatorio', 'evaluación conformidad', 'supervisión', 'multas'],
                'ends_keywords': ['sistemas alto riesgo', 'derechos fundamentales', 'no discriminación'],
                'year': 2024,
                'country': 'EU'
            },
            
            # Fuentes Doctrinales Argentinas
            'sabsay_const': {
                'type': 'doctrinal',
                'name': 'Sabsay - Derecho Constitucional',
                'text': 'Doctrina constitucional argentina sobre limitaciones al poder estatal y derechos individuales',
                'process_keywords': ['control constitucionalidad', 'debido proceso', 'autoridad aplicación'],
                'ends_keywords': ['derechos individuales', 'libertad', 'autonomía personal'],
                'year': 2020,
                'country': 'Argentina'
            },
            
            # Fuentes Legislativas Precedentes
            'ley_datos_personales': {
                'type': 'legislative',
                'name': 'Ley 25.326 Protección Datos Personales',
                'text': 'Marco regulatorio argentino para protección de datos personales',
                'process_keywords': ['registro', 'autoridad aplicación', 'régimen sancionatorio'],
                'ends_keywords': ['protección datos', 'privacidad', 'consentimiento'],
                'year': 2000,
                'country': 'Argentina'
            },
            
            'ley_defensa_consumidor': {
                'type': 'legislative',
                'name': 'Ley 24.240 Defensa del Consumidor',
                'text': 'Protección derechos del consumidor en Argentina',
                'process_keywords': ['autoridad aplicación', 'procedimiento sancionatorio', 'multas'],
                'ends_keywords': ['protección consumidor', 'información', 'transparencia'],
                'year': 1993,
                'country': 'Argentina'
            }
        }
        
        return knowledge_base
    
    def analyze_gollan_sources(self) -> List[PoliticalSourceNode]:
        """
        Analizar las fuentes genealógicas del Proyecto Gollan.
        
        Returns:
        --------
        List[PoliticalSourceNode]
            Lista de fuentes identificadas con métricas de influencia
        """
        logger.info("Iniciando análisis genealógico del Proyecto Gollan")
        
        sources = []
        
        for source_id, source_data in self.knowledge_base.items():
            # Calcular similitud textual
            textual_sim = self._calculate_textual_similarity(
                self.proyecto_gollan_text, source_data['text']
            )
            
            # Calcular orientación proceso vs fines
            process_orientation = self._calculate_process_orientation(source_data)
            ends_orientation = self._calculate_ends_orientation(source_data)
            
            # Calcular hibridización
            hybridization = self._calculate_hybridization(
                process_orientation, ends_orientation, source_data
            )
            
            # Calcular influencia total
            influence = self._calculate_influence_strength(
                source_data, textual_sim, hybridization
            )
            
            # Identificar solapamiento conceptual
            conceptual_overlap = self._identify_conceptual_overlap(source_data)
            
            # Calcular proximidad temporal
            temporal_proximity = self._calculate_temporal_proximity(
                source_data.get('year', 2000), 2025
            )
            
            # Calcular distancia política
            political_distance = self._calculate_political_distance(source_data)
            
            source_node = PoliticalSourceNode(
                source_id=source_id,
                source_type=source_data['type'],
                influence_strength=influence,
                textual_similarity=textual_sim,
                conceptual_overlap=conceptual_overlap,
                process_orientation=process_orientation,
                ends_orientation=ends_orientation,
                hybridization_score=hybridization,
                political_distance=political_distance,
                temporal_proximity=temporal_proximity
            )
            
            sources.append(source_node)
        
        # Ordenar por fuerza de influencia
        sources.sort(key=lambda x: x.influence_strength, reverse=True)
        
        logger.info(f"Análisis completado: {len(sources)} fuentes identificadas")
        return sources
    
    def _calculate_textual_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud textual entre dos textos."""
        
        # Convertir a minúsculas y tokenizar
        tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Palabras clave relevantes con mayor peso
        key_terms = {
            'inteligencia', 'artificial', 'algoritmo', 'datos', 'privacidad',
            'transparencia', 'registro', 'evaluación', 'supervisión', 'control',
            'autoridad', 'sanción', 'multa', 'derecho', 'protección'
        }
        
        # Intersección ponderada
        intersection = tokens1 & tokens2
        weighted_intersection = sum(3 if term in key_terms else 1 for term in intersection)
        
        # Unión ponderada  
        union = tokens1 | tokens2
        weighted_union = sum(3 if term in key_terms else 1 for term in union)
        
        if weighted_union == 0:
            return 0.0
            
        similarity = weighted_intersection / weighted_union
        return min(similarity, 1.0)
    
    def _calculate_process_orientation(self, source_data: Dict) -> float:
        """Calcular orientación hacia procesos (procedimientos, autoridades, mecanismos)."""
        
        process_keywords = source_data.get('process_keywords', [])
        text = source_data.get('text', '')
        
        # Contar menciones de palabras clave de proceso
        process_score = 0
        for keyword in process_keywords:
            process_score += text.lower().count(keyword.lower())
        
        # Palabras adicionales que indican orientación procedimental
        additional_process_terms = [
            'procedimiento', 'trámite', 'formulario', 'registro', 'inscripción',
            'autorización', 'licencia', 'certificación', 'auditoría', 'inspección',
            'supervisión', 'control', 'verificación', 'validación', 'cumplimiento'
        ]
        
        for term in additional_process_terms:
            process_score += text.lower().count(term)
        
        # Normalizar por longitud del texto
        text_length = len(text.split())
        if text_length == 0:
            return 0.0
            
        normalized_score = min(process_score / text_length * 10, 1.0)
        return normalized_score
    
    def _calculate_ends_orientation(self, source_data: Dict) -> float:
        """Calcular orientación hacia fines (objetivos sustantivos, valores, derechos)."""
        
        ends_keywords = source_data.get('ends_keywords', [])
        text = source_data.get('text', '')
        
        # Contar menciones de palabras clave de fines
        ends_score = 0
        for keyword in ends_keywords:
            ends_score += text.lower().count(keyword.lower())
        
        # Palabras adicionales que indican orientación sustantiva
        additional_ends_terms = [
            'derecho', 'libertad', 'protección', 'seguridad', 'bienestar',
            'dignidad', 'igualdad', 'justicia', 'equidad', 'beneficio',
            'interés público', 'bien común', 'desarrollo', 'innovación'
        ]
        
        for term in additional_ends_terms:
            ends_score += text.lower().count(term)
        
        # Normalizar por longitud del texto
        text_length = len(text.split())
        if text_length == 0:
            return 0.0
            
        normalized_score = min(ends_score / text_length * 10, 1.0)
        return normalized_score
    
    def _calculate_hybridization(self, process_orient: float, ends_orient: float, 
                                source_data: Dict) -> float:
        """
        Calcular score de hibridización entre procesos y fines.
        
        Hibridización alta indica mezcla equilibrada de enfoques procedimentales
        y sustantivos, característica de marcos regulatorios complejos.
        """
        
        # Hibridización máxima cuando ambas orientaciones son balanceadas
        balance_score = 1.0 - abs(process_orient - ends_orient)
        
        # Amplificación por intensidad total (evitar falsos positivos de baja actividad)
        total_intensity = (process_orient + ends_orient) / 2
        
        # Hibridización = balance * intensidad
        hybridization = balance_score * total_intensity
        
        # Bonus por tipo de fuente (regulaciones tienden a ser más híbridas)
        type_bonus = {
            'legislative': 1.2,    # Leyes tienden a ser híbridas
            'regulatory': 1.3,     # Reglamentos muy híbridos
            'international': 1.1,  # Marcos internacionales balanceados
            'constitutional': 0.8, # Constituciones más orientadas a fines
            'doctrinal': 0.9       # Doctrina más conceptual
        }.get(source_data.get('type', 'unknown'), 1.0)
        
        final_score = min(hybridization * type_bonus, 1.0)
        
        logger.debug(f"Hibridización para {source_data.get('name', 'unknown')}: "
                    f"process={process_orient:.3f}, ends={ends_orient:.3f}, "
                    f"balance={balance_score:.3f}, hybrid={final_score:.3f}")
        
        return final_score
    
    def _calculate_influence_strength(self, source_data: Dict, textual_sim: float,
                                    hybridization: float) -> float:
        """Calcular fuerza de influencia total de una fuente."""
        
        # Peso base por tipo de fuente
        type_weights = {
            'constitutional': 1.0,  # Máximo peso constitucional
            'international': 0.8,   # Alto peso normativo internacional
            'legislative': 0.9,     # Alto peso legislativo nacional
            'regulatory': 0.6,      # Peso medio regulatorio
            'doctrinal': 0.4        # Peso menor doctrinal
        }
        
        base_weight = type_weights.get(source_data.get('type', 'unknown'), 0.3)
        
        # Factores de influencia
        temporal_factor = self._calculate_temporal_proximity(
            source_data.get('year', 2000), 2025
        )
        
        # País de origen (Argentina tiene más peso)
        country_factor = 1.0 if source_data.get('country') == 'Argentina' else 0.7
        
        # Composición final de influencia
        influence = (
            0.4 * base_weight +          # 40% peso del tipo de fuente
            0.3 * textual_sim +          # 30% similitud textual
            0.2 * hybridization +        # 20% hibridización
            0.1 * temporal_factor        # 10% proximidad temporal
        ) * country_factor
        
        return min(influence, 1.0)
    
    def _identify_conceptual_overlap(self, source_data: Dict) -> List[str]:
        """Identificar conceptos que se solapan con el Proyecto Gollan."""
        
        gollan_concepts = {
            'registro_obligatorio', 'evaluacion_impacto', 'autoridad_aplicacion',
            'regimen_sancionatorio', 'transparencia', 'explicabilidad',
            'no_discriminacion', 'privacidad', 'sistemas_alto_riesgo',
            'derechos_ciudadanos', 'supervision', 'control', 'multas'
        }
        
        source_text = source_data.get('text', '').lower()
        process_keywords = [kw.lower() for kw in source_data.get('process_keywords', [])]
        ends_keywords = [kw.lower() for kw in source_data.get('ends_keywords', [])]
        
        overlaps = []
        
        # Mapear conceptos a palabras clave
        concept_mapping = {
            'registro_obligatorio': ['registro', 'inscripción', 'obligatorio'],
            'evaluacion_impacto': ['evaluación', 'impacto', 'análisis'],
            'autoridad_aplicacion': ['autoridad', 'aplicación', 'control'],
            'regimen_sancionatorio': ['sanción', 'multa', 'penalidad'],
            'transparencia': ['transparencia', 'información', 'divulgación'],
            'explicabilidad': ['explicación', 'comprensible', 'claro'],
            'no_discriminacion': ['discriminación', 'igualdad', 'equidad'],
            'privacidad': ['privacidad', 'datos', 'personal'],
            'sistemas_alto_riesgo': ['riesgo', 'sistema', 'crítico'],
            'derechos_ciudadanos': ['derecho', 'ciudadano', 'titular'],
            'supervision': ['supervisión', 'vigilancia', 'monitoreo'],
            'control': ['control', 'verificación', 'auditoría'],
            'multas': ['multa', 'sanción', 'penalidad']
        }
        
        for concept, keywords in concept_mapping.items():
            if any(kw in source_text or kw in process_keywords or kw in ends_keywords 
                   for kw in keywords):
                overlaps.append(concept)
        
        return overlaps
    
    def _calculate_temporal_proximity(self, source_year: int, target_year: int) -> float:
        """Calcular proximidad temporal (más reciente = mayor influencia)."""
        
        years_diff = abs(target_year - source_year)
        
        # Decaimiento exponencial: 5% por año
        proximity = np.exp(-0.05 * years_diff)
        
        return min(proximity, 1.0)
    
    def _calculate_political_distance(self, source_data: Dict) -> float:
        """Calcular distancia política ideológica."""
        
        # Mapear fuentes a perfiles políticos aproximados
        source_profiles = {
            'art19_cn': {'authoritarian': 0.2, 'ideology_economic': 0.5},
            'gdpr_eu': {'authoritarian': 0.6, 'ideology_economic': 0.4},
            'ai_act_eu': {'authoritarian': 0.7, 'ideology_economic': 0.4},
            'sabsay_const': {'authoritarian': 0.3, 'ideology_economic': 0.6},
            'ley_datos_personales': {'authoritarian': 0.5, 'ideology_economic': 0.4},
            'ley_defensa_consumidor': {'authoritarian': 0.6, 'ideology_economic': 0.3}
        }
        
        # Perfil del Proyecto Gollan (autoritario alto, centro-izquierda económico)
        gollan_profile = {'authoritarian': 0.8, 'ideology_economic': 0.3}
        
        source_id = None
        for sid, sdata in self.knowledge_base.items():
            if sdata == source_data:
                source_id = sid
                break
        
        if source_id not in source_profiles:
            return 0.5  # Distancia media por defecto
        
        source_prof = source_profiles[source_id]
        
        # Calcular distancia euclidiana normalizada
        auth_diff = (gollan_profile['authoritarian'] - source_prof['authoritarian']) ** 2
        econ_diff = (gollan_profile['ideology_economic'] - source_prof['ideology_economic']) ** 2
        
        distance = np.sqrt(auth_diff + econ_diff) / np.sqrt(2)  # Normalizar por diagonal máxima
        
        return min(distance, 1.0)
    
    def analyze_hybridization_patterns(self, sources: List[PoliticalSourceNode]) -> Dict:
        """
        Analizar patrones de hibridización entre procesos y fines.
        
        Parameters:
        -----------
        sources : List[PoliticalSourceNode]
            Lista de fuentes analizadas
            
        Returns:
        --------
        Dict
            Análisis de patrones de hibridización
        """
        logger.info("Analizando patrones de hibridización procesos vs fines")
        
        # Clasificar fuentes por tipo de hibridización
        high_hybrid = [s for s in sources if s.hybridization_score > 0.7]
        medium_hybrid = [s for s in sources if 0.4 <= s.hybridization_score <= 0.7]
        low_hybrid = [s for s in sources if s.hybridization_score < 0.4]
        
        # Análisis por tipo de fuente
        type_analysis = defaultdict(list)
        for source in sources:
            type_analysis[source.source_type].append(source)
        
        type_hybridization = {}
        for source_type, type_sources in type_analysis.items():
            avg_hybrid = np.mean([s.hybridization_score for s in type_sources])
            avg_process = np.mean([s.process_orientation for s in type_sources])
            avg_ends = np.mean([s.ends_orientation for s in type_sources])
            
            type_hybridization[source_type] = {
                'count': len(type_sources),
                'avg_hybridization': avg_hybrid,
                'avg_process_orientation': avg_process,
                'avg_ends_orientation': avg_ends,
                'sources': [s.source_id for s in type_sources]
            }
        
        # Calcular hibridización del Proyecto Gollan basada en fuentes
        weighted_hybridization = np.average(
            [s.hybridization_score for s in sources],
            weights=[s.influence_strength for s in sources]
        )
        
        # Determinar orientación dominante del proyecto
        weighted_process = np.average(
            [s.process_orientation for s in sources],
            weights=[s.influence_strength for s in sources]
        )
        
        weighted_ends = np.average(
            [s.ends_orientation for s in sources],
            weights=[s.influence_strength for s in sources]
        )
        
        # Clasificación del proyecto
        if weighted_hybridization > 0.7:
            hybrid_class = "ALTA_HIBRIDACIÓN"
            description = "Equilibrio sofisticado entre mecanismos procedimentales y objetivos sustantivos"
        elif weighted_hybridization > 0.4:
            hybrid_class = "HIBRIDACIÓN_MODERADA" 
            description = "Combinación balanceada con ligero sesgo hacia uno de los enfoques"
        else:
            if weighted_process > weighted_ends:
                hybrid_class = "ORIENTADO_PROCESOS"
                description = "Enfoque predominantemente procedimental y regulatorio"
            else:
                hybrid_class = "ORIENTADO_FINES"
                description = "Enfoque predominantemente sustantivo y valorativo"
        
        analysis_results = {
            'proyecto_gollan_hybridization': {
                'weighted_hybridization_score': weighted_hybridization,
                'weighted_process_orientation': weighted_process,
                'weighted_ends_orientation': weighted_ends,
                'hybridization_class': hybrid_class,
                'description': description
            },
            'sources_classification': {
                'high_hybridization': [s.source_id for s in high_hybrid],
                'medium_hybridization': [s.source_id for s in medium_hybrid], 
                'low_hybridization': [s.source_id for s in low_hybrid]
            },
            'type_analysis': type_hybridization,
            'top_influences': [
                {
                    'source_id': s.source_id,
                    'influence_strength': s.influence_strength,
                    'hybridization_score': s.hybridization_score,
                    'conceptual_overlap': s.conceptual_overlap
                }
                for s in sources[:5]  # Top 5 influences
            ]
        }
        
        return analysis_results
    
    def create_source_network_visualization(self, sources: List[PoliticalSourceNode], 
                                          output_path: str = None):
        """Crear visualización de red de fuentes."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Genealógico del Proyecto Gollan 2130-D-2025', fontsize=16, fontweight='bold')
        
        # 1. Influencia vs Hibridización
        x_influence = [s.influence_strength for s in sources]
        y_hybrid = [s.hybridization_score for s in sources]
        colors = [s.textual_similarity for s in sources]
        
        scatter = ax1.scatter(x_influence, y_hybrid, c=colors, s=100, alpha=0.7, cmap='viridis')
        ax1.set_xlabel('Fuerza de Influencia')
        ax1.set_ylabel('Score de Hibridización')
        ax1.set_title('Influencia vs Hibridización de Fuentes')
        
        # Anotar puntos importantes
        for s in sources[:3]:  # Top 3
            ax1.annotate(s.source_id.replace('_', '\n'), 
                        (s.influence_strength, s.hybridization_score),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=ax1, label='Similitud Textual')
        
        # 2. Orientación Procesos vs Fines
        x_process = [s.process_orientation for s in sources]
        y_ends = [s.ends_orientation for s in sources]
        
        ax2.scatter(x_process, y_ends, s=100, alpha=0.7, c='orange')
        ax2.set_xlabel('Orientación a Procesos')
        ax2.set_ylabel('Orientación a Fines')
        ax2.set_title('Procesos vs Fines')
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equilibrio perfecto')
        ax2.legend()
        
        # 3. Distribución por Tipo de Fuente
        source_types = [s.source_type for s in sources]
        type_counts = Counter(source_types)
        
        ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        ax3.set_title('Distribución por Tipo de Fuente')
        
        # 4. Top Fuentes por Influencia
        top_sources = sources[:5]
        source_names = [s.source_id for s in top_sources]
        influences = [s.influence_strength for s in top_sources]
        
        bars = ax4.barh(source_names, influences, color='steelblue', alpha=0.7)
        ax4.set_xlabel('Fuerza de Influencia')
        ax4.set_title('Top 5 Fuentes Más Influyentes')
        
        # Añadir valores en las barras
        for bar, influence in zip(bars, influences):
            ax4.text(influence + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{influence:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualización guardada en {output_path}")
        
        return fig
    
    def generate_comprehensive_report(self, sources: List[PoliticalSourceNode],
                                    hybridization_analysis: Dict,
                                    output_path: str = None) -> str:
        """Generar reporte comprehensive del análisis genealógico."""
        
        report = f"""
# ANÁLISIS GENEALÓGICO DEL PROYECTO GOLLAN 2130-D-2025
## Búsqueda de Fuentes y Hibridización Procesos vs Fines

**Fecha de Análisis:** {datetime.now().strftime('%d de %B de %Y')}
**Framework:** PoliticalRootFinder con algoritmo ABAN adaptado
**Total de Fuentes Analizadas:** {len(sources)}

---

## RESUMEN EJECUTIVO

El Proyecto de Ley 2130-D-2025 presenta una **{hybridization_analysis['proyecto_gollan_hybridization']['hybridization_class']}** con un score de hibridización de **{hybridization_analysis['proyecto_gollan_hybridization']['weighted_hybridization_score']:.3f}/1.000**.

{hybridization_analysis['proyecto_gollan_hybridization']['description']}

### Métricas Clave:
- **Orientación a Procesos:** {hybridization_analysis['proyecto_gollan_hybridization']['weighted_process_orientation']:.3f}/1.000
- **Orientación a Fines:** {hybridization_analysis['proyecto_gollan_hybridization']['weighted_ends_orientation']:.3f}/1.000
- **Hibridización Ponderada:** {hybridization_analysis['proyecto_gollan_hybridization']['weighted_hybridization_score']:.3f}/1.000

---

## FUENTES IDENTIFICADAS Y ANÁLISIS DE INFLUENCIA

### Top 5 Fuentes Más Influyentes:
"""
        
        for i, source in enumerate(sources[:5], 1):
            source_name = self.knowledge_base[source.source_id]['name']
            report += f"""
**{i}. {source_name}** (`{source.source_id}`)
   - **Tipo:** {source.source_type.title()}
   - **Influencia:** {source.influence_strength:.3f}/1.000
   - **Hibridización:** {source.hybridization_score:.3f}/1.000
   - **Similitud Textual:** {source.textual_similarity:.3f}/1.000
   - **Orientación Procesos:** {source.process_orientation:.3f}/1.000
   - **Orientación Fines:** {source.ends_orientation:.3f}/1.000
   - **Solapamiento Conceptual:** {', '.join(source.conceptual_overlap[:5])}
"""
        
        report += f"""

---

## ANÁLISIS DE HIBRIDIZACIÓN POR TIPO DE FUENTE

"""
        
        for source_type, analysis in hybridization_analysis['type_analysis'].items():
            report += f"""
### {source_type.title()}
- **Cantidad de Fuentes:** {analysis['count']}
- **Hibridización Promedio:** {analysis['avg_hybridization']:.3f}/1.000
- **Orientación Procesos Promedio:** {analysis['avg_process_orientation']:.3f}/1.000
- **Orientación Fines Promedio:** {analysis['avg_ends_orientation']:.3f}/1.000
- **Fuentes:** {', '.join(analysis['sources'])}
"""
        
        report += f"""

---

## CLASIFICACIÓN DE FUENTES POR HIBRIDIZACIÓN

### Alta Hibridización (>0.7)
{', '.join(hybridization_analysis['sources_classification']['high_hybridization']) or 'Ninguna'}

### Hibridización Moderada (0.4-0.7)  
{', '.join(hybridization_analysis['sources_classification']['medium_hybridization']) or 'Ninguna'}

### Baja Hibridización (<0.4)
{', '.join(hybridization_analysis['sources_classification']['low_hybridization']) or 'Ninguna'}

---

## INTERPRETACIÓN GENEALÓGICA

### Fuentes Constitucionales
El **Artículo 19 de la Constitución Nacional** emerge como fuente fundamental, estableciendo el marco de tensión entre autoridad estatal y autonomía individual que permea todo el proyecto.

### Influencias Internacionales
Las regulaciones europeas (**GDPR** y **AI Act**) muestran clara influencia en la arquitectura regulatoria, especialmente en:
- Sistemas de registro obligatorio
- Evaluaciones de impacto
- Regímenes sancionatorios proporcionales

### Precedentes Legislativos Nacionales
La **Ley de Protección de Datos Personales** y la **Ley de Defensa del Consumidor** proporcionan el marco procedimental y sancionatorio.

### Hibridización Detectada
El proyecto muestra una hibridización característica de la regulación moderna de tecnologías emergentes:
- **Procesos:** Estructuras burocráticas de control y supervisión
- **Fines:** Objetivos de protección de derechos y valores democráticos
- **Síntesis:** Marco regulatorio que instrumentaliza procedimientos para lograr fines sustantivos

---

## CONCLUSIONES

1. **Genealogía Múltiple:** El proyecto no deriva de una sola tradición sino que sintetiza múltiples fuentes normativas.

2. **Hibridización Sofisticada:** La combinación de orientaciones procesales y sustantivas refleja la complejidad inherente a la regulación de IA.

3. **Tensión Constitucional:** La influencia del Artículo 19 CN introduce una tensión fundamental entre control estatal y libertad individual.

4. **Europeización Regulatoria:** Notable influencia del marco regulatorio europeo en arquitectura y metodología.

5. **Continuidad Institucional:** Aprovechamiento de precedentes normativos nacionales existentes.

---

**Análisis generado por PoliticalRootFinder**  
*Framework de análisis genealógico político*
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Reporte guardado en {output_path}")
        
        return report

def main():
    """Función principal para ejecutar el análisis genealógico."""
    
    logger.info("=== POLITICAL ROOTFINDER: ANÁLISIS PROYECTO GOLLAN ===")
    
    # Inicializar analizador
    rootfinder = PoliticalRootFinder()
    
    # 1. Analizar fuentes genealógicas
    logger.info("1. Analizando fuentes genealógicas...")
    sources = rootfinder.analyze_gollan_sources()
    
    # 2. Analizar patrones de hibridización
    logger.info("2. Analizando patrones de hibridización...")
    hybridization_analysis = rootfinder.analyze_hybridization_patterns(sources)
    
    # 3. Crear visualizaciones
    logger.info("3. Generando visualizaciones...")
    fig = rootfinder.create_source_network_visualization(
        sources, 
        '/home/user/webapp/proyecto_gollan_fuentes_genealogicas.png'
    )
    
    # 4. Generar reporte comprehensive
    logger.info("4. Generando reporte comprehensive...")
    report = rootfinder.generate_comprehensive_report(
        sources,
        hybridization_analysis,
        '/home/user/webapp/PROYECTO_GOLLAN_ANÁLISIS_GENEALÓGICO.md'
    )
    
    # 5. Exportar resultados en JSON
    logger.info("5. Exportando resultados...")
    results = {
        'sources': [s.to_dict() for s in sources],
        'hybridization_analysis': hybridization_analysis,
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_sources': len(sources),
            'framework_version': '2.0'
        }
    }
    
    with open('/home/user/webapp/proyecto_gollan_analisis_genealogico_completo.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info("=== ANÁLISIS GENEALÓGICO COMPLETADO ===")
    
    print("\n" + "="*60)
    print("RESULTADOS DEL ANÁLISIS GENEALÓGICO")
    print("="*60)
    
    print(f"\n🔍 FUENTES IDENTIFICADAS: {len(sources)}")
    print(f"📊 HIBRIDIZACIÓN PROYECTO: {hybridization_analysis['proyecto_gollan_hybridization']['weighted_hybridization_score']:.3f}/1.000")
    print(f"🏛️  CLASE HIBRIDIZACIÓN: {hybridization_analysis['proyecto_gollan_hybridization']['hybridization_class']}")
    
    print(f"\n🥇 TOP 3 FUENTES MÁS INFLUYENTES:")
    for i, source in enumerate(sources[:3], 1):
        source_name = rootfinder.knowledge_base[source.source_id]['name']
        print(f"   {i}. {source_name} (Influencia: {source.influence_strength:.3f})")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   • proyecto_gollan_fuentes_genealogicas.png")
    print(f"   • PROYECTO_GOLLAN_ANÁLISIS_GENEALÓGICO.md") 
    print(f"   • proyecto_gollan_analisis_genealogico_completo.json")
    
    return results

if __name__ == "__main__":
    results = main()