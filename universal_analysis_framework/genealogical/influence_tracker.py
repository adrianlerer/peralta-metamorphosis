"""
Universal Genealogical Analysis Module for Influence Tracking
Sistema universal de análisis genealógico aplicable a cualquier dominio.

Rastrea influencias, dependencias, orígenes y relaciones ancestrales
en cualquier tipo de análisis o toma de decisiones.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict, deque
import json
import hashlib

class InfluenceType(Enum):
    """Tipos de influencia en el análisis genealógico"""
    DIRECT_CAUSAL = "direct_causal"
    INDIRECT_CAUSAL = "indirect_causal"
    CORRELATIONAL = "correlational"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"
    CONCEPTUAL = "conceptual"
    METHODOLOGICAL = "methodological"
    DATA_DEPENDENCY = "data_dependency"

class NodeType(Enum):
    """Tipos de nodos en el grafo genealógico"""
    INPUT_DATA = "input_data"
    PROCESSING_STEP = "processing_step"
    INTERMEDIATE_RESULT = "intermediate_result"
    FINAL_RESULT = "final_result"
    EXTERNAL_SOURCE = "external_source"
    MODEL = "model"
    DECISION_POINT = "decision_point"
    VALIDATION_STEP = "validation_step"

@dataclass
class InfluenceRelation:
    """Representa una relación de influencia entre dos elementos"""
    source_id: str
    target_id: str
    influence_type: InfluenceType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "influence_type": self.influence_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class GenealogyNode:
    """Nodo en el grafo genealógico"""
    node_id: str
    node_type: NodeType
    content: Any
    importance: float = 1.0
    processing_stage: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": str(self.content),  # Serializable representation
            "importance": self.importance,
            "processing_stage": self.processing_stage,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class GenealogyAnalysis:
    """Resultado completo del análisis genealógico"""
    analysis_id: str
    nodes: Dict[str, GenealogyNode] = field(default_factory=dict)
    relations: List[InfluenceRelation] = field(default_factory=list)
    influence_metrics: Dict[str, float] = field(default_factory=dict)
    ancestry_paths: Dict[str, List[str]] = field(default_factory=dict)
    centrality_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "analysis_id": self.analysis_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "relations": [r.to_dict() for r in self.relations],
            "influence_metrics": self.influence_metrics,
            "ancestry_paths": self.ancestry_paths,
            "centrality_metrics": self.centrality_metrics,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }

class UniversalInfluenceTracker:
    """Rastreador universal de influencias aplicable a cualquier dominio"""
    
    def __init__(self, analysis_id: Optional[str] = None):
        self.analysis_id = analysis_id or self._generate_analysis_id()
        self.nodes: Dict[str, GenealogyNode] = {}
        self.relations: List[InfluenceRelation] = []
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger("UniversalInfluenceTracker")
        
    def _generate_analysis_id(self) -> str:
        """Genera ID único para el análisis"""
        return hashlib.md5(f"genealogy_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        content: Any,
        importance: float = 1.0,
        processing_stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Añade nodo al grafo genealógico
        
        Returns:
            str: ID del nodo añadido
        """
        node = GenealogyNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            importance=importance,
            processing_stage=processing_stage,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        
        self.logger.debug(f"Nodo añadido: {node_id} ({node_type.value})")
        return node_id
    
    def add_influence(
        self,
        source_id: str,
        target_id: str,
        influence_type: InfluenceType,
        strength: float,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Añade relación de influencia entre dos nodos"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Nodos no encontrados: {source_id}, {target_id}")
        
        relation = InfluenceRelation(
            source_id=source_id,
            target_id=target_id,
            influence_type=influence_type,
            strength=strength,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.relations.append(relation)
        self.graph.add_edge(
            source_id, 
            target_id, 
            influence_type=influence_type.value,
            strength=strength,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.logger.debug(f"Influencia añadida: {source_id} -> {target_id} ({influence_type.value})")
    
    def track_processing_step(
        self,
        step_name: str,
        input_data: Any,
        output_data: Any,
        processing_function: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, str]:
        """
        Rastrea un paso de procesamiento completo
        
        Returns:
            Tuple[str, str, str]: (input_node_id, process_node_id, output_node_id)
        """
        # Crear nodos
        input_id = f"input_{step_name}_{self._get_content_hash(input_data)}"
        process_id = f"process_{step_name}"
        output_id = f"output_{step_name}_{self._get_content_hash(output_data)}"
        
        # Añadir nodos si no existen
        if input_id not in self.nodes:
            self.add_node(input_id, NodeType.INPUT_DATA, input_data, processing_stage=step_name)
        
        if process_id not in self.nodes:
            self.add_node(
                process_id, 
                NodeType.PROCESSING_STEP, 
                processing_function or step_name,
                processing_stage=step_name,
                metadata=metadata
            )
        
        if output_id not in self.nodes:
            self.add_node(output_id, NodeType.INTERMEDIATE_RESULT, output_data, processing_stage=step_name)
        
        # Añadir influencias
        self.add_influence(input_id, process_id, InfluenceType.DATA_DEPENDENCY, 1.0)
        self.add_influence(process_id, output_id, InfluenceType.DIRECT_CAUSAL, 1.0)
        
        return input_id, process_id, output_id
    
    def track_model_prediction(
        self,
        model_id: str,
        input_data: Any,
        prediction: Any,
        confidence: float,
        model_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, str]:
        """Rastrea predicción de un modelo"""
        input_id = f"model_input_{model_id}_{self._get_content_hash(input_data)}"
        model_node_id = f"model_{model_id}"
        prediction_id = f"prediction_{model_id}_{self._get_content_hash(prediction)}"
        
        # Añadir nodos
        if input_id not in self.nodes:
            self.add_node(input_id, NodeType.INPUT_DATA, input_data)
        
        if model_node_id not in self.nodes:
            model_metadata = {"model_type": model_type, **(metadata or {})}
            self.add_node(model_node_id, NodeType.MODEL, model_id, metadata=model_metadata)
        
        if prediction_id not in self.nodes:
            pred_metadata = {"model_confidence": confidence, **(metadata or {})}
            self.add_node(prediction_id, NodeType.INTERMEDIATE_RESULT, prediction, metadata=pred_metadata)
        
        # Añadir influencias
        self.add_influence(input_id, model_node_id, InfluenceType.DATA_DEPENDENCY, 1.0)
        self.add_influence(model_node_id, prediction_id, InfluenceType.DIRECT_CAUSAL, confidence)
        
        return input_id, model_node_id, prediction_id
    
    def track_decision_point(
        self,
        decision_name: str,
        inputs: List[str],
        decision_result: Any,
        decision_logic: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Rastrea un punto de decisión"""
        decision_id = f"decision_{decision_name}"
        result_id = f"decision_result_{decision_name}_{self._get_content_hash(decision_result)}"
        
        # Añadir nodos
        decision_metadata = {"decision_logic": decision_logic, **(metadata or {})}
        self.add_node(decision_id, NodeType.DECISION_POINT, decision_name, metadata=decision_metadata)
        self.add_node(result_id, NodeType.INTERMEDIATE_RESULT, decision_result)
        
        # Añadir influencias de inputs
        for input_id in inputs:
            if input_id in self.nodes:
                self.add_influence(input_id, decision_id, InfluenceType.DIRECT_CAUSAL, confidence)
        
        # Influencia de decisión a resultado
        self.add_influence(decision_id, result_id, InfluenceType.DIRECT_CAUSAL, confidence)
        
        return result_id
    
    def _get_content_hash(self, content: Any) -> str:
        """Genera hash del contenido para IDs únicos"""
        content_str = str(content)[:100]  # Limitar longitud
        return hashlib.md5(content_str.encode()).hexdigest()[:8]
    
    def analyze_genealogy(self) -> GenealogyAnalysis:
        """Realiza análisis completo de la genealogía de influencias"""
        # Calcular métricas de centralidad
        centrality_metrics = self._calculate_centrality_metrics()
        
        # Calcular métricas de influencia
        influence_metrics = self._calculate_influence_metrics()
        
        # Encontrar caminos de ancestría
        ancestry_paths = self._find_ancestry_paths()
        
        # Metadatos del análisis
        metadata = {
            "num_nodes": len(self.nodes),
            "num_relations": len(self.relations),
            "graph_density": nx.density(self.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "strongly_connected_components": len(list(nx.strongly_connected_components(self.graph)))
        }
        
        return GenealogyAnalysis(
            analysis_id=self.analysis_id,
            nodes=self.nodes.copy(),
            relations=self.relations.copy(),
            influence_metrics=influence_metrics,
            ancestry_paths=ancestry_paths,
            centrality_metrics=centrality_metrics,
            metadata=metadata
        )
    
    def _calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calcula métricas de centralidad para todos los nodos"""
        metrics = {}
        
        if len(self.graph.nodes()) == 0:
            return metrics
        
        try:
            # Centralidad de grado (in/out degree)
            in_degree_centrality = nx.in_degree_centrality(self.graph)
            out_degree_centrality = nx.out_degree_centrality(self.graph)
            
            # Centralidad de cercanía
            try:
                closeness_centrality = nx.closeness_centrality(self.graph)
            except:
                closeness_centrality = {node: 0.0 for node in self.graph.nodes()}
            
            # Centralidad de intermediación
            try:
                betweenness_centrality = nx.betweenness_centrality(self.graph)
            except:
                betweenness_centrality = {node: 0.0 for node in self.graph.nodes()}
            
            # PageRank (como medida de importancia)
            try:
                pagerank = nx.pagerank(self.graph)
            except:
                pagerank = {node: 1.0/len(self.graph.nodes()) for node in self.graph.nodes()}
            
            # Combinar métricas
            for node in self.graph.nodes():
                metrics[node] = {
                    "in_degree_centrality": in_degree_centrality.get(node, 0.0),
                    "out_degree_centrality": out_degree_centrality.get(node, 0.0),
                    "closeness_centrality": closeness_centrality.get(node, 0.0),
                    "betweenness_centrality": betweenness_centrality.get(node, 0.0),
                    "pagerank": pagerank.get(node, 0.0)
                }
                
        except Exception as e:
            self.logger.error(f"Error calculando centralidad: {str(e)}")
        
        return metrics
    
    def _calculate_influence_metrics(self) -> Dict[str, float]:
        """Calcula métricas globales de influencia"""
        metrics = {}
        
        if not self.relations:
            return metrics
        
        # Fuerza de influencia promedio
        strengths = [r.strength for r in self.relations]
        metrics["average_influence_strength"] = np.mean(strengths)
        metrics["max_influence_strength"] = np.max(strengths)
        metrics["min_influence_strength"] = np.min(strengths)
        metrics["influence_strength_std"] = np.std(strengths)
        
        # Confianza promedio
        confidences = [r.confidence for r in self.relations]
        metrics["average_confidence"] = np.mean(confidences)
        metrics["min_confidence"] = np.min(confidences)
        
        # Distribución por tipo de influencia
        type_counts = defaultdict(int)
        for relation in self.relations:
            type_counts[relation.influence_type.value] += 1
        
        total_relations = len(self.relations)
        for influence_type, count in type_counts.items():
            metrics[f"{influence_type}_ratio"] = count / total_relations
        
        # Métricas de conectividad
        if len(self.graph.nodes()) > 0:
            metrics["graph_connectivity"] = nx.node_connectivity(self.graph.to_undirected())
            metrics["average_clustering"] = nx.average_clustering(self.graph.to_undirected())
        
        return metrics
    
    def _find_ancestry_paths(self) -> Dict[str, List[str]]:
        """Encuentra caminos de ancestría para nodos importantes"""
        ancestry_paths = {}
        
        # Encontrar nodos finales (sin sucesores)
        final_nodes = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
        
        for final_node in final_nodes:
            # Encontrar todos los caminos desde nodos iniciales hasta este nodo final
            initial_nodes = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
            
            paths = []
            for initial_node in initial_nodes:
                try:
                    if nx.has_path(self.graph, initial_node, final_node):
                        # Encontrar el camino más corto
                        path = nx.shortest_path(self.graph, initial_node, final_node)
                        paths.append(path)
                except:
                    continue
            
            if paths:
                ancestry_paths[final_node] = paths
        
        return ancestry_paths
    
    def get_node_ancestry(self, node_id: str, max_depth: int = 10) -> List[List[str]]:
        """Obtiene ancestría completa de un nodo específico"""
        if node_id not in self.nodes:
            return []
        
        ancestry_paths = []
        
        def dfs_ancestry(current_node: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            predecessors = list(self.graph.predecessors(current_node))
            
            if not predecessors:
                # Nodo raíz encontrado
                ancestry_paths.append(path[::-1])  # Revertir para orden cronológico
            else:
                for pred in predecessors:
                    if pred not in path:  # Evitar ciclos
                        dfs_ancestry(pred, path + [pred], depth + 1)
        
        dfs_ancestry(node_id, [node_id], 0)
        return ancestry_paths
    
    def get_influence_chain(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """Obtiene cadena de influencia entre dos nodos"""
        if not nx.has_path(self.graph, source_id, target_id):
            return []
        
        path = nx.shortest_path(self.graph, source_id, target_id)
        influence_chain = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Encontrar la relación de influencia
            edge_data = self.graph.get_edge_data(current, next_node)
            
            # Encontrar la relación original
            matching_relation = None
            for relation in self.relations:
                if relation.source_id == current and relation.target_id == next_node:
                    matching_relation = relation
                    break
            
            influence_step = {
                "step": i + 1,
                "source": current,
                "target": next_node,
                "source_type": self.nodes[current].node_type.value,
                "target_type": self.nodes[next_node].node_type.value,
                "influence_type": edge_data.get("influence_type", "unknown") if edge_data else "unknown",
                "strength": edge_data.get("strength", 0.0) if edge_data else 0.0,
                "confidence": edge_data.get("confidence", 0.0) if edge_data else 0.0,
                "relation": matching_relation.to_dict() if matching_relation else None
            }
            
            influence_chain.append(influence_step)
        
        return influence_chain
    
    def find_critical_influences(self, importance_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Encuentra influencias críticas en el análisis"""
        critical_influences = []
        
        # Analizar cada relación
        for relation in self.relations:
            criticality_score = 0.0
            
            # Factor 1: Fuerza de la influencia
            criticality_score += 0.3 * relation.strength
            
            # Factor 2: Confianza en la relación
            criticality_score += 0.2 * relation.confidence
            
            # Factor 3: Importancia de los nodos involucrados
            source_importance = self.nodes[relation.source_id].importance
            target_importance = self.nodes[relation.target_id].importance
            criticality_score += 0.3 * (source_importance + target_importance) / 2
            
            # Factor 4: Tipo de influencia (algunas son más críticas)
            type_weights = {
                InfluenceType.DIRECT_CAUSAL: 1.0,
                InfluenceType.DATA_DEPENDENCY: 0.9,
                InfluenceType.METHODOLOGICAL: 0.8,
                InfluenceType.INDIRECT_CAUSAL: 0.7,
                InfluenceType.STRUCTURAL: 0.6,
                InfluenceType.TEMPORAL: 0.5,
                InfluenceType.CORRELATIONAL: 0.4,
                InfluenceType.CONCEPTUAL: 0.3
            }
            criticality_score += 0.2 * type_weights.get(relation.influence_type, 0.5)
            
            if criticality_score >= importance_threshold:
                critical_influences.append({
                    "relation": relation.to_dict(),
                    "criticality_score": criticality_score,
                    "source_node": self.nodes[relation.source_id].to_dict(),
                    "target_node": self.nodes[relation.target_id].to_dict()
                })
        
        # Ordenar por criticidad
        critical_influences.sort(key=lambda x: x["criticality_score"], reverse=True)
        
        return critical_influences
    
    def export_genealogy_graph(self, format: str = "gexf") -> str:
        """Exporta el grafo genealógico en el formato especificado"""
        if format.lower() == "gexf":
            return nx.write_gexf(self.graph)
        elif format.lower() == "json":
            return json.dumps(nx.node_link_data(self.graph), indent=2)
        elif format.lower() == "graphml":
            return nx.write_graphml(self.graph)
        else:
            raise ValueError(f"Formato no soportado: {format}")

# Instancia global del rastreador de influencias
universal_influence_tracker = UniversalInfluenceTracker()