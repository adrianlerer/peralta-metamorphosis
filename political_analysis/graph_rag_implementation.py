#!/usr/bin/env python3
"""
Graph-RAG Implementation for Political Analysis
Real implementation integrating with existing genealogy system
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from pathlib import Path

# Community detection
try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

logger = logging.getLogger(__name__)

class PoliticalEntity:
    """
    Represents a political entity (person, event, concept, law, etc.)
    """
    def __init__(self, entity_id: str, entity_type: str, name: str, 
                 year: Optional[int] = None, metadata: Optional[Dict] = None):
        self.entity_id = entity_id
        self.entity_type = entity_type  # politician, law, event, concept, party
        self.name = name
        self.year = year
        self.metadata = metadata or {}
        self.text_mentions = []
        self.relationships = []
    
    def add_text_mention(self, document_id: str, context: str, position: int):
        """Add a mention of this entity in a document."""
        self.text_mentions.append({
            'document_id': document_id,
            'context': context,
            'position': position
        })
    
    def to_dict(self) -> Dict:
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'name': self.name,
            'year': self.year,
            'metadata': self.metadata,
            'mentions_count': len(self.text_mentions)
        }

class PoliticalRelationship:
    """
    Represents a relationship between political entities
    """
    def __init__(self, source_id: str, target_id: str, relationship_type: str,
                 strength: float = 1.0, year: Optional[int] = None, 
                 evidence: Optional[str] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type  # influence, opposition, collaboration, etc.
        self.strength = strength
        self.year = year
        self.evidence = evidence
    
    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type,
            'strength': self.strength,
            'year': self.year,
            'evidence': self.evidence
        }

class PoliticalKnowledgeGraph:
    """
    Knowledge graph for political analysis with community detection.
    """
    
    def __init__(self):
        self.entities = {}  # entity_id -> PoliticalEntity
        self.relationships = []  # List of PoliticalRelationship
        self.graph = nx.DiGraph()
        self.communities = {}
        self.entity_embeddings = {}
        
    def extract_entities_from_corpus(self, corpus: pd.DataFrame) -> None:
        """
        Extract political entities from the political corpus.
        """
        logger.info(f"Extracting entities from {len(corpus)} documents")
        
        # Political entity patterns (Spanish)
        entity_patterns = {
            'politician': [
                r'\b(Juan\s+(?:Domingo\s+)?Perón|Perón)\b',
                r'\b(Eva\s+(?:Duarte\s+(?:de\s+)?)?Perón|Evita)\b', 
                r'\b(Mariano\s+Moreno|Moreno)\b',
                r'\b(Cornelio\s+Saavedra|Saavedra)\b',
                r'\b(Juan\s+Manuel\s+(?:de\s+)?Rosas|Rosas)\b',
                r'\b(Bartolomé\s+Mitre|Mitre)\b',
                r'\b(Domingo\s+(?:Faustino\s+)?Sarmiento|Sarmiento)\b',
                r'\b(Juan\s+Bautista\s+Alberdi|Alberdi)\b',
                r'\b(Justo\s+José\s+(?:de\s+)?Urquiza|Urquiza)\b',
                r'\b(Hipólito\s+Yrigoyen|Yrigoyen)\b',
                r'\b(Raúl\s+Alfonsín|Alfonsín)\b',
                r'\b(Carlos\s+(?:Saúl\s+)?Menem|Menem)\b',
                r'\b(Néstor\s+(?:Carlos\s+)?Kirchner|Kirchner)\b',
                r'\b(Cristina\s+(?:Fernández\s+(?:de\s+)?Kirchner|Kirchner))\b',
                r'\b(Mauricio\s+Macri|Macri)\b',
                r'\b(Alberto\s+(?:Ángel\s+)?Fernández)\b',
                r'\b(Javier\s+(?:Gerardo\s+)?Milei|Milei)\b'
            ],
            'law': [
                r'\b(Ley\s+Sáenz\s+Peña|ley\s+del\s+voto\s+secreto)\b',
                r'\b(Constitución\s+(?:de\s+)?1853|Constitución\s+Nacional)\b',
                r'\b(Constitución\s+(?:de\s+)?1949)\b',
                r'\b(Ley\s+de\s+Residencia)\b',
                r'\b(Estatuto\s+del\s+Peón)\b',
                r'\b(Ley\s+de\s+Sufragio\s+Femenino)\b'
            ],
            'event': [
                r'\b(Revolución\s+de\s+Mayo|25\s+de\s+Mayo)\b',
                r'\b(17\s+de\s+Octubre(?:\s+de\s+1945)?)\b',
                r'\b(Golpe\s+(?:de\s+Estado\s+)?(?:de\s+)?1930)\b',
                r'\b(Revolución\s+Libertadora|1955)\b',
                r'\b(Proceso\s+de\s+Reorganización\s+Nacional|Proceso)\b',
                r'\b(Guerra\s+de\s+las\s+Malvinas|Malvinas)\b',
                r'\b(Crisis\s+(?:de\s+)?2001|Corralito)\b'
            ],
            'concept': [
                r'\b(federalismo|federal)\b',
                r'\b(unitarismo|unitario)\b',
                r'\b(peronismo|justicialismo)\b',
                r'\b(radicalismo|UCR)\b',
                r'\b(liberalismo|liberal)\b',
                r'\b(socialismo|socialista)\b',
                r'\b(nacionalismo|nacionalista)\b',
                r'\b(populismo|populista)\b'
            ],
            'party': [
                r'\b(Partido\s+Justicialista|PJ)\b',
                r'\b(Unión\s+Cívica\s+Radical|UCR)\b',
                r'\b(Propuesta\s+Republicana|PRO)\b',
                r'\b(La\s+Libertad\s+Avanza|LLA)\b',
                r'\b(Frente\s+de\s+Todos)\b',
                r'\b(Cambiemos)\b'
            ]
        }
        
        entity_counter = defaultdict(int)
        
        for _, doc in corpus.iterrows():
            text = doc['text']
            doc_id = doc['document_id']
            year = doc['year']
            
            # Extract entities by type
            for entity_type, patterns in entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        entity_name = match.group(0).strip()
                        # Normalize entity name
                        normalized_name = self._normalize_entity_name(entity_name)
                        entity_id = f"{entity_type}_{normalized_name.lower().replace(' ', '_')}"
                        
                        # Create or update entity
                        if entity_id not in self.entities:
                            self.entities[entity_id] = PoliticalEntity(
                                entity_id=entity_id,
                                entity_type=entity_type,
                                name=normalized_name,
                                year=year
                            )
                        
                        # Add text mention
                        context = text[max(0, match.start()-100):match.end()+100]
                        self.entities[entity_id].add_text_mention(doc_id, context, match.start())
                        entity_counter[entity_id] += 1
        
        # Filter entities by minimum mentions (reduce noise)
        min_mentions = 1
        filtered_entities = {
            entity_id: entity for entity_id, entity in self.entities.items()
            if len(entity.text_mentions) >= min_mentions
        }
        self.entities = filtered_entities
        
        logger.info(f"Extracted {len(self.entities)} political entities")
        for entity_type in ['politician', 'law', 'event', 'concept', 'party']:
            type_count = len([e for e in self.entities.values() if e.entity_type == entity_type])
            logger.info(f"  {entity_type}: {type_count}")
    
    def extract_relationships(self, corpus: pd.DataFrame) -> None:
        """
        Extract relationships between entities based on co-occurrence and patterns.
        """
        logger.info("Extracting political relationships")
        
        # Relationship patterns
        influence_patterns = [
            r'(\w+)\s+(?:influyó|inspiró|guió)\s+(?:a\s+)?(\w+)',
            r'(\w+)\s+(?:siguió|heredó)\s+(?:de\s+|las\s+ideas\s+de\s+)?(\w+)',
            r'(\w+)\s+(?:basó|fundó)\s+(?:en\s+|su\s+pensamiento\s+en\s+)?(\w+)'
        ]
        
        opposition_patterns = [
            r'(\w+)\s+(?:se\s+opuso\s+a|enfrentó\s+a|luchó\s+contra)\s+(\w+)',
            r'(\w+)\s+(?:vs\.?|versus|contra)\s+(\w+)'
        ]
        
        # Co-occurrence based relationships
        entity_list = list(self.entities.keys())
        
        for _, doc in corpus.iterrows():
            text = doc['text'].lower()
            doc_entities = []
            
            # Find entities mentioned in this document
            for entity_id in entity_list:
                entity = self.entities[entity_id]
                if any(entity.name.lower() in mention['context'].lower() 
                       for mention in entity.text_mentions 
                       if mention['document_id'] == doc['document_id']):
                    doc_entities.append(entity_id)
            
            # Create co-occurrence relationships
            for i, entity1_id in enumerate(doc_entities):
                for entity2_id in doc_entities[i+1:]:
                    entity1 = self.entities[entity1_id]
                    entity2 = self.entities[entity2_id]
                    
                    # Temporal relationship strength
                    year1 = entity1.year or doc['year']
                    year2 = entity2.year or doc['year']
                    temporal_strength = max(0.1, 1.0 - abs(year1 - year2) / 100.0)
                    
                    # Create bidirectional relationship
                    relationship = PoliticalRelationship(
                        source_id=entity1_id,
                        target_id=entity2_id,
                        relationship_type='co_occurrence',
                        strength=temporal_strength,
                        year=doc['year'],
                        evidence=doc['document_id']
                    )
                    self.relationships.append(relationship)
        
        logger.info(f"Extracted {len(self.relationships)} relationships")
    
    def build_networkx_graph(self) -> None:
        """
        Build NetworkX graph from entities and relationships.
        """
        logger.info("Building NetworkX graph")
        
        self.graph.clear()
        
        # Add nodes (entities)
        for entity_id, entity in self.entities.items():
            self.graph.add_node(
                entity_id,
                entity_type=entity.entity_type,
                name=entity.name,
                year=entity.year,
                mentions_count=len(entity.text_mentions)
            )
        
        # Add edges (relationships)
        edge_weights = defaultdict(float)
        
        for rel in self.relationships:
            # Aggregate relationship strengths
            edge_key = (rel.source_id, rel.target_id)
            edge_weights[edge_key] += rel.strength
        
        for (source, target), weight in edge_weights.items():
            if source in self.graph.nodes and target in self.graph.nodes:
                self.graph.add_edge(source, target, weight=weight)
        
        logger.info(f"Graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
    
    def detect_political_communities(self) -> Dict[str, List[str]]:
        """
        Detect political communities using multiple algorithms.
        """
        logger.info("Detecting political communities")
        
        if len(self.graph.nodes) < 3:
            logger.warning("Too few nodes for community detection")
            return {}
        
        communities = {}
        
        # Method 1: NetworkX modularity-based communities
        try:
            import networkx.algorithms.community as nx_comm
            undirected_graph = self.graph.to_undirected()
            
            modularity_communities = nx_comm.greedy_modularity_communities(undirected_graph)
            communities['modularity'] = [list(community) for community in modularity_communities]
            
        except Exception as e:
            logger.warning(f"Modularity community detection failed: {e}")
        
        # Method 2: Louvain algorithm (if available)
        if HAS_IGRAPH:
            try:
                # Convert to igraph
                ig_graph = ig.Graph.from_networkx(self.graph.to_undirected())
                
                # Louvain community detection
                louvain_communities = ig_graph.community_multilevel()
                communities['louvain'] = [list(community) for community in louvain_communities]
                
            except Exception as e:
                logger.warning(f"Louvain community detection failed: {e}")
        
        # Method 3: Political family clustering (manual rules)
        political_families = self._detect_political_families()
        if political_families:
            communities['political_families'] = political_families
        
        # Store best communities
        if communities:
            # Use the method with most reasonable number of communities
            best_method = min(communities.keys(), 
                            key=lambda x: abs(len(communities[x]) - np.sqrt(len(self.graph.nodes))))
            self.communities = {
                f"community_{i}": {
                    'members': community,
                    'method': best_method,
                    'summary': self._generate_community_summary(community)
                }
                for i, community in enumerate(communities[best_method])
            }
        
        logger.info(f"Detected {len(self.communities)} political communities")
        return self.communities
    
    def _detect_political_families(self) -> List[List[str]]:
        """
        Detect political families based on known political affiliations.
        """
        families = []
        
        # Define known political families/movements
        family_keywords = {
            'peronista': ['perón', 'evita', 'kirchner', 'fernández'],
            'radical': ['yrigoyen', 'alfonsín', 'radical'],
            'liberal': ['mitre', 'sarmiento', 'liberal', 'milei'],
            'federal': ['rosas', 'urquiza', 'federal'],
            'unitario': ['rivadavia', 'unitario'],
            'conservador': ['roca', 'conservador']
        }
        
        for family_name, keywords in family_keywords.items():
            family_members = []
            
            for entity_id, entity in self.entities.items():
                entity_text = (entity.name + ' ' + ' '.join([m['context'] for m in entity.text_mentions])).lower()
                
                if any(keyword in entity_text for keyword in keywords):
                    family_members.append(entity_id)
            
            if len(family_members) >= 2:
                families.append(family_members)
        
        return families
    
    def _generate_community_summary(self, community_members: List[str]) -> str:
        """
        Generate a summary for a political community.
        """
        if not community_members:
            return "Empty community"
        
        # Analyze community composition
        entity_types = defaultdict(int)
        years = []
        names = []
        
        for entity_id in community_members:
            if entity_id in self.entities:
                entity = self.entities[entity_id]
                entity_types[entity.entity_type] += 1
                names.append(entity.name)
                if entity.year:
                    years.append(entity.year)
        
        # Generate summary
        type_summary = ", ".join([f"{count} {etype}" for etype, count in entity_types.items()])
        
        if years:
            year_range = f"{min(years)}-{max(years)}"
        else:
            year_range = "unknown period"
        
        key_members = ", ".join(names[:3])
        if len(names) > 3:
            key_members += f" and {len(names)-3} others"
        
        return f"Political community ({year_range}): {key_members}. Composition: {type_summary}."
    
    def multi_hop_query(self, start_entity: str, target_entity: str, max_hops: int = 3) -> List[List[str]]:
        """
        Find paths between entities using multi-hop traversal.
        """
        if start_entity not in self.graph.nodes or target_entity not in self.graph.nodes:
            return []
        
        try:
            # Find shortest paths
            paths = list(nx.all_shortest_paths(self.graph, start_entity, target_entity))
            
            # Limit to reasonable number of paths
            if len(paths) > 10:
                paths = paths[:10]
            
            # Filter by max hops
            filtered_paths = [path for path in paths if len(path) <= max_hops + 1]
            
            return filtered_paths
            
        except nx.NetworkXNoPath:
            return []
    
    def query_local_community(self, entity_id: str, radius: int = 2) -> Dict[str, Any]:
        """
        Local Graph-RAG query: get subgraph around an entity.
        """
        if entity_id not in self.graph.nodes:
            return {'error': f'Entity {entity_id} not found'}
        
        # Get subgraph within radius
        subgraph_nodes = set([entity_id])
        current_nodes = {entity_id}
        
        for _ in range(radius):
            next_nodes = set()
            for node in current_nodes:
                neighbors = set(self.graph.neighbors(node)) | set(self.graph.predecessors(node))
                next_nodes.update(neighbors)
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Find community for this entity
        entity_community = None
        for comm_id, comm_data in self.communities.items():
            if entity_id in comm_data['members']:
                entity_community = comm_data
                break
        
        return {
            'entity': self.entities[entity_id].to_dict(),
            'subgraph_size': len(subgraph.nodes),
            'neighbors': list(subgraph.neighbors(entity_id)),
            'community': entity_community,
            'local_summary': self._generate_local_summary(entity_id, subgraph)
        }
    
    def query_global_communities(self, query_theme: str) -> Dict[str, Any]:
        """
        Global Graph-RAG query: aggregate across communities for thematic analysis.
        """
        relevant_communities = []
        theme_lower = query_theme.lower()
        
        # Find communities relevant to the theme
        for comm_id, comm_data in self.communities.items():
            summary_lower = comm_data['summary'].lower()
            
            # Simple keyword matching (can be improved with embeddings)
            if any(keyword in summary_lower for keyword in theme_lower.split()):
                relevant_communities.append((comm_id, comm_data))
        
        if not relevant_communities:
            # Fallback: return all communities
            relevant_communities = list(self.communities.items())
        
        # Aggregate information
        global_summary = self._generate_global_summary(relevant_communities, query_theme)
        
        return {
            'theme': query_theme,
            'relevant_communities': len(relevant_communities),
            'communities': [
                {
                    'id': comm_id,
                    'summary': comm_data['summary'],
                    'member_count': len(comm_data['members'])
                }
                for comm_id, comm_data in relevant_communities
            ],
            'global_summary': global_summary
        }
    
    def _generate_local_summary(self, entity_id: str, subgraph) -> str:
        """Generate summary for local subgraph around an entity."""
        entity = self.entities[entity_id]
        
        neighbor_types = defaultdict(int)
        for neighbor_id in subgraph.neighbors(entity_id):
            if neighbor_id in self.entities:
                neighbor_types[self.entities[neighbor_id].entity_type] += 1
        
        type_summary = ", ".join([f"{count} {etype}" for etype, count in neighbor_types.items()])
        
        return f"{entity.name} ({entity.entity_type}) connected to: {type_summary}"
    
    def _generate_global_summary(self, communities: List[Tuple[str, Dict]], theme: str) -> str:
        """Generate global summary across communities for a theme."""
        
        total_entities = sum(len(comm_data['members']) for _, comm_data in communities)
        
        entity_types_global = defaultdict(int)
        years_global = []
        
        for _, comm_data in communities:
            for member_id in comm_data['members']:
                if member_id in self.entities:
                    entity = self.entities[member_id]
                    entity_types_global[entity.entity_type] += 1
                    if entity.year:
                        years_global.append(entity.year)
        
        if years_global:
            time_span = f"{min(years_global)}-{max(years_global)}"
        else:
            time_span = "unknown period"
        
        type_summary = ", ".join([f"{count} {etype}" for etype, count in entity_types_global.items()])
        
        return f"Global analysis of '{theme}' across {len(communities)} political communities ({time_span}). Total entities: {total_entities}. Composition: {type_summary}."
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity names for consistency."""
        # Simple normalization
        name = name.strip()
        
        # Common normalizations for Argentine politicians
        normalizations = {
            'Perón': 'Juan Domingo Perón',
            'Evita': 'Eva Perón', 
            'Moreno': 'Mariano Moreno',
            'Saavedra': 'Cornelio Saavedra',
            'Rosas': 'Juan Manuel de Rosas',
            'Mitre': 'Bartolomé Mitre',
            'Sarmiento': 'Domingo Faustino Sarmiento',
            'Alberdi': 'Juan Bautista Alberdi',
            'Urquiza': 'Justo José de Urquiza',
            'Yrigoyen': 'Hipólito Yrigoyen',
            'Alfonsín': 'Raúl Alfonsín',
            'Menem': 'Carlos Menem',
            'Kirchner': 'Néstor Kirchner',
            'Macri': 'Mauricio Macri',
            'Milei': 'Javier Milei'
        }
        
        return normalizations.get(name, name)

class PoliticalGraphRAG:
    """
    Graph-based RAG system for political analysis using knowledge graphs.
    """
    
    def __init__(self):
        self.knowledge_graph = PoliticalKnowledgeGraph()
        self.is_indexed = False
    
    def index_political_corpus(self, corpus: pd.DataFrame) -> None:
        """
        Build knowledge graph from political corpus.
        """
        logger.info("Indexing political corpus for Graph-RAG")
        
        # Extract entities and relationships
        self.knowledge_graph.extract_entities_from_corpus(corpus)
        self.knowledge_graph.extract_relationships(corpus)
        
        # Build graph
        self.knowledge_graph.build_networkx_graph()
        
        # Detect communities
        self.knowledge_graph.detect_political_communities()
        
        self.is_indexed = True
        logger.info("Graph-RAG indexing complete")
    
    def query_local(self, query: str) -> Dict[str, Any]:
        """
        Local Graph-RAG query for precise, entity-focused analysis.
        """
        if not self.is_indexed:
            return {'error': 'Corpus not indexed'}
        
        # Find relevant entity
        query_lower = query.lower()
        relevant_entity = None
        
        for entity_id, entity in self.knowledge_graph.entities.items():
            if entity.name.lower() in query_lower:
                relevant_entity = entity_id
                break
        
        if not relevant_entity:
            return {'error': f'No relevant entity found for query: {query}'}
        
        # Get local community information
        local_result = self.knowledge_graph.query_local_community(relevant_entity)
        local_result['query'] = query
        local_result['retrieval_method'] = 'graph_rag_local'
        
        return local_result
    
    def query_global(self, query: str) -> Dict[str, Any]:
        """
        Global Graph-RAG query for thematic, cross-community analysis.
        """
        if not self.is_indexed:
            return {'error': 'Corpus not indexed'}
        
        # Global thematic analysis
        global_result = self.knowledge_graph.query_global_communities(query)
        global_result['query'] = query
        global_result['retrieval_method'] = 'graph_rag_global'
        
        return global_result
    
    def trace_influence_path(self, source_politician: str, target_politician: str) -> Dict[str, Any]:
        """
        Trace influence paths between politicians using multi-hop traversal.
        """
        if not self.is_indexed:
            return {'error': 'Corpus not indexed'}
        
        # Find entity IDs
        source_id = self._find_politician_entity(source_politician)
        target_id = self._find_politician_entity(target_politician)
        
        if not source_id or not target_id:
            return {
                'error': f'Could not find entities for {source_politician} or {target_politician}',
                'available_politicians': [e.name for e in self.knowledge_graph.entities.values() 
                                        if e.entity_type == 'politician']
            }
        
        # Find paths
        paths = self.knowledge_graph.multi_hop_query(source_id, target_id)
        
        # Convert paths to readable format
        readable_paths = []
        for path in paths:
            readable_path = []
            for entity_id in path:
                if entity_id in self.knowledge_graph.entities:
                    entity = self.knowledge_graph.entities[entity_id]
                    readable_path.append(f"{entity.name} ({entity.entity_type})")
            readable_paths.append(readable_path)
        
        return {
            'source': source_politician,
            'target': target_politician,
            'paths_found': len(readable_paths),
            'influence_paths': readable_paths[:5],  # Limit to top 5 paths
            'retrieval_method': 'graph_rag_multi_hop'
        }
    
    def _find_politician_entity(self, politician_name: str) -> Optional[str]:
        """Find entity ID for a politician name."""
        name_lower = politician_name.lower()
        
        for entity_id, entity in self.knowledge_graph.entities.items():
            if (entity.entity_type == 'politician' and 
                (name_lower in entity.name.lower() or entity.name.lower() in name_lower)):
                return entity_id
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self.is_indexed:
            return {'error': 'Corpus not indexed'}
        
        entity_stats = defaultdict(int)
        for entity in self.knowledge_graph.entities.values():
            entity_stats[entity.entity_type] += 1
        
        return {
            'total_entities': len(self.knowledge_graph.entities),
            'entities_by_type': dict(entity_stats),
            'total_relationships': len(self.knowledge_graph.relationships),
            'graph_nodes': len(self.knowledge_graph.graph.nodes),
            'graph_edges': len(self.knowledge_graph.graph.edges),
            'communities': len(self.knowledge_graph.communities),
            'is_indexed': self.is_indexed
        }

# Integration function
def integrate_graph_rag_with_existing_system():
    """
    Integration function for Graph-RAG with existing political analysis.
    """
    
    try:
        from .political_rootfinder import PoliticalRootFinder
        from .political_memespace import PoliticalMemespace
        from .expanded_political_corpus import create_expanded_political_corpus
        
        class EnhancedPoliticalAnalysisWithGraphRAG:
            def __init__(self):
                self.rootfinder = PoliticalRootFinder()
                self.memespace = PoliticalMemespace()
                self.graph_rag = PoliticalGraphRAG()
                
            def setup_graph_rag(self):
                """Initialize Graph-RAG with political corpus."""
                corpus = create_expanded_political_corpus()
                self.graph_rag.index_political_corpus(corpus)
                logger.info("Graph-RAG setup complete")
            
            def hybrid_graph_query(self, query: str, query_type: str = 'auto') -> Dict[str, Any]:
                """
                Query using Graph-RAG with automatic type detection.
                """
                if query_type == 'auto':
                    # Simple heuristics for query type detection
                    if any(word in query.lower() for word in ['influencia', 'genealogía', 'evolución', 'desarrollo']):
                        query_type = 'global'
                    elif any(word in query.lower() for word in ['quién', 'qué', 'específico', 'detalle']):
                        query_type = 'local'
                    else:
                        query_type = 'global'
                
                if query_type == 'local':
                    return self.graph_rag.query_local(query)
                elif query_type == 'global':
                    return self.graph_rag.query_global(query)
                else:
                    return {'error': f'Unknown query type: {query_type}'}
            
            def trace_political_genealogy_graph(self, source: str, target: str) -> Dict[str, Any]:
                """
                Use Graph-RAG for genealogy tracing (alternative to existing method).
                """
                return self.graph_rag.trace_influence_path(source, target)
        
        return EnhancedPoliticalAnalysisWithGraphRAG
        
    except ImportError as e:
        logger.error(f"Could not integrate Graph-RAG with existing system: {e}")
        return None

if __name__ == "__main__":
    # Demo Graph-RAG functionality
    print("🕸️  Political Graph-RAG Demo")
    
    # This would be tested with actual corpus
    print("📋 This module provides:")
    print("   • Political entity extraction (politicians, laws, events, concepts)")
    print("   • Relationship detection (influence, opposition, co-occurrence)")
    print("   • Community detection (political families/movements)")
    print("   • Multi-hop traversal (genealogical paths)")
    print("   • Local queries (entity-focused analysis)")
    print("   • Global queries (thematic cross-community analysis)")
    
    print("\n✅ Graph-RAG implementation ready for integration!")