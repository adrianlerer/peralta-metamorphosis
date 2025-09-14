"""
Political RootFinder: Tracing Genealogical Paths Through Political Evolution
Adapts ABAN (Ancestral Backward Analysis of Networks) algorithm for political texts
Based on: rootfinder.rootfinder.RootFinder
Author: Ignacio Adrián Lerer
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Import base RootFinder
import sys
sys.path.append('..')
from rootfinder.rootfinder import RootFinder, GenealogyNode

logger = logging.getLogger(__name__)

@dataclass
class PoliticalGenealogyNode:
    """Represents a node in the political genealogical tree."""
    document_id: str
    author: str
    year: int
    generation: int
    inherited_themes: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    semantic_similarity: float = 0.0
    mutation_type: str = "unknown"
    ideological_distance: float = 0.0
    influence_weight: float = 0.0
    political_position: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'author': self.author,
            'year': self.year,
            'generation': self.generation,
            'inherited_themes': self.inherited_themes,
            'mutations': self.mutations,
            'semantic_similarity': self.semantic_similarity,
            'mutation_type': self.mutation_type,
            'ideological_distance': self.ideological_distance,
            'influence_weight': self.influence_weight,
            'political_position': self.political_position
        }

class PoliticalRootFinder(RootFinder):
    """
    Adapts RootFinder to trace political memes and antagonisms through historical texts.
    
    Instead of legal citations, uses:
    - Semantic similarity between texts
    - Shared political themes and concepts
    - Historical influence patterns
    - Ideological inheritance and mutations
    """
    
    def __init__(self, min_semantic_similarity: float = 0.3,
                 max_ideological_distance: float = 0.7,
                 similarity_threshold: float = 0.4,
                 political_dimensions: int = 4):
        """
        Initialize Political RootFinder.
        
        Parameters:
        -----------
        min_semantic_similarity : float
            Minimum semantic similarity to consider genealogical connection
        max_ideological_distance : float
            Maximum ideological distance to consider same lineage
        similarity_threshold : float
            Threshold for considering inheritance as meaningful
        political_dimensions : int
            Number of political dimensions for position analysis
        """
        # Initialize base class with adapted parameters
        super().__init__(
            min_citation_strength=min_semantic_similarity,
            max_doctrinal_distance=max_ideological_distance,
            fidelity_threshold=similarity_threshold
        )
        
        self.min_semantic_similarity = min_semantic_similarity
        self.max_ideological_distance = max_ideological_distance
        self.similarity_threshold = similarity_threshold
        self.political_dimensions = political_dimensions
        
        # Political-specific components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Allow single occurrence terms
            max_df=0.95  # Remove very common terms
        )
        self.political_genealogy_cache = {}
        self.theme_genealogy_cache = {}
        
    def build_semantic_network(self, political_documents: pd.DataFrame) -> nx.DiGraph:
        """
        Build a network based on semantic similarity instead of citations.
        
        Parameters:
        -----------
        political_documents : pd.DataFrame
            DataFrame with columns: document_id, author, year, text, political_position
            
        Returns:
        --------
        nx.DiGraph
            Directed graph where edges represent semantic influence
        """
        logger.info(f"Building semantic network from {len(political_documents)} documents")
        
        # Create TF-IDF vectors for all documents
        texts = political_documents['text'].fillna('').tolist()
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Build network
        network = nx.DiGraph()
        
        # Add nodes with metadata
        for _, doc in political_documents.iterrows():
            network.add_node(doc['document_id'], 
                           author=doc['author'],
                           year=doc['year'],
                           text=doc['text'],
                           political_position=doc.get('political_position', []))
        
        # Add edges based on semantic similarity and temporal precedence
        for i, doc1 in political_documents.iterrows():
            for j, doc2 in political_documents.iterrows():
                if i != j and doc1['year'] > doc2['year']:  # doc1 cites/inherits from doc2
                    similarity = similarity_matrix[i, j]
                    
                    if similarity >= self.min_semantic_similarity:
                        # Add edge from newer to older (inheritance direction)
                        network.add_edge(doc1['document_id'], doc2['document_id'],
                                       semantic_similarity=similarity,
                                       temporal_distance=doc1['year'] - doc2['year'],
                                       weight=similarity)
        
        logger.info(f"Semantic network created: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
        return network
    
    def trace_political_genealogy(self, target_document: str, 
                                political_network: nx.DiGraph,
                                max_depth: int = 8,
                                include_weak_paths: bool = False) -> List[PoliticalGenealogyNode]:
        """
        Trace the genealogy of political ideas from a target document.
        
        Parameters:
        -----------
        target_document : str
            Document ID to trace backwards from
        political_network : nx.DiGraph
            Network of political documents with semantic connections
        max_depth : int
            Maximum generations to trace back
        include_weak_paths : bool
            Whether to include paths with weak semantic similarity
            
        Returns:
        --------
        List[PoliticalGenealogyNode]
            Genealogical path from target to ideological ancestors
        """
        cache_key = f"{target_document}_{max_depth}_{include_weak_paths}"
        if cache_key in self.political_genealogy_cache:
            return self.political_genealogy_cache[cache_key]
        
        logger.info(f"Tracing political genealogy for {target_document}")
        
        if target_document not in political_network:
            logger.warning(f"Document {target_document} not found in network")
            return []
        
        genealogy = []
        visited = set()
        queue = deque([(target_document, 0)])
        
        while queue and len(genealogy) < max_depth:
            current_doc, generation = queue.popleft()
            
            if current_doc in visited:
                continue
            visited.add(current_doc)
            
            # Get document metadata
            doc_data = political_network.nodes[current_doc]
            
            # Find predecessors (documents this one inherits from)
            predecessors = list(political_network.successors(current_doc))
            
            # Calculate inherited themes and mutations
            inherited_themes = []
            mutations = []
            max_similarity = 0
            
            if predecessors:
                for pred in predecessors:
                    edge_data = political_network[current_doc][pred]
                    similarity = edge_data.get('semantic_similarity', 0)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                    
                    if similarity >= self.similarity_threshold:
                        inherited_themes.extend(self._extract_shared_themes(
                            doc_data.get('text', ''), 
                            political_network.nodes[pred].get('text', '')
                        ))
                
                # Identify mutations (new themes not in predecessors)
                mutations = self._identify_mutations(current_doc, predecessors, political_network)
            
            # Create genealogy node
            node = PoliticalGenealogyNode(
                document_id=current_doc,
                author=doc_data.get('author', 'Unknown'),
                year=doc_data.get('year', 0),
                generation=generation,
                inherited_themes=inherited_themes,
                mutations=mutations,
                semantic_similarity=max_similarity,
                ideological_distance=self._calculate_ideological_distance(current_doc, political_network),
                political_position=doc_data.get('political_position', [])
            )
            
            genealogy.append(node)
            
            # Add predecessors to queue for next generation
            for pred in predecessors:
                edge_data = political_network[current_doc][pred]
                if (include_weak_paths or 
                    edge_data.get('semantic_similarity', 0) >= self.min_semantic_similarity):
                    queue.append((pred, generation + 1))
        
        # Cache and return
        self.political_genealogy_cache[cache_key] = genealogy
        return genealogy
    
    def find_common_political_ancestor(self, document1: str, document2: str,
                                     political_network: nx.DiGraph) -> Optional[str]:
        """
        Find the earliest common ideological ancestor of two political documents.
        
        This reveals when opposing political movements diverged from common roots.
        """
        logger.info(f"Finding common ancestor of {document1} and {document2}")
        
        # Trace genealogies for both documents
        genealogy1 = self.trace_political_genealogy(document1, political_network)
        genealogy2 = self.trace_political_genealogy(document2, political_network)
        
        # Extract ancestor sets
        ancestors1 = {node.document_id for node in genealogy1}
        ancestors2 = {node.document_id for node in genealogy2}
        
        # Find common ancestors
        common_ancestors = ancestors1.intersection(ancestors2)
        
        if not common_ancestors:
            return None
        
        # Return the most recent common ancestor (lowest generation number)
        common_genealogy = [node for node in genealogy1 if node.document_id in common_ancestors]
        common_genealogy.extend([node for node in genealogy2 if node.document_id in common_ancestors])
        
        # Sort by generation and return the earliest
        common_genealogy.sort(key=lambda x: x.generation)
        return common_genealogy[0].document_id if common_genealogy else None
    
    def calculate_inheritance_strength(self, genealogy_path: List[PoliticalGenealogyNode]) -> float:
        """
        Calculate overall inheritance strength across a genealogical path.
        
        Measures how much political DNA is preserved across generations.
        """
        if len(genealogy_path) <= 1:
            return 0.0
        
        similarities = [node.semantic_similarity for node in genealogy_path[1:]]  # Skip target
        weights = [1.0 / (i + 1) for i in range(len(similarities))]  # Decay with distance
        
        weighted_similarity = sum(s * w for s, w in zip(similarities, weights))
        total_weight = sum(weights)
        
        return weighted_similarity / total_weight if total_weight > 0 else 0.0
    
    def identify_major_mutations(self, genealogy_path: List[PoliticalGenealogyNode]) -> List[Dict]:
        """
        Identify major ideological mutations in a genealogical path.
        
        Returns points where significant new political ideas emerge.
        """
        mutations = []
        
        for i, node in enumerate(genealogy_path):
            if node.mutations and len(node.mutations) > 2:  # Significant mutation
                mutation_info = {
                    'document_id': node.document_id,
                    'author': node.author,
                    'year': node.year,
                    'generation': node.generation,
                    'new_themes': node.mutations,
                    'mutation_strength': len(node.mutations) / max(len(node.inherited_themes), 1)
                }
                mutations.append(mutation_info)
        
        return mutations
    
    def _extract_shared_themes(self, text1: str, text2: str) -> List[str]:
        """Extract shared political themes between two texts."""
        # Simple implementation - in practice, would use more sophisticated NLP
        political_keywords = [
            'federalismo', 'centralismo', 'pueblo', 'elite', 'revolución', 'evolución',
            'buenos aires', 'interior', 'provincia', 'nación', 'libertad', 'orden',
            'progreso', 'tradición', 'democracia', 'autoritarismo', 'popular', 'oligarquía'
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        shared_themes = []
        for keyword in political_keywords:
            if keyword in text1_lower and keyword in text2_lower:
                shared_themes.append(keyword)
        
        return shared_themes
    
    def _identify_mutations(self, current_doc: str, predecessors: List[str], 
                          network: nx.DiGraph) -> List[str]:
        """Identify new political themes not present in predecessors."""
        current_text = network.nodes[current_doc].get('text', '').lower()
        
        # Get all predecessor texts
        predecessor_texts = []
        for pred in predecessors:
            predecessor_texts.append(network.nodes[pred].get('text', '').lower())
        combined_predecessor_text = ' '.join(predecessor_texts)
        
        # Find themes in current but not in predecessors
        political_keywords = [
            'federalismo', 'centralismo', 'pueblo', 'elite', 'revolución', 'evolución',
            'buenos aires', 'interior', 'provincia', 'nación', 'libertad', 'orden',
            'progreso', 'tradición', 'democracia', 'autoritarismo', 'popular', 'oligarquía',
            'peronismo', 'antiperonismo', 'neoliberalismo', 'populismo', 'liberal',
            'conservador', 'radical', 'socialista'
        ]
        
        mutations = []
        for keyword in political_keywords:
            if keyword in current_text and keyword not in combined_predecessor_text:
                mutations.append(keyword)
        
        return mutations
    
    def _calculate_ideological_distance(self, document_id: str, network: nx.DiGraph) -> float:
        """Calculate ideological distance from political center."""
        # Simplified implementation - would use actual political positioning in practice
        doc_data = network.nodes[document_id]
        political_position = doc_data.get('political_position', [0.5, 0.5, 0.5, 0.5])
        
        # Calculate Euclidean distance from center [0.5, 0.5, 0.5, 0.5]
        center = [0.5] * len(political_position)
        distance = np.sqrt(sum((p - c) ** 2 for p, c in zip(political_position, center)))
        
        return distance
    
    def export_genealogy_results(self, genealogies: Dict[str, List[PoliticalGenealogyNode]], 
                               filename: str) -> None:
        """Export genealogy results to JSON file."""
        export_data = {}
        
        for target, genealogy in genealogies.items():
            export_data[target] = {
                'genealogy_path': [node.to_dict() for node in genealogy],
                'inheritance_strength': self.calculate_inheritance_strength(genealogy),
                'major_mutations': self.identify_major_mutations(genealogy),
                'path_length': len(genealogy),
                'oldest_ancestor': genealogy[-1].document_id if genealogy else None
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Genealogy results exported to {filename}")