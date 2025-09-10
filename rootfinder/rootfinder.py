"""
RootFinder: Tracing Genealogical Paths Through Legal Evolution
Implements ABAN (Ancestral Backward Analysis of Networks) algorithm
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

logger = logging.getLogger(__name__)

@dataclass
class GenealogyNode:
    """Represents a node in the genealogical tree."""
    case_id: str
    generation: int
    inherited_elements: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    inheritance_fidelity: float = 0.0
    mutation_type: str = "unknown"
    citation_strength: float = 1.0
    doctrinal_distance: float = 0.0
    precedential_weight: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'case_id': self.case_id,
            'generation': self.generation,
            'inherited_elements': self.inherited_elements,
            'mutations': self.mutations,
            'inheritance_fidelity': self.inheritance_fidelity,
            'mutation_type': self.mutation_type,
            'citation_strength': self.citation_strength,
            'doctrinal_distance': self.doctrinal_distance,
            'precedential_weight': self.precedential_weight
        }

class RootFinder:
    """
    Traces genealogical lineages of legal doctrines through citation networks.
    
    The ABAN algorithm performs backward traversal through citation networks
    to identify precedential lineages and quantify doctrinal evolution.
    """
    
    def __init__(self, min_citation_strength: float = 0.1,
                 max_doctrinal_distance: float = 0.8,
                 fidelity_threshold: float = 0.3):
        """
        Initialize RootFinder.
        
        Parameters:
        -----------
        min_citation_strength : float
            Minimum citation strength to follow genealogical paths
        max_doctrinal_distance : float
            Maximum doctrinal distance to consider as same lineage
        fidelity_threshold : float
            Threshold for considering inheritance as "faithful"
        """
        self.min_citation_strength = min_citation_strength
        self.max_doctrinal_distance = max_doctrinal_distance
        self.fidelity_threshold = fidelity_threshold
        self.genealogy_cache = {}
        
    def trace_genealogy(self, target_case: str, citation_network: nx.DiGraph, 
                       max_depth: int = 10, include_weak_paths: bool = False) -> List[GenealogyNode]:
        """
        Trace the genealogical lineage of a legal doctrine using ABAN algorithm.
        
        Parameters:
        -----------
        target_case : str
            Case ID to trace backwards from
        citation_network : nx.DiGraph
            Directed graph of citations (edge from citing to cited case)
        max_depth : int
            Maximum generations to trace back
        include_weak_paths : bool
            Whether to include genealogical paths with weak citations
            
        Returns:
        --------
        List[GenealogyNode]
            Genealogical path from target to root ancestors
        """
        if target_case in self.genealogy_cache:
            logger.info(f"Using cached genealogy for {target_case}")
            return self.genealogy_cache[target_case]
        
        logger.info(f"Tracing genealogy for {target_case} (max_depth: {max_depth})")
        
        genealogy = []
        current_case = target_case
        visited = set()
        depth = 0
        
        while depth < max_depth and current_case not in visited:
            visited.add(current_case)
            
            # Get citations (ancestors)
            ancestors = list(citation_network.predecessors(current_case))
            
            if not ancestors:
                logger.debug(f"No more ancestors found for {current_case} at depth {depth}")
                break
            
            # Filter ancestors by citation strength if required
            if not include_weak_paths:
                filtered_ancestors = []
                for ancestor in ancestors:
                    edge_data = citation_network.get_edge_data(current_case, ancestor, {})
                    strength = edge_data.get('weight', 1.0)
                    if strength >= self.min_citation_strength:
                        filtered_ancestors.append(ancestor)
                ancestors = filtered_ancestors
                
                if not ancestors:
                    logger.debug(f"No strong citations found for {current_case}")
                    break
            
            # Find primary ancestor (strongest precedential connection)
            primary_ancestor = self._identify_primary_ancestor(
                current_case, ancestors, citation_network
            )
            
            # Analyze inheritance and mutations
            inherited, mutations = self._analyze_inheritance(
                current_case, primary_ancestor, citation_network
            )
            
            # Calculate inheritance fidelity
            fidelity = self._calculate_fidelity(inherited, mutations)
            
            # Classify mutation type
            mutation_type = self._classify_mutation(mutations, inherited, fidelity)
            
            # Calculate additional metrics
            citation_strength = self._get_citation_strength(
                current_case, primary_ancestor, citation_network
            )
            
            doctrinal_distance = self._calculate_doctrinal_distance(
                current_case, primary_ancestor, citation_network
            )
            
            precedential_weight = self._calculate_precedential_weight(
                primary_ancestor, citation_network
            )
            
            # Create genealogy node
            node = GenealogyNode(
                case_id=primary_ancestor,
                generation=depth,
                inherited_elements=inherited,
                mutations=mutations,
                inheritance_fidelity=fidelity,
                mutation_type=mutation_type,
                citation_strength=citation_strength,
                doctrinal_distance=doctrinal_distance,
                precedential_weight=precedential_weight
            )
            
            genealogy.append(node)
            current_case = primary_ancestor
            depth += 1
            
        # Cache the result
        self.genealogy_cache[target_case] = genealogy
        
        logger.info(f"Genealogy traced: {len(genealogy)} generations for {target_case}")
        return genealogy
    
    def _identify_primary_ancestor(self, case: str, ancestors: List[str], 
                                   network: nx.DiGraph) -> str:
        """
        Identify the primary precedential ancestor using multiple criteria.
        """
        if len(ancestors) == 1:
            return ancestors[0]
        
        scores = {}
        
        for ancestor in ancestors:
            score = 0.0
            
            # Citation strength weight (40%)
            edge_data = network.get_edge_data(case, ancestor, {})
            citation_weight = edge_data.get('weight', 1.0)
            score += 0.4 * citation_weight
            
            # Precedential importance (30%)
            precedential_weight = self._calculate_precedential_weight(ancestor, network)
            score += 0.3 * precedential_weight
            
            # Doctrinal similarity (20%)
            doctrinal_similarity = 1.0 - self._calculate_doctrinal_distance(case, ancestor, network)
            score += 0.2 * doctrinal_similarity
            
            # Temporal proximity (10%)
            temporal_score = self._calculate_temporal_proximity(case, ancestor, network)
            score += 0.1 * temporal_score
            
            scores[ancestor] = score
        
        # Return ancestor with highest composite score
        primary_ancestor = max(scores.items(), key=lambda x: x[1])[0]
        logger.debug(f"Primary ancestor for {case}: {primary_ancestor} (score: {scores[primary_ancestor]:.3f})")
        
        return primary_ancestor
    
    def _analyze_inheritance(self, child: str, parent: str, 
                            network: nx.DiGraph) -> Tuple[List[str], List[str]]:
        """
        Analyze doctrinal inheritance and mutations between cases.
        """
        # Extract doctrinal elements from node attributes
        child_attrs = network.nodes.get(child, {})
        parent_attrs = network.nodes.get(parent, {})
        
        # Handle different formats of doctrinal elements
        child_elements = self._extract_doctrinal_elements(child_attrs)
        parent_elements = self._extract_doctrinal_elements(parent_attrs)
        
        # Calculate inheritance and mutations
        inherited = list(child_elements & parent_elements)
        mutations = list(child_elements - parent_elements)
        
        logger.debug(f"Inheritance analysis {parent} -> {child}: "
                    f"{len(inherited)} inherited, {len(mutations)} mutations")
        
        return inherited, mutations
    
    def _extract_doctrinal_elements(self, attrs: Dict) -> Set[str]:
        """Extract doctrinal elements from case attributes."""
        elements = set()
        
        # Try multiple possible attribute names
        element_keys = ['doctrinal_elements', 'doctrine_tags', 'legal_principles', 
                       'holdings', 'ratio_decidendi']
        
        for key in element_keys:
            if key in attrs:
                value = attrs[key]
                if isinstance(value, (list, tuple)):
                    elements.update(str(v) for v in value)
                elif isinstance(value, str):
                    # Handle comma-separated strings
                    elements.update(v.strip() for v in value.split(',') if v.strip())
                break
        
        # If no explicit doctrinal elements, infer from other attributes
        if not elements and attrs:
            # Use case characteristics as proxy doctrinal elements
            for key, value in attrs.items():
                if key in ['emergency_power', 'state_intervention', 'property_rights', 
                          'due_process', 'constitutional_interpretation']:
                    if value:
                        elements.add(f"{key}:{value}")
        
        return elements
    
    def _calculate_fidelity(self, inherited: List[str], mutations: List[str]) -> float:
        """Calculate inheritance fidelity score."""
        total_elements = len(inherited) + len(mutations)
        if total_elements == 0:
            return 0.0
        
        fidelity = len(inherited) / total_elements
        return fidelity
    
    def _classify_mutation(self, mutations: List[str], inherited: List[str], 
                          fidelity: float) -> str:
        """
        Classify the type of doctrinal mutation.
        """
        if not mutations:
            return "faithful"
        
        mutation_ratio = len(mutations) / (len(mutations) + len(inherited)) if inherited else 1.0
        
        if fidelity >= self.fidelity_threshold:
            if mutation_ratio <= 0.2:
                return "conservative"  # Minor additions/refinements
            elif mutation_ratio <= 0.4:
                return "incremental"   # Moderate evolution
            else:
                return "expansive"     # Significant additions
        else:
            if mutation_ratio >= 0.7:
                return "revolutionary" # Major doctrinal shift
            else:
                return "transformative" # Substantial change
    
    def _get_citation_strength(self, citing_case: str, cited_case: str, 
                              network: nx.DiGraph) -> float:
        """Get citation strength between two cases."""
        edge_data = network.get_edge_data(citing_case, cited_case, {})
        return edge_data.get('weight', 1.0)
    
    def _calculate_doctrinal_distance(self, case1: str, case2: str, 
                                     network: nx.DiGraph) -> float:
        """Calculate doctrinal distance between two cases."""
        attrs1 = network.nodes.get(case1, {})
        attrs2 = network.nodes.get(case2, {})
        
        elements1 = self._extract_doctrinal_elements(attrs1)
        elements2 = self._extract_doctrinal_elements(attrs2)
        
        if not elements1 and not elements2:
            return 0.0
        
        if not elements1 or not elements2:
            return 1.0
        
        # Calculate Jaccard distance (1 - Jaccard similarity)
        intersection = len(elements1 & elements2)
        union = len(elements1 | elements2)
        
        jaccard_similarity = intersection / union if union > 0 else 0
        jaccard_distance = 1.0 - jaccard_similarity
        
        return jaccard_distance
    
    def _calculate_precedential_weight(self, case: str, network: nx.DiGraph) -> float:
        """Calculate the precedential importance of a case."""
        # Number of cases citing this case (in-degree)
        in_degree = network.in_degree(case, weight='weight')
        
        # Normalize by network size
        max_possible_citations = len(network.nodes) - 1
        if max_possible_citations == 0:
            return 0.0
        
        normalized_weight = min(in_degree / max_possible_citations, 1.0)
        
        # Apply court hierarchy boost if available
        attrs = network.nodes.get(case, {})
        court_level = attrs.get('court_level', 'Lower Court')
        
        hierarchy_multiplier = {
            'Supreme Court': 1.0,
            'Appeals Court': 0.8,
            'Federal Court': 0.7,
            'Provincial Supreme': 0.6,
            'Lower Court': 0.5
        }.get(court_level, 0.5)
        
        return normalized_weight * hierarchy_multiplier
    
    def _calculate_temporal_proximity(self, case1: str, case2: str, 
                                     network: nx.DiGraph) -> float:
        """Calculate temporal proximity score between cases."""
        attrs1 = network.nodes.get(case1, {})
        attrs2 = network.nodes.get(case2, {})
        
        date1 = attrs1.get('date')
        date2 = attrs2.get('date')
        
        if not date1 or not date2:
            return 0.5  # Default medium proximity
        
        try:
            dt1 = pd.to_datetime(date1)
            dt2 = pd.to_datetime(date2)
            
            years_diff = abs((dt1 - dt2).days / 365.25)
            
            # Exponential decay: closer in time = higher score
            proximity = np.exp(-0.1 * years_diff)  # 10% decay per year
            
            return min(proximity, 1.0)
            
        except Exception:
            return 0.5
    
    def calculate_peralta_dominance(self, all_cases: List[str], 
                                   citation_network: nx.DiGraph,
                                   peralta_case_id: str = 'Peralta_1990') -> Dict[str, float]:
        """
        Calculate percentage of cases tracing back to Peralta and related metrics.
        
        Parameters:
        -----------
        all_cases : List[str]
            List of all case IDs to analyze
        citation_network : nx.DiGraph
            Citation network
        peralta_case_id : str
            Case ID for the Peralta decision
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with dominance metrics
        """
        logger.info(f"Calculating Peralta dominance for {len(all_cases)} cases")
        
        peralta_descendants = 0
        total_analyzed = 0
        generation_stats = defaultdict(int)
        fidelity_scores = []
        
        for case in all_cases:
            if case == peralta_case_id:
                continue
                
            try:
                genealogy = self.trace_genealogy(case, citation_network)
                case_ids = [node.case_id for node in genealogy]
                
                total_analyzed += 1
                
                if peralta_case_id in case_ids:
                    peralta_descendants += 1
                    
                    # Find Peralta in genealogy
                    peralta_generation = None
                    peralta_fidelity = None
                    
                    for node in genealogy:
                        if node.case_id == peralta_case_id:
                            peralta_generation = node.generation
                            peralta_fidelity = node.inheritance_fidelity
                            break
                    
                    if peralta_generation is not None:
                        generation_stats[peralta_generation] += 1
                        
                    if peralta_fidelity is not None:
                        fidelity_scores.append(peralta_fidelity)
                        
            except Exception as e:
                logger.warning(f"Error tracing genealogy for {case}: {e}")
                continue
        
        # Calculate metrics
        dominance_rate = peralta_descendants / total_analyzed if total_analyzed > 0 else 0
        avg_generation = np.mean(list(generation_stats.keys())) if generation_stats else 0
        avg_fidelity = np.mean(fidelity_scores) if fidelity_scores else 0
        
        results = {
            'dominance_rate': dominance_rate,
            'total_descendants': peralta_descendants,
            'total_analyzed': total_analyzed,
            'average_generation_distance': avg_generation,
            'average_inheritance_fidelity': avg_fidelity,
            'generation_distribution': dict(generation_stats)
        }
        
        logger.info(f"Peralta dominance: {dominance_rate:.1%} ({peralta_descendants}/{total_analyzed})")
        
        return results
    
    def find_doctrinal_roots(self, cases: List[str], citation_network: nx.DiGraph,
                           min_descendants: int = 5) -> Dict[str, Dict]:
        """
        Identify foundational cases that serve as doctrinal roots.
        
        Parameters:
        -----------
        cases : List[str]
            Cases to analyze
        citation_network : nx.DiGraph
            Citation network
        min_descendants : int
            Minimum number of descendant cases to be considered a root
            
        Returns:
        --------
        Dict[str, Dict]
            Dictionary mapping root case IDs to their statistics
        """
        logger.info(f"Finding doctrinal roots among {len(cases)} cases")
        
        # Count descendants for each potential root
        descendant_counts = defaultdict(set)
        
        for case in cases:
            genealogy = self.trace_genealogy(case, citation_network)
            
            for node in genealogy:
                descendant_counts[node.case_id].add(case)
        
        # Filter roots by minimum descendant threshold
        roots = {}
        
        for potential_root, descendants in descendant_counts.items():
            if len(descendants) >= min_descendants:
                # Calculate root statistics
                fidelities = []
                mutation_types = defaultdict(int)
                
                # Analyze all genealogies that include this root
                for descendant in descendants:
                    genealogy = self.trace_genealogy(descendant, citation_network)
                    
                    for node in genealogy:
                        if node.case_id == potential_root:
                            fidelities.append(node.inheritance_fidelity)
                            mutation_types[node.mutation_type] += 1
                            break
                
                roots[potential_root] = {
                    'descendant_count': len(descendants),
                    'descendants': list(descendants),
                    'average_fidelity': np.mean(fidelities) if fidelities else 0,
                    'mutation_distribution': dict(mutation_types),
                    'influence_score': len(descendants) * (np.mean(fidelities) if fidelities else 0)
                }
        
        # Sort by influence score
        sorted_roots = dict(sorted(roots.items(), 
                                 key=lambda x: x[1]['influence_score'], 
                                 reverse=True))
        
        logger.info(f"Found {len(sorted_roots)} doctrinal roots")
        
        return sorted_roots
    
    def export_genealogy_tree(self, genealogy: List[GenealogyNode], 
                             output_path: str, format: str = 'json'):
        """
        Export genealogy tree to file.
        
        Parameters:
        -----------
        genealogy : List[GenealogyNode]
            Genealogy to export
        output_path : str
            Output file path
        format : str
            Export format ('json', 'csv', 'txt')
        """
        if format == 'json':
            data = [node.to_dict() for node in genealogy]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format == 'csv':
            data = [node.to_dict() for node in genealogy]
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
        elif format == 'txt':
            with open(output_path, 'w') as f:
                f.write("GENEALOGICAL LINEAGE\n")
                f.write("=" * 50 + "\n\n")
                
                for node in genealogy:
                    f.write(f"Generation {node.generation}: {node.case_id}\n")
                    f.write(f"  Inheritance Fidelity: {node.inheritance_fidelity:.3f}\n")
                    f.write(f"  Mutation Type: {node.mutation_type}\n")
                    f.write(f"  Citation Strength: {node.citation_strength:.3f}\n")
                    f.write(f"  Inherited Elements: {', '.join(node.inherited_elements[:5])}\n")
                    if len(node.inherited_elements) > 5:
                        f.write(f"    ... and {len(node.inherited_elements) - 5} more\n")
                    f.write(f"  Mutations: {', '.join(node.mutations[:3])}\n")
                    if len(node.mutations) > 3:
                        f.write(f"    ... and {len(node.mutations) - 3} more\n")
                    f.write("\n")
        
        logger.info(f"Genealogy exported to {output_path} ({format} format)")
        
    def clear_cache(self):
        """Clear the genealogy cache."""
        self.genealogy_cache.clear()
        logger.info("Genealogy cache cleared")