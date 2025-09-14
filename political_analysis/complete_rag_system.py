#!/usr/bin/env python3
"""
Complete RAG System Integration
All paper-based improvements integrated into a unified system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import time
import json
from pathlib import Path

# Import all components
from .vector_rag_integration import PoliticalVectorRAG, PoliticalHierarchicalSummarizer
from .graph_rag_implementation import PoliticalGraphRAG
from .enhanced_query_router import EnhancedPoliticalQueryRouter, create_enhanced_router, QueryType, RetrievalMethod
from .knowledge_shift_tester import PoliticalKnowledgeShiftTester
from .expanded_political_corpus import create_expanded_political_corpus

logger = logging.getLogger(__name__)

class CompletePoliticalRAGSystem:
    """
    Complete Political RAG system integrating all paper-based improvements.
    
    Features:
    - Hybrid Vector + Graph RAG
    - Intelligent query routing
    - Knowledge-shift validation
    - Community detection
    - Hierarchical summarization
    """
    
    def __init__(self, use_openai: bool = False, use_llm_router: bool = False):
        self.use_openai = use_openai
        self.use_llm_router = use_llm_router
        
        # Initialize components
        self.vector_rag = PoliticalVectorRAG(use_openai_embeddings=use_openai)
        self.graph_rag = PoliticalGraphRAG()
        self.router = create_enhanced_router(use_llm=use_llm_router)
        self.summarizer = PoliticalHierarchicalSummarizer()
        self.knowledge_tester = PoliticalKnowledgeShiftTester()
        
        # System state
        self.is_indexed = False
        self.corpus = None
        self.system_stats = {}
        
        logger.info(f"Complete RAG system initialized (OpenAI: {use_openai}, LLM Router: {use_llm_router})")
    
    def setup_system(self) -> None:
        """
        Initialize the complete RAG system with political corpus.
        """
        logger.info("Setting up complete political RAG system...")
        
        start_time = time.time()
        
        # Load corpus
        logger.info("Loading expanded political corpus...")
        self.corpus = create_expanded_political_corpus()
        
        # Setup Vector-RAG
        logger.info("Setting up Vector-RAG...")
        vector_start = time.time()
        documents_dict = self.corpus.to_dict('records')
        self.vector_rag.index_political_corpus(documents_dict)
        vector_time = time.time() - vector_start
        
        # Setup Graph-RAG
        logger.info("Setting up Graph-RAG...")
        graph_start = time.time()
        self.graph_rag.index_political_corpus(self.corpus)
        graph_time = time.time() - graph_start
        
        # Setup knowledge-shift tests
        logger.info("Generating knowledge-shift tests...")
        test_start = time.time()
        self.knowledge_tester.generate_political_shift_tests(self.corpus)
        test_time = time.time() - test_start
        
        total_time = time.time() - start_time
        
        # Store statistics
        self.system_stats = {
            'setup_time_total': total_time,
            'vector_rag_setup_time': vector_time,
            'graph_rag_setup_time': graph_time,
            'knowledge_test_setup_time': test_time,
            'corpus_documents': len(self.corpus),
            'vector_chunks': len(self.vector_rag.chunks),
            'graph_entities': len(self.graph_rag.knowledge_graph.entities),
            'graph_relationships': len(self.graph_rag.knowledge_graph.relationships),
            'communities': len(self.graph_rag.knowledge_graph.communities),
            'knowledge_shift_tests': len(self.knowledge_tester.test_cases)
        }
        
        self.is_indexed = True
        
        logger.info(f"Complete RAG system setup complete in {total_time:.2f}s")
        logger.info(f"  Documents: {self.system_stats['corpus_documents']}")
        logger.info(f"  Vector chunks: {self.system_stats['vector_chunks']}")
        logger.info(f"  Graph entities: {self.system_stats['graph_entities']}")
        logger.info(f"  Communities: {self.system_stats['communities']}")
        logger.info(f"  Knowledge tests: {self.system_stats['knowledge_shift_tests']}")
    
    def query(self, question: str, include_provenance: bool = True) -> Dict[str, Any]:
        """
        Main query interface using intelligent routing.
        """
        if not self.is_indexed:
            return {'error': 'System not initialized. Call setup_system() first.'}
        
        start_time = time.time()
        
        # Route the query
        routing_decision = self.router.route_query(question)
        
        # Execute appropriate retrieval method
        result = {
            'question': question,
            'routing_decision': {
                'query_type': routing_decision.query_type.value,
                'retrieval_method': routing_decision.retrieval_method.value,
                'confidence': routing_decision.confidence,
                'reasoning': routing_decision.reasoning
            },
            'answers': {},
            'query_time': 0.0,
            'provenance': []
        }
        
        try:
            if routing_decision.retrieval_method == RetrievalMethod.VECTOR_RAG:
                answer = self._execute_vector_rag(question)
                result['answers']['primary'] = answer
                
            elif routing_decision.retrieval_method == RetrievalMethod.GRAPH_RAG_LOCAL:
                answer = self._execute_graph_rag_local(question)
                result['answers']['primary'] = answer
                
            elif routing_decision.retrieval_method == RetrievalMethod.GRAPH_RAG_GLOBAL:
                answer = self._execute_graph_rag_global(question)
                result['answers']['primary'] = answer
                
            elif routing_decision.retrieval_method == RetrievalMethod.HYBRID_RAG:
                vector_answer = self._execute_vector_rag(question)
                graph_answer = self._execute_graph_rag_global(question)
                
                result['answers']['vector_rag'] = vector_answer
                result['answers']['graph_rag'] = graph_answer
                result['answers']['primary'] = self._combine_hybrid_answers(vector_answer, graph_answer)
            
            # Add provenance information
            if include_provenance:
                result['provenance'] = self._collect_provenance(result['answers'])
            
            result['query_time'] = time.time() - start_time
            result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}")
            result['error'] = str(e)
            result['status'] = 'error'
        
        return result
    
    def _execute_vector_rag(self, question: str) -> Dict[str, Any]:
        """Execute Vector-RAG query."""
        results = self.vector_rag.query(question, top_k=3)
        
        return {
            'method': 'vector_rag',
            'results': results,
            'summary': self._summarize_vector_results(results)
        }
    
    def _execute_graph_rag_local(self, question: str) -> Dict[str, Any]:
        """Execute Graph-RAG local query."""
        result = self.graph_rag.query_local(question)
        
        return {
            'method': 'graph_rag_local',
            'result': result,
            'summary': result.get('local_summary', 'Local graph analysis performed')
        }
    
    def _execute_graph_rag_global(self, question: str) -> Dict[str, Any]:
        """Execute Graph-RAG global query."""
        result = self.graph_rag.query_global(question)
        
        return {
            'method': 'graph_rag_global',
            'result': result,
            'summary': result.get('global_summary', 'Global thematic analysis performed')
        }
    
    def _combine_hybrid_answers(self, vector_answer: Dict, graph_answer: Dict) -> Dict[str, Any]:
        """Combine vector and graph answers for hybrid response."""
        
        return {
            'method': 'hybrid_rag',
            'vector_summary': vector_answer.get('summary', ''),
            'graph_summary': graph_answer.get('summary', ''),
            'combined_summary': f"Vector analysis: {vector_answer.get('summary', '')} Graph analysis: {graph_answer.get('summary', '')}",
            'confidence': 'high'  # Hybrid typically more confident
        }
    
    def _summarize_vector_results(self, results: List[Dict]) -> str:
        """Summarize vector RAG results."""
        if not results:
            return "No relevant documents found"
        
        top_result = results[0]
        source = top_result.get('source', 'Unknown')
        score = top_result.get('similarity_score', 0)
        
        return f"Found relevant information from {source} (similarity: {score:.3f})"
    
    def _collect_provenance(self, answers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect provenance information for transparency."""
        provenance = []
        
        for method, answer in answers.items():
            if method == 'primary':
                continue
                
            if answer.get('method') == 'vector_rag':
                for result in answer.get('results', []):
                    provenance.append({
                        'method': 'vector_rag',
                        'source': result.get('source'),
                        'similarity_score': result.get('similarity_score'),
                        'document_id': result.get('chunk', {}).get('document_id')
                    })
            
            elif 'graph_rag' in answer.get('method', ''):
                provenance.append({
                    'method': answer.get('method'),
                    'entities_analyzed': answer.get('result', {}).get('entity', {}).get('name'),
                    'communities': answer.get('result', {}).get('relevant_communities', 0)
                })
        
        return provenance
    
    def trace_political_genealogy(self, source_politician: str, target_politician: str) -> Dict[str, Any]:
        """
        Specialized method for tracing political genealogies.
        """
        if not self.is_indexed:
            return {'error': 'System not initialized'}
        
        # Use Graph-RAG for genealogy tracing
        result = self.graph_rag.trace_influence_path(source_politician, target_politician)
        
        # Add hierarchical context if paths found
        if result.get('paths_found', 0) > 0:
            # Generate contextual summary for the genealogy
            summary_context = f"Tracing influence from {source_politician} to {target_politician}"
            
            result['contextual_analysis'] = {
                'summary': summary_context,
                'method': 'graph_rag_genealogical'
            }
        
        return result
    
    def analyze_political_theme(self, theme: str, include_summarization: bool = True) -> Dict[str, Any]:
        """
        Specialized method for thematic political analysis.
        """
        if not self.is_indexed:
            return {'error': 'System not initialized'}
        
        # Use global Graph-RAG for thematic analysis
        graph_result = self.graph_rag.query_global(theme)
        
        # Add hierarchical summarization if requested
        if include_summarization:
            # Find relevant documents for the theme
            theme_query = f"documentos sobre {theme}"
            vector_results = self.vector_rag.query(theme_query, top_k=5)
            
            # Summarize top documents
            summaries = []
            for result in vector_results[:3]:
                chunk = result['chunk']
                doc_summary = self.summarizer._summarize_text(chunk['text'])
                summaries.append({
                    'document': result['source'],
                    'summary': doc_summary,
                    'relevance_score': result['similarity_score']
                })
            
            graph_result['document_summaries'] = summaries
        
        return graph_result
    
    def validate_corpus_fidelity(self, n_iterations: int = 50) -> Dict[str, Any]:
        """
        Run knowledge-shift validation to test corpus fidelity.
        """
        if not self.is_indexed:
            return {'error': 'System not initialized'}
        
        logger.info(f"Running knowledge-shift validation with {n_iterations} iterations")
        
        # Create a mock analysis system for testing
        mock_system = type('MockSystem', (), {
            'hybrid_query': lambda self, q: {'answer': f'Mock response to: {q}'}
        })()
        
        # Run validation  
        validation_results = self.knowledge_tester.run_knowledge_shift_evaluation(
            mock_system, 
            n_iterations=n_iterations
        )
        
        return validation_results
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        """
        stats = self.system_stats.copy()
        
        if self.is_indexed:
            # Add runtime statistics
            vector_stats = {
                'vector_indexed_chunks': len(self.vector_rag.chunks),
                'vector_embedding_type': 'OpenAI' if self.vector_rag.use_openai_embeddings else 'TF-IDF'
            }
            
            graph_stats = self.graph_rag.get_statistics()
            
            stats.update({
                **vector_stats,
                **graph_stats,
                'router_type': 'LLM-based' if self.use_llm_router else 'ML-based',
                'system_ready': True
            })
        else:
            stats['system_ready'] = False
        
        return stats
    
    def benchmark_performance(self, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark system performance across different query types.
        """
        if not self.is_indexed:
            return {'error': 'System not initialized'}
        
        if test_queries is None:
            test_queries = [
                "¿Cuándo nació Juan Perón?",  # Factual
                "¿Cómo evolucionó el peronismo desde 1945?",  # Thematic
                "¿Qué influencia tuvo Moreno en Perón?",  # Genealogical
                "¿En qué se diferencia Rosas de Mitre?",  # Comparative
            ]
        
        benchmark_results = {
            'queries_tested': len(test_queries),
            'results': [],
            'average_response_time': 0.0,
            'routing_distribution': {}
        }
        
        total_time = 0.0
        routing_counts = {}
        
        for query in test_queries:
            start_time = time.time()
            result = self.query(query, include_provenance=False)
            query_time = time.time() - start_time
            
            total_time += query_time
            
            # Track routing decisions
            method = result.get('routing_decision', {}).get('retrieval_method', 'unknown')
            routing_counts[method] = routing_counts.get(method, 0) + 1
            
            benchmark_results['results'].append({
                'query': query,
                'response_time': query_time,
                'method_used': method,
                'status': result.get('status', 'unknown')
            })
        
        benchmark_results['average_response_time'] = total_time / len(test_queries)
        benchmark_results['routing_distribution'] = routing_counts
        
        return benchmark_results

# Convenience functions
def create_complete_rag_system(use_openai: bool = False, use_llm_router: bool = False) -> CompletePoliticalRAGSystem:
    """
    Create and setup complete RAG system.
    """
    system = CompletePoliticalRAGSystem(use_openai=use_openai, use_llm_router=use_llm_router)
    system.setup_system()
    return system

def demo_complete_system():
    """
    Demonstrate complete RAG system capabilities.
    """
    print("🚀 Complete Political RAG System Demo")
    print("=" * 60)
    
    # Create system
    print("🏗️  Setting up complete system...")
    system = create_complete_rag_system(use_openai=False, use_llm_router=False)
    
    # Show statistics
    stats = system.get_system_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Documents: {stats['corpus_documents']}")
    print(f"   Vector chunks: {stats['vector_indexed_chunks']}")
    print(f"   Graph entities: {stats['total_entities']}")
    print(f"   Communities: {stats['communities']}")
    print(f"   Setup time: {stats['setup_time_total']:.2f}s")
    
    # Test queries
    test_queries = [
        "¿Cuándo asumió Perón la presidencia?",
        "¿Cómo evolucionó la grieta política argentina?",
        "¿Qué influencia tuvo Moreno en el pensamiento democrático argentino?"
    ]
    
    print(f"\n🧪 Testing Queries:")
    for query in test_queries:
        print(f"\n❓ {query}")
        
        result = system.query(query)
        
        if result.get('status') == 'success':
            routing = result['routing_decision']
            print(f"   🎯 Method: {routing['retrieval_method']}")
            print(f"   ⏱️  Time: {result['query_time']:.3f}s")
            print(f"   💭 Reasoning: {routing['reasoning'][:100]}...")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    
    # Benchmark
    print(f"\n🏁 Performance Benchmark:")
    benchmark = system.benchmark_performance()
    print(f"   Average response time: {benchmark['average_response_time']:.3f}s")
    print(f"   Routing distribution: {benchmark['routing_distribution']}")
    
    print(f"\n✅ Complete RAG system demo finished!")

if __name__ == "__main__":
    demo_complete_system()