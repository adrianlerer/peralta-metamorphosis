#!/usr/bin/env python3
"""
COMPREHENSIVE POLITICAL ANALYSIS EXECUTION
Using Enhanced RAG System with Graph-RAG and ML Query Routing

This script executes the complete analysis requested, leveraging all 
improvements from the 'Branching and Merging' paper implementation.
"""

import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append('/home/user/webapp')

from political_analysis.complete_rag_system import CompletePoliticalRAGSystem
from political_analysis.political_memespace import PoliticalMemespace
from political_analysis.political_rootfinder import PoliticalRootFinder

class ComprehensivePoliticalAnalysis:
    """
    Comprehensive analysis orchestrator using enhanced RAG system
    """
    
    def __init__(self):
        """Initialize all analysis components"""
        print("🚀 Initializing Comprehensive Political Analysis System...")
        
        # Initialize core RAG system
        self.rag_system = CompletePoliticalRAGSystem()
        
        # Initialize additional analysis tools
        self.memespace = PoliticalMemespace()
        self.rootfinder = PoliticalRootFinder()
        
        # Analysis results storage
        self.analysis_results = {
            'execution_timestamp': datetime.now().isoformat(),
            'system_performance': {},
            'antagonisms_analysis': {},
            'genealogical_tracings': {},
            'community_detection': {},
            'factional_analysis': {},
            'cross_temporal_patterns': {},
            'narrative_coherence': {},
            'recommendations': {}
        }
        
        print("✅ All systems initialized successfully")
    
    def setup_and_index_corpus(self):
        """Setup and index the political corpus"""
        print("\n📚 Setting up and indexing political corpus...")
        start_time = time.time()
        
        # Setup the system with full indexing
        self.rag_system.setup_system()
        
        setup_time = time.time() - start_time
        self.analysis_results['system_performance']['corpus_setup_time'] = setup_time
        
        # Get system statistics
        system_stats = self.rag_system.get_system_statistics()
        self.analysis_results['system_performance']['system_stats'] = system_stats
        
        print(f"✅ Corpus indexed in {setup_time:.2f}s")
        print(f"📊 System contains {system_stats['corpus_documents']} documents")
        return True
    
    def analyze_core_antagonisms(self):
        """Analyze core political antagonisms using hybrid RAG"""
        print("\n🔍 Analyzing Core Political Antagonisms...")
        
        # Key antagonism queries to analyze
        antagonism_queries = [
            "¿Cuáles son los principales antagonismos políticos entre Cristina Kirchner y Mauricio Macri?",
            "¿Cómo se manifiesta el conflicto entre peronismo y anti-peronismo en la política argentina?",
            "¿Qué diferencias ideológicas fundamentales existen entre La Cámpora y PRO?",
            "¿Cuál es la naturaleza del antagonismo entre populismo y liberalismo en Argentina?",
            "¿Cómo se expresan los conflictos entre kirchnerismo y macrismo en políticas concretas?",
            "¿Qué rol juegan los antagonismos históricos en la polarización política actual?",
            "¿Cuáles son los puntos de fricción entre el modelo económico peronista y el liberal?",
            "¿Cómo se manifiestan los antagonismos en el discurso político mediático argentino?"
        ]
        
        antagonism_results = {}
        
        for query in antagonism_queries:
            print(f"  🔎 Analyzing: {query[:50]}...")
            
            start_time = time.time()
            result = self.rag_system.query(query)
            query_time = time.time() - start_time
            
            # Store result with metadata
            primary_answer = result.get('answers', {}).get('primary', {})
            antagonism_results[query] = {
                'answer': primary_answer.get('combined_summary', primary_answer.get('vector_summary', 'No answer available')),
                'routing_decision': result.get('routing_decision', {}),
                'query_type': result.get('routing_decision', {}).get('query_type', 'unknown'),
                'response_time': query_time,
                'confidence': primary_answer.get('confidence', 'medium'),
                'provenance': result.get('provenance', [])
            }
            
            print(f"    ✅ Completed in {query_time:.3f}s")
        
        self.analysis_results['antagonisms_analysis'] = antagonism_results
        print(f"\n✅ Core antagonisms analysis completed ({len(antagonism_queries)} queries)")
        return antagonism_results
    
    def perform_genealogical_tracing(self):
        """Perform detailed genealogical tracing of political lineages"""
        print("\n🌳 Performing Genealogical Tracing...")
        
        # Key political figures for genealogical analysis
        genealogy_queries = [
            "Trace la genealogía política de Cristina Fernández de Kirchner desde sus orígenes hasta la actualidad",
            "¿Cuál es la genealogía política de Mauricio Macri y sus influencias formativas?",
            "Analice la genealogía del movimiento La Cámpora desde su fundación",
            "¿Cuál es el linaje político del PRO y sus antecedentes institucionales?",
            "Trace la evolución genealógica del peronismo desde Perón hasta el kirchnerismo",
            "¿Cuál es la genealogía del radicalismo argentino y sus transformaciones?",
            "Analice los linajes políticos que confluyen en Juntos por el Cambio",
            "¿Cómo se puede trazar la genealogía de los movimientos de izquierda en Argentina?"
        ]
        
        genealogical_results = {}
        
        for query in genealogy_queries:
            print(f"  🔍 Tracing: {query[:40]}...")
            
            # Use regular query for genealogical analysis
            start_time = time.time()
            result = self.rag_system.query(query)
            trace_time = time.time() - start_time
            
            primary_answer = result.get('answers', {}).get('primary', {})
            genealogical_results[query] = {
                'genealogy': primary_answer.get('combined_summary', primary_answer.get('vector_summary', 'No genealogy available')),
                'routing_decision': result.get('routing_decision', {}),
                'query_type': result.get('routing_decision', {}).get('query_type', 'unknown'),
                'confidence': primary_answer.get('confidence', 'medium'),
                'tracing_time': trace_time,
                'provenance': result.get('provenance', [])
            }
            
            print(f"    ✅ Traced in {trace_time:.3f}s")
        
        self.analysis_results['genealogical_tracings'] = genealogical_results
        print(f"\n✅ Genealogical tracing completed ({len(genealogy_queries)} traces)")
        return genealogical_results
    
    def detect_political_communities(self):
        """Perform community detection and factional analysis"""
        print("\n🏘️ Performing Community Detection and Factional Analysis...")
        
        # Get community detection results from graph RAG's knowledge graph
        communities = getattr(self.rag_system.graph_rag.knowledge_graph, 'communities', {})
        
        # Analyze factional structures
        factional_queries = [
            "¿Cuáles son las principales facciones dentro del peronismo actual?",
            "¿Qué grupos internos existen en Juntos por el Cambio?",
            "¿Cómo se organizan las facciones dentro de La Cámpora?",
            "¿Qué divisiones internas caracterizan al PRO?",
            "¿Cuáles son las líneas internas del radicalismo contemporáneo?",
            "¿Qué facciones componen el Frente de Todos?"
        ]
        
        factional_results = {}
        for query in factional_queries:
            result = self.rag_system.query(query)
            primary_answer = result.get('answers', {}).get('primary', {})
            factional_results[query] = {
                'answer': primary_answer.get('combined_summary', primary_answer.get('vector_summary', 'No answer available')),
                'routing_decision': result.get('routing_decision', {}),
                'confidence': primary_answer.get('confidence', 'medium')
            }
        
        self.analysis_results['community_detection'] = {
            'detected_communities': communities,
            'community_count': len(communities),
            'factional_analysis': factional_results
        }
        
        self.analysis_results['factional_analysis'] = factional_results
        
        print(f"✅ Detected {len(communities)} political communities")
        print(f"✅ Analyzed {len(factional_queries)} factional structures")
        return communities, factional_results
    
    def analyze_cross_temporal_patterns(self):
        """Analyze cross-temporal patterns and evolution"""
        print("\n⏰ Analyzing Cross-Temporal Patterns...")
        
        temporal_queries = [
            "¿Cómo han evolucionado los antagonismos políticos argentinos desde 2003 hasta 2023?",
            "¿Qué patrones temporales se observan en los conflictos entre peronismo y oposición?",
            "¿Cómo se han transformado las estrategias discursivas de los principales actores políticos?",
            "¿Qué ciclos políticos se pueden identificar en las últimas dos décadas argentinas?",
            "¿Cómo han cambiado las alianzas y enemistades políticas a lo largo del tiempo?",
            "¿Qué continuidades y rupturas se observan en el sistema político argentino?"
        ]
        
        temporal_results = {}
        for query in temporal_queries:
            result = self.rag_system.query(query)
            primary_answer = result.get('answers', {}).get('primary', {})
            temporal_results[query] = {
                'answer': primary_answer.get('combined_summary', primary_answer.get('vector_summary', 'No answer available')),
                'routing_decision': result.get('routing_decision', {}),
                'confidence': primary_answer.get('confidence', 'medium')
            }
        
        self.analysis_results['cross_temporal_patterns'] = temporal_results
        
        print(f"✅ Analyzed {len(temporal_queries)} temporal patterns")
        return temporal_results
    
    def assess_narrative_coherence(self):
        """Assess narrative coherence using knowledge shift testing"""
        print("\n📖 Assessing Narrative Coherence...")
        
        # Use the knowledge shift tester to assess corpus fidelity
        shift_results = self.rag_system.knowledge_tester.run_knowledge_shift_evaluation(
            self.rag_system
        )
        
        # Analyze narrative consistency
        coherence_queries = [
            "¿Qué tan consistente es la narrativa sobre Cristina Kirchner en diferentes fuentes?",
            "¿Existe coherencia en la descripción de las políticas de Mauricio Macri?",
            "¿Cómo de consistente es la caracterización del peronismo en el corpus?",
            "¿Qué nivel de coherencia narrativa existe sobre La Cámpora?"
        ]
        
        coherence_results = {}
        for query in coherence_queries:
            result = self.rag_system.query(query)
            primary_answer = result.get('answers', {}).get('primary', {})
            coherence_results[query] = {
                'answer': primary_answer.get('combined_summary', primary_answer.get('vector_summary', 'No answer available')),
                'routing_decision': result.get('routing_decision', {}),
                'confidence': primary_answer.get('confidence', 'medium')
            }
        
        self.analysis_results['narrative_coherence'] = {
            'knowledge_shift_test': shift_results,
            'narrative_consistency': coherence_results
        }
        
        print("✅ Narrative coherence assessment completed")
        return shift_results, coherence_results
    
    def generate_recommendations(self):
        """Generate strategic recommendations based on analysis"""
        print("\n💡 Generating Strategic Recommendations...")
        
        recommendation_queries = [
            "¿Qué estrategias podrían reducir la polarización política argentina?",
            "¿Cómo se podrían construir puentes entre los principales antagonismos políticos?",
            "¿Qué reformas institucionales ayudarían a moderar los conflictos políticos?",
            "¿Qué rol pueden jugar los medios en la desescalada de antagonismos?",
            "¿Cómo se puede promover el diálogo entre facciones opuestas?"
        ]
        
        recommendations = {}
        for query in recommendation_queries:
            result = self.rag_system.query(query)
            primary_answer = result.get('answers', {}).get('primary', {})
            recommendations[query] = {
                'answer': primary_answer.get('combined_summary', primary_answer.get('vector_summary', 'No recommendation available')),
                'routing_decision': result.get('routing_decision', {}),
                'confidence': primary_answer.get('confidence', 'medium')
            }
        
        self.analysis_results['recommendations'] = recommendations
        
        print(f"✅ Generated {len(recommendations)} strategic recommendations")
        return recommendations
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n📊 Generating Comprehensive Report...")
        
        # Calculate overall performance metrics
        total_queries = (
            len(self.analysis_results.get('antagonisms_analysis', {})) +
            len(self.analysis_results.get('genealogical_tracings', {})) +
            len(self.analysis_results.get('factional_analysis', {})) +
            len(self.analysis_results.get('cross_temporal_patterns', {})) +
            len(self.analysis_results.get('narrative_coherence', {}).get('narrative_consistency', {})) +
            len(self.analysis_results.get('recommendations', {}))
        )
        
        self.analysis_results['system_performance']['total_queries_executed'] = total_queries
        
        # Get final system statistics
        final_stats = self.rag_system.get_system_statistics()
        self.analysis_results['system_performance']['final_statistics'] = final_stats
        
        # Save results to file
        results_file = f"/home/user/webapp/political_analysis/comprehensive_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Comprehensive report saved to: {results_file}")
        return results_file, self.analysis_results
    
    def execute_full_analysis(self):
        """Execute the complete comprehensive analysis"""
        print("=" * 80)
        print("🎯 EXECUTING COMPREHENSIVE POLITICAL ANALYSIS")
        print("   Using Enhanced RAG System with Graph-RAG and ML Query Routing")
        print("=" * 80)
        
        overall_start = time.time()
        
        try:
            # Step 1: Setup and indexing
            self.setup_and_index_corpus()
            
            # Step 2: Core antagonisms analysis
            self.analyze_core_antagonisms()
            
            # Step 3: Genealogical tracing
            self.perform_genealogical_tracing()
            
            # Step 4: Community detection and factional analysis
            self.detect_political_communities()
            
            # Step 5: Cross-temporal pattern analysis
            self.analyze_cross_temporal_patterns()
            
            # Step 6: Narrative coherence assessment
            self.assess_narrative_coherence()
            
            # Step 7: Strategic recommendations
            self.generate_recommendations()
            
            # Step 8: Generate comprehensive report
            results_file, results = self.generate_comprehensive_report()
            
            total_time = time.time() - overall_start
            
            print("\n" + "=" * 80)
            print(f"🎉 COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"⏱️  Total execution time: {total_time:.2f} seconds")
            print(f"📄 Results saved to: {results_file}")
            print("=" * 80)
            
            return results_file, results
            
        except Exception as e:
            print(f"\n❌ ERROR during analysis execution: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main execution function"""
    analyzer = ComprehensivePoliticalAnalysis()
    results_file, results = analyzer.execute_full_analysis()
    
    if results_file:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 View results in: {results_file}")
    else:
        print(f"\n❌ Analysis failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())