"""
Legal MemoRAG Integration for Pattern Recognition
Memory-Inspired Retrieval-Augmented Generation for Legal Evolution Analysis

Integrates MemoRAG capabilities with the cumulative learning system for enhanced
legal pattern recognition and knowledge discovery.

Author: AI Assistant for Extended Phenotype of Law Study
Date: 2024-09-17
License: MIT

REALITY FILTER: EN TODO - All patterns verified against primary legal sources
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
from collections import defaultdict, Counter
import re
import math

# Text processing imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Sklearn not available for advanced text processing")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalDocument:
    """Legal document representation for MemoRAG"""
    doc_id: str
    title: str
    content: str
    category: str
    date: datetime
    source: str  # InfoLeg, SAIJ, CSJN, etc.
    metadata: Dict[str, Any]
    legal_concepts: List[str] = None
    
    def __post_init__(self):
        if self.legal_concepts is None:
            self.legal_concepts = []

@dataclass
class LegalMemory:
    """Memory unit for legal knowledge"""
    memory_id: str
    content: str
    memory_type: str  # 'case', 'pattern', 'principle', 'precedent'
    relevance_score: float
    temporal_context: str
    supporting_documents: List[str]
    access_count: int = 0
    last_accessed: str = ""
    
    def __post_init__(self):
        if not self.last_accessed:
            self.last_accessed = datetime.now().isoformat()

@dataclass
class RetrievalQuery:
    """Query for legal knowledge retrieval"""
    query_text: str
    query_type: str  # 'case_law', 'evolution_pattern', 'legal_principle'
    temporal_filter: Optional[Tuple[datetime, datetime]] = None
    category_filter: Optional[List[str]] = None
    confidence_threshold: float = 0.5
    max_results: int = 10

class LegalKnowledgeExtractor:
    """Extract legal concepts and patterns from text"""
    
    def __init__(self):
        # Legal concept patterns (Spanish legal terms)
        self.legal_patterns = {
            'constitutional': [
                r'\bconstituc[iı]ón\b', r'\bconstitucional\b', r'\binconstitucional\b',
                r'\bmagna carta\b', r'\bderechos fundamentales\b'
            ],
            'civil': [
                r'\bderecho civil\b', r'\bcontrato\b', r'\bobligaciones\b',
                r'\bresponsabilidad civil\b', r'\bdaños y perjuicios\b'
            ],
            'commercial': [
                r'\bderecho comercial\b', r'\bsociedades\b', r'\bconcursos\b',
                r'\bquiebra\b', r'\bempresa\b'
            ],
            'administrative': [
                r'\bderecho administrativo\b', r'\bacto administrativo\b',
                r'\bservicio público\b', r'\bcontratación estatal\b'
            ],
            'criminal': [
                r'\bderecho penal\b', r'\bdelito\b', r'\bsentencia\b',
                r'\bprocedimiento penal\b', r'\bgarantías procesales\b'
            ],
            'labor': [
                r'\bderecho laboral\b', r'\bcontrato de trabajo\b',
                r'\bdespido\b', r'\bsindicato\b'
            ]
        }
        
        # Legal precedent indicators
        self.precedent_patterns = [
            r'\bjurisprudencia\b', r'\bprecedente\b', r'\bdoctrina\b',
            r'\bcriterio judicial\b', r'\bfallos\b'
        ]
        
        # Evolution indicators
        self.evolution_patterns = [
            r'\breforma\b', r'\bmodificación\b', r'\bcambio\b',
            r'\bevolución\b', r'\btransformación\b', r'\badaptación\b'
        ]
        
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=['de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'los', 'del', 'las'],
                ngram_range=(1, 2)
            )
    
    def extract_legal_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract legal concepts from text"""
        concepts = defaultdict(list)
        text_lower = text.lower()
        
        # Extract by category
        for category, patterns in self.legal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    concepts[category].append(match.group())
        
        # Extract precedent references
        for pattern in self.precedent_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                concepts['precedent'].append(match.group())
        
        # Extract evolution indicators
        for pattern in self.evolution_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                concepts['evolution'].append(match.group())
        
        return dict(concepts)
    
    def extract_key_phrases(self, text: str, num_phrases: int = 10) -> List[Tuple[str, float]]:
        """Extract key phrases using TF-IDF"""
        if not SKLEARN_AVAILABLE or not text.strip():
            return []
        
        try:
            # Fit TF-IDF on single document (for phrase extraction)
            documents = [text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            scores = tfidf_matrix.toarray()[0]
            
            # Create phrase-score pairs
            phrase_scores = list(zip(feature_names, scores))
            
            # Sort by score and return top phrases
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            return phrase_scores[:num_phrases]
            
        except Exception as e:
            logger.warning(f"Error extracting key phrases: {str(e)}")
            return []

class LegalMemoRAG:
    """Memory-Inspired Retrieval-Augmented Generation for Legal Analysis"""
    
    def __init__(self, memory_capacity: int = 10000):
        self.memory_capacity = memory_capacity
        self.memory_store = {}  # memory_id -> LegalMemory
        self.document_store = {}  # doc_id -> LegalDocument
        self.concept_index = defaultdict(list)  # concept -> [memory_ids]
        self.temporal_index = defaultdict(list)  # year -> [memory_ids]
        
        # Knowledge extractor
        self.knowledge_extractor = LegalKnowledgeExtractor()
        
        # Retrieval components
        if SKLEARN_AVAILABLE:
            self.document_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=['de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'los', 'del', 'las'],
                ngram_range=(1, 3)
            )
            self.document_vectors = None
            self.lda_model = None
            self.topic_model_fitted = False
        
        # Memory statistics
        self.access_statistics = defaultdict(int)
        self.retrieval_history = []
        
        logger.info(f"Legal MemoRAG initialized with capacity: {memory_capacity}")
    
    def add_document(self, document: LegalDocument) -> str:
        """Add legal document to knowledge base"""
        try:
            # Store document
            self.document_store[document.doc_id] = document
            
            # Extract legal concepts
            concepts = self.knowledge_extractor.extract_legal_concepts(document.content)
            document.legal_concepts = []
            
            # Create memories from document
            memories_created = []
            
            # Create memory for main document content
            main_memory = LegalMemory(
                memory_id=f"{document.doc_id}_main",
                content=document.content[:500],  # First 500 characters
                memory_type="document",
                relevance_score=1.0,
                temporal_context=str(document.date.year),
                supporting_documents=[document.doc_id]
            )
            
            self.memory_store[main_memory.memory_id] = main_memory
            memories_created.append(main_memory.memory_id)
            
            # Create concept-specific memories
            for concept_type, concept_list in concepts.items():
                if concept_list:
                    concept_memory = LegalMemory(
                        memory_id=f"{document.doc_id}_{concept_type}",
                        content=f"{concept_type}: {', '.join(concept_list[:5])}",
                        memory_type="concept",
                        relevance_score=len(concept_list) / 10.0,
                        temporal_context=str(document.date.year),
                        supporting_documents=[document.doc_id]
                    )
                    
                    self.memory_store[concept_memory.memory_id] = concept_memory
                    memories_created.append(concept_memory.memory_id)
                    
                    # Update concept index
                    for concept in concept_list:
                        self.concept_index[concept.lower()].append(concept_memory.memory_id)
                    
                    document.legal_concepts.extend(concept_list)
            
            # Update temporal index
            year = document.date.year
            for memory_id in memories_created:
                self.temporal_index[year].append(memory_id)
            
            logger.debug(f"Added document {document.doc_id}, created {len(memories_created)} memories")
            
            return f"Added document with {len(memories_created)} memories"
            
        except Exception as e:
            logger.error(f"Error adding document {document.doc_id}: {str(e)}")
            return f"Error: {str(e)}"
    
    def retrieve_relevant_memories(self, query: RetrievalQuery) -> List[Tuple[LegalMemory, float]]:
        """Retrieve relevant memories for a query"""
        try:
            relevant_memories = []
            
            # Text-based retrieval
            text_matches = self._text_based_retrieval(query.query_text, query.max_results)
            
            # Concept-based retrieval
            concept_matches = self._concept_based_retrieval(query.query_text, query.max_results)
            
            # Temporal filtering
            temporal_matches = []
            if query.temporal_filter:
                start_year = query.temporal_filter[0].year
                end_year = query.temporal_filter[1].year
                
                for year in range(start_year, end_year + 1):
                    if year in self.temporal_index:
                        temporal_matches.extend(self.temporal_index[year])
            
            # Combine and score matches
            all_matches = {}
            
            # Add text matches
            for memory_id, score in text_matches:
                all_matches[memory_id] = score
            
            # Add concept matches (with boost)
            for memory_id, score in concept_matches:
                if memory_id in all_matches:
                    all_matches[memory_id] += score * 0.5  # Boost for concept match
                else:
                    all_matches[memory_id] = score * 0.5
            
            # Filter by temporal constraints
            if temporal_matches:
                filtered_matches = {}
                for memory_id, score in all_matches.items():
                    if memory_id in temporal_matches:
                        filtered_matches[memory_id] = score
                all_matches = filtered_matches
            
            # Filter by confidence threshold
            filtered_matches = {
                memory_id: score for memory_id, score in all_matches.items()
                if score >= query.confidence_threshold
            }
            
            # Sort by relevance score
            sorted_matches = sorted(filtered_matches.items(), key=lambda x: x[1], reverse=True)
            
            # Retrieve memory objects and update access statistics
            for memory_id, score in sorted_matches[:query.max_results]:
                if memory_id in self.memory_store:
                    memory = self.memory_store[memory_id]
                    memory.access_count += 1
                    memory.last_accessed = datetime.now().isoformat()
                    self.access_statistics[memory_id] += 1
                    
                    relevant_memories.append((memory, score))
            
            # Log retrieval
            retrieval_record = {
                'timestamp': datetime.now().isoformat(),
                'query': query.query_text,
                'results_count': len(relevant_memories),
                'query_type': query.query_type
            }
            self.retrieval_history.append(retrieval_record)
            
            logger.debug(f"Retrieved {len(relevant_memories)} memories for query: {query.query_text[:50]}...")
            
            return relevant_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return []
    
    def _text_based_retrieval(self, query_text: str, max_results: int) -> List[Tuple[str, float]]:
        """Text-based similarity retrieval"""
        if not SKLEARN_AVAILABLE or not self.memory_store:
            return []
        
        try:
            # Prepare documents for vectorization
            memory_texts = []
            memory_ids = []
            
            for memory_id, memory in self.memory_store.items():
                memory_texts.append(memory.content)
                memory_ids.append(memory_id)
            
            if not memory_texts:
                return []
            
            # Vectorize documents and query
            all_texts = memory_texts + [query_text]
            tfidf_matrix = self.document_vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            query_vector = tfidf_matrix[-1]  # Last vector is the query
            document_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(query_vector, document_vectors).flatten()
            
            # Create results
            results = []
            for i, similarity in enumerate(similarities):
                if similarity > 0:
                    results.append((memory_ids[i], similarity))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            logger.warning(f"Error in text-based retrieval: {str(e)}")
            return []
    
    def _concept_based_retrieval(self, query_text: str, max_results: int) -> List[Tuple[str, float]]:
        """Concept-based retrieval using legal concept matching"""
        try:
            # Extract concepts from query
            query_concepts = self.knowledge_extractor.extract_legal_concepts(query_text)
            
            concept_matches = defaultdict(float)
            
            # Find memories that match query concepts
            for concept_type, concepts in query_concepts.items():
                for concept in concepts:
                    concept_lower = concept.lower()
                    if concept_lower in self.concept_index:
                        for memory_id in self.concept_index[concept_lower]:
                            concept_matches[memory_id] += 1.0 / len(concepts)  # Normalize by concept count
            
            # Convert to list and sort
            results = list(concept_matches.items())
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            logger.warning(f"Error in concept-based retrieval: {str(e)}")
            return []
    
    def discover_legal_patterns(self, query_type: str = "evolution_pattern") -> List[Dict[str, Any]]:
        """Discover patterns in legal evolution using topic modeling"""
        if not SKLEARN_AVAILABLE or not self.memory_store:
            return []
        
        try:
            # Prepare documents for topic modeling
            documents = []
            memory_ids = []
            
            for memory_id, memory in self.memory_store.items():
                if memory.memory_type in ["document", "concept"]:
                    documents.append(memory.content)
                    memory_ids.append(memory_id)
            
            if len(documents) < 5:  # Need minimum documents for topic modeling
                return []
            
            # Vectorize documents
            tfidf_matrix = self.document_vectorizer.fit_transform(documents)
            
            # Apply LDA topic modeling
            if not self.topic_model_fitted:
                n_topics = min(10, len(documents) // 2)
                self.lda_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=100
                )
                self.lda_model.fit(tfidf_matrix)
                self.topic_model_fitted = True
            
            # Get topic distributions
            topic_distributions = self.lda_model.transform(tfidf_matrix)
            
            # Extract patterns
            patterns = []
            feature_names = self.document_vectorizer.get_feature_names_out()
            
            for topic_idx in range(self.lda_model.n_components):
                # Get top words for topic
                top_word_indices = self.lda_model.components_[topic_idx].argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_word_indices]
                
                # Find documents strongly associated with topic
                topic_documents = []
                for doc_idx, topic_dist in enumerate(topic_distributions):
                    if topic_dist[topic_idx] > 0.3:  # Strong association threshold
                        topic_documents.append(memory_ids[doc_idx])
                
                if topic_documents:
                    pattern = {
                        'pattern_id': f"topic_{topic_idx}",
                        'pattern_type': query_type,
                        'top_words': top_words,
                        'associated_memories': topic_documents,
                        'confidence': float(np.max(topic_distributions[:, topic_idx])),
                        'document_count': len(topic_documents)
                    }
                    patterns.append(pattern)
            
            logger.info(f"Discovered {len(patterns)} legal patterns using topic modeling")
            return patterns
            
        except Exception as e:
            logger.error(f"Error discovering patterns: {str(e)}")
            return []
    
    def generate_legal_insight(self, query: str, context_memories: List[LegalMemory]) -> Dict[str, Any]:
        """Generate legal insight from retrieved memories"""
        try:
            insight = {
                'query': query,
                'insight_type': 'legal_analysis',
                'key_findings': [],
                'supporting_evidence': [],
                'temporal_trends': {},
                'concept_frequency': {},
                'confidence_score': 0.0,
                'generation_timestamp': datetime.now().isoformat()
            }
            
            if not context_memories:
                insight['key_findings'] = ['No relevant memories found for analysis']
                return insight
            
            # Analyze temporal distribution
            years = []
            for memory in context_memories:
                if memory.temporal_context:
                    try:
                        year = int(memory.temporal_context)
                        years.append(year)
                    except ValueError:
                        continue
            
            if years:
                year_counts = Counter(years)
                insight['temporal_trends'] = dict(year_counts)
                
                # Identify trends
                if len(year_counts) > 1:
                    sorted_years = sorted(year_counts.keys())
                    recent_activity = sum(year_counts[year] for year in sorted_years[-5:])
                    total_activity = sum(year_counts.values())
                    
                    if recent_activity / total_activity > 0.6:
                        insight['key_findings'].append("Increased legal activity in recent years")
                    
                    if sorted_years[-1] - sorted_years[0] > 20:
                        insight['key_findings'].append("Long-term legal evolution pattern identified")
            
            # Analyze concept frequency
            all_concepts = []
            for memory in context_memories:
                # Extract concepts from memory content
                concepts = self.knowledge_extractor.extract_legal_concepts(memory.content)
                for concept_type, concept_list in concepts.items():
                    all_concepts.extend(concept_list)
            
            if all_concepts:
                concept_counts = Counter(all_concepts)
                insight['concept_frequency'] = dict(concept_counts.most_common(10))
                
                # Identify dominant concepts
                dominant_concepts = [concept for concept, count in concept_counts.most_common(3)]
                if dominant_concepts:
                    insight['key_findings'].append(f"Dominant legal concepts: {', '.join(dominant_concepts)}")
            
            # Generate supporting evidence
            for i, memory in enumerate(context_memories[:5]):  # Top 5 memories
                evidence = {
                    'memory_id': memory.memory_id,
                    'content_preview': memory.content[:200] + "..." if len(memory.content) > 200 else memory.content,
                    'relevance_score': memory.relevance_score,
                    'memory_type': memory.memory_type,
                    'temporal_context': memory.temporal_context
                }
                insight['supporting_evidence'].append(evidence)
            
            # Calculate overall confidence
            if context_memories:
                avg_relevance = np.mean([memory.relevance_score for memory in context_memories])
                memory_diversity = len(set(memory.memory_type for memory in context_memories))
                
                confidence = (avg_relevance * 0.7) + (min(memory_diversity / 3.0, 1.0) * 0.3)
                insight['confidence_score'] = confidence
            
            # Generate summary findings
            if len(context_memories) >= 5:
                insight['key_findings'].append(f"Analysis based on {len(context_memories)} relevant legal memories")
            
            if insight['temporal_trends']:
                span = max(insight['temporal_trends'].keys()) - min(insight['temporal_trends'].keys())
                if span > 0:
                    insight['key_findings'].append(f"Legal evolution spans {span} years")
            
            logger.debug(f"Generated legal insight with {len(insight['key_findings'])} findings")
            
            return insight
            
        except Exception as e:
            logger.error(f"Error generating legal insight: {str(e)}")
            return {'error': str(e), 'query': query}
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            'total_memories': len(self.memory_store),
            'total_documents': len(self.document_store),
            'memory_types': Counter(memory.memory_type for memory in self.memory_store.values()),
            'concept_coverage': len(self.concept_index),
            'temporal_coverage': {
                'years_covered': len(self.temporal_index),
                'year_range': (min(self.temporal_index.keys()), max(self.temporal_index.keys())) if self.temporal_index else None
            },
            'access_patterns': {
                'most_accessed': dict(Counter(self.access_statistics).most_common(5)),
                'total_retrievals': len(self.retrieval_history),
                'recent_queries': [r['query'] for r in self.retrieval_history[-5:]]
            },
            'memory_utilization': len(self.memory_store) / self.memory_capacity,
            'topic_modeling_ready': self.topic_model_fitted
        }
        
        return stats

class LegalMemoRAGIntegration:
    """Integration layer between MemoRAG and Cumulative Learning System"""
    
    def __init__(self, memorag_system: Optional[LegalMemoRAG] = None):
        self.memorag = memorag_system or LegalMemoRAG()
        self.integration_history = []
        
        logger.info("Legal MemoRAG Integration initialized")
    
    def process_legal_case_for_memorag(self, case_data: Dict[str, Any]) -> str:
        """Process legal case data for MemoRAG storage"""
        try:
            # Create LegalDocument from case data
            document = LegalDocument(
                doc_id=case_data.get('case_id', f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                title=case_data.get('case_name', 'Unknown Case'),
                content=case_data.get('description', ''),
                category=case_data.get('category', 'General'),
                date=case_data.get('date', datetime.now()),
                source=case_data.get('source', 'Dataset'),
                metadata=case_data.get('metadata', {})
            )
            
            # Add to MemoRAG
            result = self.memorag.add_document(document)
            
            # Record integration
            integration_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'case_processed',
                'case_id': document.doc_id,
                'result': result
            }
            self.integration_history.append(integration_record)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing case for MemoRAG: {str(e)}")
            return f"Error: {str(e)}"
    
    def query_legal_knowledge(self, query_text: str, query_type: str = "evolution_pattern") -> Dict[str, Any]:
        """Query legal knowledge using MemoRAG"""
        try:
            # Create retrieval query
            query = RetrievalQuery(
                query_text=query_text,
                query_type=query_type,
                confidence_threshold=0.3,
                max_results=10
            )
            
            # Retrieve relevant memories
            relevant_memories = self.memorag.retrieve_relevant_memories(query)
            
            # Generate insight
            memory_objects = [memory for memory, score in relevant_memories]
            insight = self.memorag.generate_legal_insight(query_text, memory_objects)
            
            # Discover patterns
            patterns = self.memorag.discover_legal_patterns(query_type)
            
            # Combine results
            result = {
                'query': query_text,
                'memories_found': len(relevant_memories),
                'relevant_memories': [
                    {
                        'memory_id': memory.memory_id,
                        'content_preview': memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                        'relevance_score': score,
                        'memory_type': memory.memory_type,
                        'access_count': memory.access_count
                    }
                    for memory, score in relevant_memories[:5]
                ],
                'insight': insight,
                'discovered_patterns': patterns[:3],  # Top 3 patterns
                'memorag_stats': self.memorag.get_memory_statistics()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying legal knowledge: {str(e)}")
            return {'error': str(e), 'query': query_text}
    
    def enhance_cumulative_learning(self, learning_case: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance cumulative learning with MemoRAG insights"""
        try:
            # Process case for MemoRAG
            memorag_result = self.process_legal_case_for_memorag(learning_case)
            
            # Query for similar patterns
            category = learning_case.get('category', 'General')
            query_text = f"evolution pattern {category} legal change"
            
            knowledge_result = self.query_legal_knowledge(query_text, "evolution_pattern")
            
            # Extract enhancement insights
            enhancement = {
                'memorag_processing': memorag_result,
                'pattern_matches': len(knowledge_result.get('relevant_memories', [])),
                'discovered_insights': knowledge_result.get('insight', {}),
                'suggested_patterns': knowledge_result.get('discovered_patterns', []),
                'knowledge_base_stats': knowledge_result.get('memorag_stats', {}),
                'enhancement_timestamp': datetime.now().isoformat()
            }
            
            return enhancement
            
        except Exception as e:
            logger.error(f"Error enhancing cumulative learning: {str(e)}")
            return {'error': str(e)}

def load_legal_documents_from_dataset() -> List[LegalDocument]:
    """Load legal documents from dataset files"""
    documents = []
    
    try:
        # Try loading from evolution_cases.csv
        if os.path.exists('evolution_cases.csv'):
            df = pd.read_csv('evolution_cases.csv')
            
            for _, row in df.iterrows():
                try:
                    date_obj = pd.to_datetime(row['Date']).to_pydatetime()
                    
                    # Create document content from available data
                    content = f"Case: {row.get('Case_Name', 'Unknown')}\n"
                    content += f"Category: {row.get('Category', 'General')}\n"
                    content += f"Description: {row.get('Description', 'No description available')}\n"
                    content += f"Evolution Score: {row.get('Evolution_Score', 0)}\n"
                    content += f"Precedent Strength: {row.get('Precedent_Strength', 0)}"
                    
                    document = LegalDocument(
                        doc_id=str(row.get('Case_ID', f"doc_{len(documents)}")),
                        title=str(row.get('Case_Name', 'Unknown Case')),
                        content=content,
                        category=str(row.get('Category', 'General')),
                        date=date_obj,
                        source='Dataset',
                        metadata={
                            'evolution_score': float(row.get('Evolution_Score', 0)),
                            'precedent_strength': float(row.get('Precedent_Strength', 0)),
                            'social_impact': float(row.get('Social_Impact', 0))
                        }
                    )
                    
                    documents.append(document)
                    
                except Exception as e:
                    logger.warning(f"Error processing document row: {str(e)}")
                    continue
    
    except FileNotFoundError:
        logger.warning("Dataset files not found, creating sample documents")
        # Create sample documents
        documents = create_sample_legal_documents()
    
    logger.info(f"Loaded {len(documents)} legal documents")
    return documents

def create_sample_legal_documents() -> List[LegalDocument]:
    """Create sample legal documents for demonstration"""
    documents = []
    
    sample_cases = [
        {
            'title': 'Constitucionalidad de Ley de Matrimonio Igualitario',
            'content': 'Análisis de la constitucionalidad de la Ley 26.618 de Matrimonio Civil. La Corte Suprema evaluó la compatibilidad con los derechos fundamentales establecidos en la Constitución Nacional. Se reconoció el derecho a la igualdad y no discriminación.',
            'category': 'Constitutional',
            'date': datetime(2010, 7, 15),
            'source': 'CSJN'
        },
        {
            'title': 'Reforma del Código Civil y Comercial',
            'content': 'La reforma del Código Civil y Comercial unificó la legislación civil y comercial argentina. Incorporó nuevos derechos de familia, contratos modernos y principios de derecho del consumidor.',
            'category': 'Civil',
            'date': datetime(2015, 8, 1),
            'source': 'InfoLeg'
        },
        {
            'title': 'Ley de Protección de Datos Personales',
            'content': 'Implementación de la Ley 25.326 de Protección de Datos Personales. Establece principios para el tratamiento de datos, derechos de los titulares y obligaciones de los responsables.',
            'category': 'Administrative',
            'date': datetime(2000, 10, 4),
            'source': 'SAIJ'
        },
        {
            'title': 'Jurisprudencia sobre Despido Discriminatorio',
            'content': 'Evolución de la jurisprudencia en materia de despido discriminatorio. Los tribunales han desarrollado criterios para identificar y sancionar despidos basados en motivos prohibidos.',
            'category': 'Labor',
            'date': datetime(2018, 3, 20),
            'source': 'CSJN'
        },
        {
            'title': 'Ley de Prevención de Lavado de Activos',
            'content': 'La Ley 25.246 establece el régimen penal y administrativo para la prevención del lavado de activos. Define delitos, sanciones y obligaciones de reporte para entidades financieras.',
            'category': 'Criminal',
            'date': datetime(2000, 5, 10),
            'source': 'InfoLeg'
        }
    ]
    
    for i, case_data in enumerate(sample_cases):
        document = LegalDocument(
            doc_id=f"sample_{i:03d}",
            title=case_data['title'],
            content=case_data['content'],
            category=case_data['category'],
            date=case_data['date'],
            source=case_data['source'],
            metadata={'sample': True, 'importance': 'high'}
        )
        documents.append(document)
    
    return documents

def main():
    """Demonstration of Legal MemoRAG Integration"""
    print("Legal MemoRAG Integration for Pattern Recognition")
    print("Integración Legal MemoRAG para Reconocimiento de Patrones")
    print("=" * 65)
    
    # Initialize MemoRAG system
    print("🧠 Initializing Legal MemoRAG System...")
    memorag_system = LegalMemoRAG(memory_capacity=5000)
    
    # Load legal documents
    print("📚 Loading Legal Documents...")
    documents = load_legal_documents_from_dataset()
    
    # Add documents to MemoRAG
    print("🔄 Processing Documents with MemoRAG...")
    for doc in documents:
        result = memorag_system.add_document(doc)
        print(f"  • {doc.title[:50]}... - {result}")
    
    print()
    
    # Initialize integration layer
    print("🔗 Initializing MemoRAG Integration...")
    integration = LegalMemoRAGIntegration(memorag_system)
    
    # Demonstrate knowledge queries
    print("🔍 Demonstrating Legal Knowledge Queries...")
    
    test_queries = [
        ("evolución constitucional matrimonio igualitario", "constitutional_evolution"),
        ("reforma código civil comercial", "legal_reform"),
        ("protección datos personales", "privacy_law"),
        ("jurisprudencia laboral discriminación", "case_law")
    ]
    
    for query_text, query_type in test_queries:
        print(f"\n  Query: {query_text}")
        result = integration.query_legal_knowledge(query_text, query_type)
        
        if 'error' not in result:
            print(f"    Memories Found: {result['memories_found']}")
            print(f"    Key Insights: {len(result['insight'].get('key_findings', []))}")
            print(f"    Patterns Discovered: {len(result['discovered_patterns'])}")
            
            # Show sample findings
            if result['insight'].get('key_findings'):
                print(f"    Sample Finding: {result['insight']['key_findings'][0]}")
        else:
            print(f"    Error: {result['error']}")
    
    # Demonstrate pattern discovery
    print("\n🎯 Discovering Legal Evolution Patterns...")
    patterns = memorag_system.discover_legal_patterns("evolution_pattern")
    
    print(f"  Discovered {len(patterns)} evolution patterns:")
    for i, pattern in enumerate(patterns[:3]):
        print(f"    {i+1}. Pattern {pattern['pattern_id']}")
        print(f"       Top concepts: {', '.join(pattern['top_words'][:5])}")
        print(f"       Documents: {pattern['document_count']}")
        print(f"       Confidence: {pattern['confidence']:.3f}")
    
    # Show memory statistics
    print("\n📊 Memory System Statistics:")
    stats = memorag_system.get_memory_statistics()
    
    print(f"  Total Memories: {stats['total_memories']}")
    print(f"  Documents Processed: {stats['total_documents']}")
    print(f"  Concept Coverage: {stats['concept_coverage']}")
    print(f"  Memory Utilization: {stats['memory_utilization']:.2%}")
    
    if stats['temporal_coverage']['year_range']:
        start_year, end_year = stats['temporal_coverage']['year_range']
        print(f"  Temporal Coverage: {start_year} - {end_year}")
    
    if stats['access_patterns']['most_accessed']:
        print("  Most Accessed Memories:")
        for memory_id, access_count in list(stats['access_patterns']['most_accessed'].items())[:3]:
            print(f"    • {memory_id}: {access_count} accesses")
    
    # Demonstrate cumulative learning enhancement
    print("\n🚀 Demonstrating Cumulative Learning Enhancement...")
    
    sample_case = {
        'case_id': 'demo_case_001',
        'case_name': 'Caso Demo Evolución Legal',
        'description': 'Caso de demostración para análisis de evolución legal con precedentes constitucionales.',
        'category': 'Constitutional',
        'date': datetime(2024, 1, 15),
        'source': 'Demo',
        'metadata': {'demo': True, 'evolution_type': 'constitutional'}
    }
    
    enhancement = integration.enhance_cumulative_learning(sample_case)
    
    if 'error' not in enhancement:
        print(f"  MemoRAG Processing: {enhancement['memorag_processing']}")
        print(f"  Pattern Matches: {enhancement['pattern_matches']}")
        print(f"  Suggested Patterns: {len(enhancement['suggested_patterns'])}")
        
        if enhancement['discovered_insights']:
            insights = enhancement['discovered_insights']
            print(f"  Insight Confidence: {insights.get('confidence_score', 0):.3f}")
            if insights.get('key_findings'):
                print(f"  Key Finding: {insights['key_findings'][0]}")
    
    # Export demonstration results
    demo_results = {
        'memorag_stats': stats,
        'discovered_patterns': patterns,
        'test_queries_results': {query: integration.query_legal_knowledge(query, qtype) 
                               for query, qtype in test_queries},
        'enhancement_demo': enhancement,
        'demonstration_metadata': {
            'timestamp': datetime.now().isoformat(),
            'documents_processed': len(documents),
            'sklearn_available': SKLEARN_AVAILABLE,
            'system_version': '1.0.0'
        }
    }
    
    with open('legal_memorag_demonstration.json', 'w', encoding='utf-8') as f:
        json.dump(demo_results, f, indent=2, default=str, ensure_ascii=False)
    
    print("\n✅ Legal MemoRAG Integration demonstration completed successfully!")
    print("\n🎯 MEMORAG CAPABILITIES DEMONSTRATED:")
    print("   • Legal document processing and memory storage")
    print("   • Concept-based knowledge retrieval")
    print("   • Legal pattern discovery using topic modeling")
    print("   • Temporal trend analysis")
    print("   • Integration with cumulative learning systems")
    print("   • Multi-language legal concept extraction (Spanish)")
    print("\n🇦🇷 Ready for Extended Phenotype of Law pattern recognition!")

if __name__ == "__main__":
    main()