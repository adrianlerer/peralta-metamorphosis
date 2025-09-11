#!/usr/bin/env python3
"""
Vector-RAG Integration for Political Analysis
Implementation based on "Branching and Merging" paper methodologies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from pathlib import Path

# For advanced embeddings (if available)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

class PoliticalDocumentChunker:
    """
    Hierarchical document chunking for political texts following paper methodology.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a political document into smaller segments with metadata.
        """
        text = document.get('text', '')
        doc_id = document.get('document_id', 'unknown')
        author = document.get('author', 'unknown')
        year = document.get('year', 0)
        
        # Split by sentences first for better semantic boundaries
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                    'document_id': doc_id,
                    'author': author,
                    'year': year,
                    'chunk_index': chunk_idx,
                    'text': current_chunk.strip(),
                    'word_count': current_length
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_length = len(overlap_text.split()) + sentence_length
                chunk_idx += 1
            else:
                current_chunk += " " + sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                'document_id': doc_id,
                'author': author,
                'year': year,
                'chunk_index': chunk_idx,
                'text': current_chunk.strip(),
                'word_count': current_length
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Basic sentence splitting for Spanish political texts."""
        import re
        
        # Spanish sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of current chunk."""
        words = text.split()
        overlap_words = words[-self.overlap:] if len(words) > self.overlap else words
        return " ".join(overlap_words)

class PoliticalVectorRAG:
    """
    Vector-based RAG system for fast political fact retrieval.
    """
    
    def __init__(self, use_openai_embeddings: bool = False):
        self.use_openai_embeddings = use_openai_embeddings and HAS_OPENAI
        
        # Enhanced OpenAI configuration  
        if self.use_openai_embeddings:
            self.embedding_model = "text-embedding-3-small"
            self.embedding_batch_size = 50  # Reduced for rate limits
        self.chunker = PoliticalDocumentChunker()
        self.chunks = []
        self.embeddings = None
        self.vectorizer = None
        
        if self.use_openai_embeddings:
            logger.info("Using OpenAI embeddings for Vector-RAG")
        else:
            logger.info("Using TF-IDF embeddings for Vector-RAG")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=None,  # Keep political stop words
                ngram_range=(1, 3),
                min_df=1
            )
    
    def index_political_corpus(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index political corpus for vector retrieval.
        """
        logger.info(f"Indexing {len(documents)} political documents")
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Create embeddings
        if self.use_openai_embeddings:
            self._create_openai_embeddings()
        else:
            self._create_tfidf_embeddings()
        
        logger.info("Vector indexing complete")
    
    def _create_tfidf_embeddings(self):
        """Create TF-IDF embeddings for chunks."""
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        self.embeddings = self.vectorizer.fit_transform(chunk_texts)
    
    def _create_openai_embeddings(self):
        """Create OpenAI embeddings for chunks."""
        if not HAS_OPENAI:
            raise ImportError("OpenAI package not available")
        
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        
        # Batch process embeddings (OpenAI has rate limits)
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            
            try:
                response = openai.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                # Fallback to TF-IDF for this batch
                fallback_vectorizer = TfidfVectorizer()
                fallback_embeddings = fallback_vectorizer.fit_transform(batch)
                embeddings.extend(fallback_embeddings.toarray())
        
        self.embeddings = np.array(embeddings)
    
    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector index for relevant political information.
        """
        if self.embeddings is None:
            raise ValueError("Corpus not indexed. Call index_political_corpus() first.")
        
        # Embed the question
        if self.use_openai_embeddings:
            question_embedding = self._embed_query_openai(question)
            similarities = cosine_similarity([question_embedding], self.embeddings)[0]
        else:
            question_embedding = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            result = {
                'chunk': chunk,
                'similarity_score': float(similarities[idx]),
                'source': f"{chunk['author']} ({chunk['year']})",
                'retrieval_method': 'vector_rag'
            }
            results.append(result)
        
        return results
    
    def _embed_query_openai(self, query: str) -> List[float]:
        """Embed a single query using OpenAI."""
        try:
            response = openai.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI query embedding error: {e}")
            # Fallback to TF-IDF
            return self.vectorizer.transform([query]).toarray()[0]

class PoliticalHierarchicalSummarizer:
    """
    Hierarchical summarization for long political documents.
    """
    
    def __init__(self):
        self.chunk_size = 10  # Chunks per section
    
    def summarize_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create hierarchical summary following paper methodology.
        """
        chunker = PoliticalDocumentChunker()
        chunks = chunker.chunk_document(document)
        
        if len(chunks) <= 1:
            return {
                'global_summary': document.get('text', '')[:500],
                'section_summaries': [],
                'chunk_summaries': [chunk['text'][:200] for chunk in chunks]
            }
        
        # Group chunks into sections
        sections = self._group_chunks_into_sections(chunks)
        
        # Summarize each section
        section_summaries = []
        for section in sections:
            section_text = " ".join([chunk['text'] for chunk in section])
            section_summary = self._summarize_text(section_text, target_length=200)
            section_summaries.append(section_summary)
        
        # Create global summary from section summaries
        all_sections_text = " ".join(section_summaries)
        global_summary = self._summarize_text(all_sections_text, target_length=300)
        
        return {
            'global_summary': global_summary,
            'section_summaries': section_summaries,
            'chunk_summaries': [chunk['text'][:100] for chunk in chunks],
            'sections': sections
        }
    
    def _group_chunks_into_sections(self, chunks: List[Dict]) -> List[List[Dict]]:
        """Group chunks into logical sections."""
        sections = []
        current_section = []
        
        for chunk in chunks:
            current_section.append(chunk)
            
            if len(current_section) >= self.chunk_size:
                sections.append(current_section)
                current_section = []
        
        # Add remaining chunks
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _summarize_text(self, text: str, target_length: int = 200) -> str:
        """
        Simple extractive summarization.
        In production, replace with LLM-based summarization.
        """
        sentences = text.split('.')
        
        # Score sentences by TF-IDF
        if len(sentences) <= 3:
            return text[:target_length]
        
        # Simple heuristic: take first, middle, and last sentences
        key_sentences = [
            sentences[0].strip(),
            sentences[len(sentences)//2].strip(),
            sentences[-1].strip() if sentences[-1].strip() else sentences[-2].strip()
        ]
        
        summary = ". ".join(key_sentences)
        return summary[:target_length]

class PoliticalQueryRouter:
    """
    Route queries to appropriate retrieval method based on paper methodology.
    """
    
    def __init__(self):
        # Simple rule-based router (upgrade to LLM-based later)
        self.factual_keywords = [
            'cuándo', 'dónde', 'qué año', 'fecha', 'quién', 'cómo se llama',
            'nombre', 'ley', 'decreto', 'artículo', 'constitución'
        ]
        
        self.thematic_keywords = [
            'evolución', 'desarrollo', 'influencia', 'genealogía', 'relación',
            'comparar', 'analizar', 'tendencia', 'cambio', 'transformación',
            'ideología', 'movimiento', 'corriente'
        ]
    
    def route_query(self, query: str) -> str:
        """
        Determine which retrieval method to use.
        """
        query_lower = query.lower()
        
        # Count keyword matches
        factual_score = sum(1 for kw in self.factual_keywords if kw in query_lower)
        thematic_score = sum(1 for kw in self.thematic_keywords if kw in query_lower)
        
        # Simple routing logic
        if factual_score > thematic_score:
            return 'vector_rag'  # Fast, precise retrieval
        elif thematic_score > factual_score:
            return 'graph_rag'   # Deep, multi-hop analysis
        else:
            return 'hybrid'      # Use both methods
        
    def explain_routing_decision(self, query: str) -> Dict[str, Any]:
        """Provide explanation for routing decision."""
        query_lower = query.lower()
        
        factual_matches = [kw for kw in self.factual_keywords if kw in query_lower]
        thematic_matches = [kw for kw in self.thematic_keywords if kw in query_lower]
        
        decision = self.route_query(query)
        
        return {
            'decision': decision,
            'factual_score': len(factual_matches),
            'thematic_score': len(thematic_matches),
            'factual_matches': factual_matches,
            'thematic_matches': thematic_matches,
            'explanation': self._get_decision_explanation(decision, factual_matches, thematic_matches)
        }
    
    def _get_decision_explanation(self, decision: str, factual: List, thematic: List) -> str:
        """Generate human-readable explanation."""
        if decision == 'vector_rag':
            return f"Query appears factual (keywords: {', '.join(factual)}). Using fast vector retrieval."
        elif decision == 'graph_rag':
            return f"Query appears thematic (keywords: {', '.join(thematic)}). Using deep graph analysis."
        else:
            return "Query has mixed characteristics. Using hybrid approach."

# Integration with existing system
def integrate_vector_rag_with_political_analysis():
    """
    Integration function to add Vector-RAG to existing political analysis.
    """
    
    # Import existing components
    try:
        from .expanded_political_corpus import create_expanded_political_corpus
        from .integrate_political_analysis import IntegratedPoliticalAnalysis
        
        # Create enhanced analyzer with Vector-RAG
        class EnhancedPoliticalAnalysis(IntegratedPoliticalAnalysis):
            def __init__(self):
                super().__init__()
                self.vector_rag = PoliticalVectorRAG(use_openai_embeddings=False)
                self.router = PoliticalQueryRouter()
                self.summarizer = PoliticalHierarchicalSummarizer()
                
            def setup_vector_rag(self):
                """Initialize vector RAG with political corpus."""
                documents = create_expanded_political_corpus()
                documents_dict = documents.to_dict('records')
                self.vector_rag.index_political_corpus(documents_dict)
                logger.info("Vector-RAG setup complete")
            
            def hybrid_query(self, question: str) -> Dict[str, Any]:
                """
                Answer political questions using hybrid RAG approach.
                """
                # Route the query
                routing_decision = self.router.explain_routing_decision(question)
                method = routing_decision['decision']
                
                results = {
                    'question': question,
                    'routing': routing_decision,
                    'answers': {}
                }
                
                # Execute appropriate retrieval
                if method in ['vector_rag', 'hybrid']:
                    vector_results = self.vector_rag.query(question, top_k=3)
                    results['answers']['vector_rag'] = vector_results
                
                if method in ['graph_rag', 'hybrid']:
                    # Use existing genealogical analysis as graph-rag
                    graph_results = self._graph_rag_query(question)
                    results['answers']['graph_rag'] = graph_results
                
                return results
            
            def _graph_rag_query(self, question: str) -> List[Dict]:
                """
                Use existing genealogical analysis as graph-based retrieval.
                """
                # This would integrate with existing genealogy tracing
                # For now, return placeholder
                return [{
                    'method': 'genealogical_analysis',
                    'result': 'Graph-based analysis would go here',
                    'retrieval_method': 'graph_rag'
                }]
        
        return EnhancedPoliticalAnalysis
        
    except ImportError as e:
        logger.error(f"Could not integrate with existing system: {e}")
        return None

if __name__ == "__main__":
    # Demo of Vector-RAG functionality
    print("🔍 Political Vector-RAG Demo")
    
    # Sample political documents
    sample_docs = [
        {
            'document_id': 'test_doc_1',
            'author': 'Juan Perón',
            'year': 1945,
            'text': 'La justicia social es el fundamento de nuestro movimiento. Los trabajadores tienen derechos inalienables.'
        },
        {
            'document_id': 'test_doc_2', 
            'author': 'Raúl Alfonsín',
            'year': 1983,
            'text': 'Con la democracia se come, se cura y se educa. La constitución es nuestra guía fundamental.'
        }
    ]
    
    # Initialize Vector-RAG
    vector_rag = PoliticalVectorRAG()
    vector_rag.index_political_corpus(sample_docs)
    
    # Test queries
    queries = [
        "¿Qué dijo Perón sobre justicia social?",
        "¿Cuál es la frase famosa de Alfonsín sobre democracia?"
    ]
    
    router = PoliticalQueryRouter()
    
    for query in queries:
        print(f"\n📋 Query: {query}")
        
        # Show routing decision
        routing = router.explain_routing_decision(query)
        print(f"🎯 Routing: {routing['decision']} - {routing['explanation']}")
        
        # Get results
        results = vector_rag.query(query, top_k=2)
        print("📄 Results:")
        for i, result in enumerate(results):
            print(f"   {i+1}. {result['source']} (score: {result['similarity_score']:.3f})")
            print(f"      \"{result['chunk']['text'][:100]}...\"")
    
    print("\n✅ Vector-RAG demo complete!")