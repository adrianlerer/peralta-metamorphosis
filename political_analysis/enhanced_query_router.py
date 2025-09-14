#!/usr/bin/env python3
"""
Enhanced LLM-based Query Router for Political Analysis
Intelligent routing using language models and advanced classification
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path

# LLM integration (if available)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of political queries."""
    FACTUAL = "factual"           # Specific facts, dates, names
    THEMATIC = "thematic"         # Broad analysis, evolution, trends
    GENEALOGICAL = "genealogical" # Influence tracing, political lineages
    COMPARATIVE = "comparative"   # Comparing politicians, movements, eras
    PROCEDURAL = "procedural"     # How-to, process questions
    HYBRID = "hybrid"            # Mixed characteristics

class RetrievalMethod(Enum):
    """RAG retrieval methods."""
    VECTOR_RAG = "vector_rag"
    GRAPH_RAG_LOCAL = "graph_rag_local"
    GRAPH_RAG_GLOBAL = "graph_rag_global"
    HYBRID_RAG = "hybrid_rag"

@dataclass
class RoutingDecision:
    """Structured routing decision with explanation."""
    query_type: QueryType
    retrieval_method: RetrievalMethod
    confidence: float
    reasoning: str
    alternative_methods: List[RetrievalMethod]
    query_features: Dict[str, Any]

class PoliticalQueryClassifier:
    """
    ML-based classifier for political query types.
    """
    
    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self.is_trained = False
        
    def create_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Create training data for query classification.
        """
        
        training_queries = [
            # Factual queries
            ("¿Cuándo nació Juan Perón?", "factual"),
            ("¿En qué año se promulgó la Ley Sáenz Peña?", "factual"),
            ("¿Dónde se sancionó la Constitución de 1853?", "factual"),
            ("¿Quién fue el primer presidente argentino?", "factual"),
            ("¿Cuál es el nombre completo de Eva Perón?", "factual"),
            ("¿Qué partido político fundó Hipólito Yrigoyen?", "factual"),
            ("¿Cuándo terminó el gobierno de Rosas?", "factual"),
            ("¿En qué provincia nació Sarmiento?", "factual"),
            ("¿Qué ley estableció el voto secreto y obligatorio?", "factual"),
            ("¿Cuántos años duró el Proceso militar?", "factual"),
            
            # Thematic queries  
            ("¿Cómo evolucionó el pensamiento federal argentino?", "thematic"),
            ("¿Cuáles fueron las principales características del peronismo?", "thematic"),
            ("¿Qué transformaciones sufrió el radicalismo en el siglo XX?", "thematic"),
            ("¿Cómo se desarrolló la grieta política argentina?", "thematic"),
            ("¿Cuál fue el impacto del liberalismo en Argentina?", "thematic"),
            ("¿Qué cambios políticos ocurrieron durante la década del 90?", "thematic"),
            ("¿Cómo influyó el contexto internacional en la política argentina?", "thematic"),
            ("¿Cuáles son los patrones de polarización en Argentina?", "thematic"),
            ("¿Qué rol jugaron los militares en la política argentina?", "thematic"),
            ("¿Cómo cambió el concepto de ciudadanía en Argentina?", "thematic"),
            
            # Genealogical queries
            ("¿Qué influencia tuvo Moreno en el pensamiento de Perón?", "genealogical"),
            ("¿Cómo se transmitieron las ideas federales de Artigas a Rosas?", "genealogical"),
            ("¿Cuál es la genealogía ideológica del kirchnerismo?", "genealogical"),
            ("¿Qué conecta intelectualmente a Alberdi con Mitre?", "genealogical"),
            ("¿Cómo influyó Sarmiento en los presidentes posteriores?", "genealogical"),
            ("¿Qué línea de pensamiento une a Yrigoyen con Alfonsín?", "genealogical"),
            ("¿Cuáles son las raíces históricas del liberalismo de Milei?", "genealogical"),
            ("¿Qué herencias políticas recibió Menem del peronismo?", "genealogical"),
            ("¿Cómo se trasmitió la tradición conservadora en Argentina?", "genealogical"),
            ("¿Qué influencias recibió el nacionalismo argentino?", "genealogical"),
            
            # Comparative queries
            ("¿Qué diferencias hay entre el peronismo y el radicalismo?", "comparative"),
            ("¿Cómo se compara Rosas con otros caudillos federales?", "comparative"),
            ("¿Qué similitudes tienen Perón y Yrigoyen como líderes populares?", "comparative"),
            ("¿En qué se diferencia el liberalismo de Mitre del de Milei?", "comparative"),
            ("¿Cómo se compara la política exterior de Menem y Kirchner?", "comparative"),
            ("¿Qué tienen en común los golpes de 1930, 1943 y 1976?", "comparative"),
            ("¿Cuáles son las diferencias entre unitarios y federales?", "comparative"),
            ("¿Cómo se compara el contexto de las crisis de 1989 y 2001?", "comparative"),
            ("¿Qué similitudes hay entre la Generación del 37 y la del 80?", "comparative"),
            ("¿En qué se parecen y diferencian Alfonsín y De la Rúa?", "comparative"),
            
            # Procedural queries
            ("¿Cómo se formó la Confederación Argentina?", "procedural"),
            ("¿Cuál fue el proceso de sanción de la Constitución de 1853?", "procedural"),
            ("¿Cómo llegó Perón al poder en 1946?", "procedural"),
            ("¿Qué pasos llevaron a la Revolución Libertadora?", "procedural"),
            ("¿Cómo se organizó la transición democrática de 1983?", "procedural"),
            ("¿Cuál fue el mecanismo del golpe de 1930?", "procedural"),
            ("¿Cómo funcionaba el sistema electoral antes de la Ley Sáenz Peña?", "procedural"),
            ("¿Qué proceso siguió la reforma constitucional de 1994?", "procedural"),
            ("¿Cómo se implementó el Plan de Convertibilidad?", "procedural"),
            ("¿Cuáles fueron los pasos de la nacionalización del petróleo?", "procedural"),
            
            # Hybrid queries (mix of types)
            ("¿Qué dijo Perón sobre justicia social y cómo influyó en gobiernos posteriores?", "hybrid"),
            ("¿Cuándo surgió el concepto de grieta y cómo evolucionó hasta hoy?", "hybrid"),
            ("¿Quién fue Mariano Moreno y qué relevancia tiene para el pensamiento democrático argentino?", "hybrid"),
            ("¿En qué se diferencia la Constitución de 1853 de la de 1949 y cuál fue su impacto?", "hybrid"),
            ("¿Cómo se compara el federalismo de Urquiza con el contexto actual?", "hybrid"),
        ]
        
        queries, labels = zip(*training_queries)
        return list(queries), list(labels)
    
    def train_classifier(self) -> None:
        """
        Train the query classifier using political training data.
        """
        logger.info("Training political query classifier")
        
        queries, labels = self.create_training_data()
        
        # Create pipeline with TF-IDF + Naive Bayes
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=1000,
                stop_words=None,  # Keep political terms
                lowercase=True
            )),
            ('nb', MultinomialNB(alpha=0.1))
        ])
        
        # Train
        self.classifier.fit(queries, labels)
        self.is_trained = True
        
        # Test accuracy
        predictions = self.classifier.predict(queries)
        accuracy = np.mean([p == l for p, l in zip(predictions, labels)])
        logger.info(f"Classifier trained with {len(queries)} examples, accuracy: {accuracy:.2%}")
    
    def classify_query(self, query: str) -> Tuple[str, float]:
        """
        Classify a query and return type with confidence.
        """
        if not self.is_trained:
            self.train_classifier()
        
        # Get prediction and probability
        predicted_type = self.classifier.predict([query])[0]
        probabilities = self.classifier.predict_proba([query])[0]
        
        # Get confidence (max probability)
        max_prob_idx = np.argmax(probabilities)
        confidence = probabilities[max_prob_idx]
        
        return predicted_type, confidence
    
    def get_query_probabilities(self, query: str) -> Dict[str, float]:
        """
        Get probabilities for all query types.
        """
        if not self.is_trained:
            self.train_classifier()
        
        probabilities = self.classifier.predict_proba([query])[0]
        classes = self.classifier.classes_
        
        return {class_name: prob for class_name, prob in zip(classes, probabilities)}

class PoliticalQueryAnalyzer:
    """
    Analyze political queries for routing features.
    """
    
    def __init__(self):
        # Political keywords by category
        self.keyword_categories = {
            'temporal': [
                'cuándo', 'año', 'fecha', 'época', 'período', 'durante', 
                'desde', 'hasta', 'antes', 'después', 'mientras', 'siglo'
            ],
            'personal': [
                'quién', 'quien', 'nombre', 'biografía', 'vida', 'nació',
                'murió', 'presidente', 'político', 'líder', 'caudillo'
            ],
            'conceptual': [
                'qué', 'que', 'cuál', 'cual', 'concepto', 'idea', 'teoría',
                'pensamiento', 'ideología', 'doctrina', 'principio'
            ],
            'causal': [
                'por qué', 'porque', 'razón', 'causa', 'motivo', 'origen',
                'consecuencia', 'resultado', 'efecto', 'impacto'
            ],
            'comparative': [
                'diferencia', 'similitud', 'comparar', 'versus', 'vs', 'contra',
                'parecido', 'distinto', 'mayor', 'menor', 'mejor', 'peor'
            ],
            'evolutionary': [
                'evolución', 'desarrollo', 'cambio', 'transformación', 'progreso',
                'crecimiento', 'decadencia', 'transición', 'proceso'
            ],
            'genealogical': [
                'influencia', 'herencia', 'tradición', 'continuidad', 'legado',
                'origen', 'raíces', 'antecedente', 'genealogía', 'línea'
            ],
            'geographical': [
                'dónde', 'donde', 'lugar', 'región', 'provincia', 'buenos aires',
                'interior', 'capital', 'federal', 'nacional', 'local'
            ]
        }
        
        self.political_entities = [
            # Politicians
            'perón', 'evita', 'moreno', 'saavedra', 'rosas', 'mitre', 
            'sarmiento', 'alberdi', 'urquiza', 'yrigoyen', 'alfonsín',
            'menem', 'kirchner', 'cristina', 'macri', 'milei',
            # Movements/Parties
            'peronismo', 'radicalismo', 'liberalismo', 'federalismo', 'unitarios',
            'justicialismo', 'ucr', 'pro', 'cambiemos', 'frente de todos',
            # Historical events
            'revolución de mayo', 'independencia', 'constitución', 'golpe',
            'proceso', 'malvinas', 'corralito', 'convertibilidad'
        ]
    
    def analyze_query_features(self, query: str) -> Dict[str, Any]:
        """
        Extract features from a political query for routing.
        """
        query_lower = query.lower()
        
        features = {
            'length': len(query.split()),
            'has_question_word': bool(re.search(r'\b(qué|quién|cuándo|dónde|cómo|por qué|cuál)\b', query_lower)),
            'entity_mentions': [],
            'keyword_scores': {},
            'specificity_score': 0.0,
            'complexity_score': 0.0
        }
        
        # Count keyword categories
        for category, keywords in self.keyword_categories.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            features['keyword_scores'][category] = score
        
        # Find political entity mentions
        for entity in self.political_entities:
            if entity in query_lower:
                features['entity_mentions'].append(entity)
        
        # Calculate specificity (more specific = better for vector RAG)
        specificity_indicators = ['cuándo', 'dónde', 'quién', 'fecha', 'año', 'nombre']
        features['specificity_score'] = sum(1 for indicator in specificity_indicators if indicator in query_lower) / len(specificity_indicators)
        
        # Calculate complexity (more complex = better for graph RAG)
        complexity_indicators = ['evolución', 'influencia', 'desarrollo', 'comparar', 'diferencia', 'relación']
        features['complexity_score'] = sum(1 for indicator in complexity_indicators if indicator in query_lower) / len(complexity_indicators)
        
        return features

class EnhancedPoliticalQueryRouter:
    """
    Advanced query router using ML classification and LLM reasoning.
    """
    
    def __init__(self, use_llm: bool = False):
        self.classifier = PoliticalQueryClassifier()
        self.analyzer = PoliticalQueryAnalyzer()
        self.use_llm = use_llm and HAS_OPENAI
        
        # Routing rules
        self.routing_rules = {
            QueryType.FACTUAL: RetrievalMethod.VECTOR_RAG,
            QueryType.THEMATIC: RetrievalMethod.GRAPH_RAG_GLOBAL,
            QueryType.GENEALOGICAL: RetrievalMethod.GRAPH_RAG_LOCAL,
            QueryType.COMPARATIVE: RetrievalMethod.HYBRID_RAG,
            QueryType.PROCEDURAL: RetrievalMethod.GRAPH_RAG_GLOBAL,
            QueryType.HYBRID: RetrievalMethod.HYBRID_RAG
        }
    
    def route_query(self, query: str) -> RoutingDecision:
        """
        Route a political query to the appropriate RAG method.
        """
        # Analyze query features
        features = self.analyzer.analyze_query_features(query)
        
        # Classify query type
        query_type_str, confidence = self.classifier.classify_query(query)
        query_type = QueryType(query_type_str)
        
        # Get primary routing method
        primary_method = self.routing_rules[query_type]
        
        # Adjust based on features
        adjusted_method, alternative_methods = self._adjust_routing_with_features(
            primary_method, features, confidence
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query_type, features, confidence, adjusted_method)
        
        # Use LLM for additional validation if available
        if self.use_llm:
            llm_validation = self._llm_validate_routing(query, adjusted_method)
            if llm_validation:
                reasoning += f" LLM validation: {llm_validation}"
        
        return RoutingDecision(
            query_type=query_type,
            retrieval_method=adjusted_method,
            confidence=confidence,
            reasoning=reasoning,
            alternative_methods=alternative_methods,
            query_features=features
        )
    
    def _adjust_routing_with_features(self, primary_method: RetrievalMethod, 
                                    features: Dict[str, Any], 
                                    confidence: float) -> Tuple[RetrievalMethod, List[RetrievalMethod]]:
        """
        Adjust routing based on query features.
        """
        alternative_methods = []
        
        # High specificity -> prefer Vector-RAG
        if features['specificity_score'] > 0.5 and primary_method != RetrievalMethod.VECTOR_RAG:
            alternative_methods.append(RetrievalMethod.VECTOR_RAG)
        
        # High complexity -> prefer Graph-RAG
        if features['complexity_score'] > 0.3:
            if primary_method == RetrievalMethod.VECTOR_RAG:
                # Upgrade to hybrid for complex queries
                primary_method = RetrievalMethod.HYBRID_RAG
            alternative_methods.append(RetrievalMethod.GRAPH_RAG_GLOBAL)
        
        # Multiple entities mentioned -> Graph-RAG
        if len(features['entity_mentions']) > 1:
            alternative_methods.append(RetrievalMethod.GRAPH_RAG_LOCAL)
        
        # Low confidence -> use hybrid approach
        if confidence < 0.7:
            primary_method = RetrievalMethod.HYBRID_RAG
        
        # Genealogical keywords -> Graph-RAG local
        if features['keyword_scores'].get('genealogical', 0) > 0:
            alternative_methods.append(RetrievalMethod.GRAPH_RAG_LOCAL)
        
        return primary_method, alternative_methods
    
    def _generate_reasoning(self, query_type: QueryType, features: Dict[str, Any], 
                          confidence: float, method: RetrievalMethod) -> str:
        """
        Generate human-readable reasoning for routing decision.
        """
        reasoning_parts = []
        
        # Query type reasoning
        reasoning_parts.append(f"Classified as {query_type.value} query (confidence: {confidence:.2%})")
        
        # Feature-based reasoning
        if features['specificity_score'] > 0.5:
            reasoning_parts.append("High specificity detected")
        
        if features['complexity_score'] > 0.3:
            reasoning_parts.append("Complex analysis required")
        
        if len(features['entity_mentions']) > 1:
            reasoning_parts.append(f"Multiple entities mentioned: {', '.join(features['entity_mentions'])}")
        
        # Method reasoning
        method_explanations = {
            RetrievalMethod.VECTOR_RAG: "Using Vector-RAG for fast, precise retrieval",
            RetrievalMethod.GRAPH_RAG_LOCAL: "Using Graph-RAG local for entity-focused analysis",
            RetrievalMethod.GRAPH_RAG_GLOBAL: "Using Graph-RAG global for thematic synthesis",
            RetrievalMethod.HYBRID_RAG: "Using hybrid approach for comprehensive analysis"
        }
        
        reasoning_parts.append(method_explanations.get(method, f"Using {method.value}"))
        
        return ". ".join(reasoning_parts) + "."
    
    def _llm_validate_routing(self, query: str, proposed_method: RetrievalMethod) -> Optional[str]:
        """
        Use LLM to validate routing decision (if available).
        """
        if not self.use_llm or not HAS_OPENAI:
            return None
        
        try:
            validation_prompt = f"""
            Query: "{query}"
            Proposed method: {proposed_method.value}
            
            Methods available:
            - vector_rag: Fast retrieval for specific facts, dates, names
            - graph_rag_local: Entity-focused analysis, relationships around specific politicians/events
            - graph_rag_global: Thematic analysis across political movements and eras
            - hybrid_rag: Combination for complex queries
            
            Is the proposed method appropriate? Answer with just "Yes" or suggest a better method with brief reasoning (max 50 words).
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in political analysis and information retrieval methods."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return None
    
    def explain_routing_options(self, query: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of all routing options for a query.
        """
        features = self.analyzer.analyze_query_features(query)
        probabilities = self.classifier.get_query_probabilities(query)
        
        # Get routing for all query types
        routing_options = {}
        for query_type in QueryType:
            method = self.routing_rules[query_type]
            routing_options[query_type.value] = {
                'method': method.value,
                'probability': probabilities.get(query_type.value, 0.0)
            }
        
        return {
            'query': query,
            'features': features,
            'type_probabilities': probabilities,
            'routing_options': routing_options,
            'recommended': self.route_query(query).retrieval_method.value
        }

# Integration functions
def create_enhanced_router(use_llm: bool = False) -> EnhancedPoliticalQueryRouter:
    """
    Create and initialize enhanced query router.
    """
    router = EnhancedPoliticalQueryRouter(use_llm=use_llm)
    
    # Pre-train classifier
    router.classifier.train_classifier()
    
    return router

if __name__ == "__main__":
    # Demo enhanced query router
    print("🎯 Enhanced Political Query Router Demo")
    
    router = create_enhanced_router(use_llm=False)  # Set to True if OpenAI available
    
    test_queries = [
        "¿Cuándo nació Juan Perón?",
        "¿Cómo evolucionó el pensamiento peronista desde 1945?",
        "¿Qué influencia tuvo Moreno en Perón?",
        "¿En qué se diferencia el federalismo de Rosas del de Urquiza?",
        "¿Cuál fue el proceso de la Revolución de Mayo?"
    ]
    
    print("\n📋 Query Routing Decisions:")
    
    for query in test_queries:
        decision = router.route_query(query)
        print(f"\n❓ \"{query}\"")
        print(f"🎯 Type: {decision.query_type.value}")
        print(f"⚙️  Method: {decision.retrieval_method.value}")
        print(f"🎯 Confidence: {decision.confidence:.2%}")
        print(f"💭 Reasoning: {decision.reasoning}")
        
        if decision.alternative_methods:
            alternatives = [m.value for m in decision.alternative_methods]
            print(f"🔄 Alternatives: {', '.join(alternatives)}")
    
    print("\n✅ Enhanced query router ready for integration!")