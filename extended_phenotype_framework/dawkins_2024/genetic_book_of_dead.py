"""
Genetic Book of the Dead - Implementation of Dawkins 2024 Concept

Implements the dual function of legal texts as:
1. ARCHIVE function: Recording past power configurations (like fossil record)
2. BETTING function: Making bets about future power configurations

Legal systems serve as both historical record of who had power
and predictive models of who will have power.

Based on Richard Dawkins' 2024 concepts applied to legal evolutionary theory
by Ignacio Adrián Lerer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from datetime import datetime, timedelta, date
import numpy as np
import uuid
import re
import json
from collections import defaultdict, Counter


class PowerSignatureType(Enum):
    """Types of power signatures detectable in legal texts"""
    ECONOMIC_DOMINANCE = "economic_dominance"
    POLITICAL_AUTHORITY = "political_authority"
    TECHNOLOGICAL_CONTROL = "technological_control"
    CULTURAL_HEGEMONY = "cultural_hegemony"
    INSTITUTIONAL_CAPTURE = "institutional_capture"
    REGULATORY_INFLUENCE = "regulatory_influence"
    JUDICIAL_PRECEDENCE = "judicial_precedence"
    MILITARY_POWER = "military_power"


class TemporalMode(Enum):
    """Temporal perspective of power analysis"""
    PAST = "past"           # Archive function - what power existed
    PRESENT = "present"     # Current power configuration
    FUTURE = "future"       # Betting function - projected power
    CROSS_TEMPORAL = "cross_temporal"  # Patterns across time


class PowerTransition(Enum):
    """Types of power transitions"""
    GRADUAL_SHIFT = "gradual_shift"         # Slow power evolution
    REVOLUTIONARY_CHANGE = "revolutionary_change"  # Rapid power overthrow
    INSTITUTIONAL_REFORM = "institutional_reform"  # Managed transition
    CRISIS_DRIVEN = "crisis_driven"         # Crisis-induced change
    TECHNOLOGICAL_DISRUPTION = "technological_disruption"  # Tech-driven change
    DEMOGRAPHIC_SHIFT = "demographic_shift" # Population-driven change


@dataclass
class PowerSignature:
    """Signature of power configuration in legal text"""
    signature_type: PowerSignatureType
    power_entities: List[str]           # Who holds this type of power
    power_magnitude: float              # Strength of power (0-1)
    temporal_scope: Tuple[datetime, Optional[datetime]]  # When this power applies
    
    # Textual evidence
    textual_markers: List[str]          # Phrases indicating this power
    institutional_references: List[str]  # Institutions mentioned
    procedural_advantages: List[str]     # Procedural benefits granted
    
    # Analytical features
    exclusivity_level: float            # How exclusive this power is (0-1)
    enforcement_mechanisms: List[str]    # How this power is enforced
    legitimacy_sources: List[str]       # Sources of legitimacy claimed
    
    # Predictive elements
    stability_indicators: List[str]     # Indicators of power stability
    vulnerability_markers: List[str]    # Indicators of vulnerability
    succession_mechanisms: List[str]    # How power transfers


@dataclass
class DualFunctionAnalysis:
    """Analysis of legal text's dual archive+betting function"""
    # Archive function analysis
    archive_function: 'PowerArchaeology'
    historical_power_signatures: List[PowerSignature]
    power_continuities: List[str]       # What power patterns persist
    
    # Betting function analysis  
    betting_function: 'FuturePowerProjection'
    future_power_projections: List[PowerSignature]
    disruption_points: List[str]        # Where text anticipates disruption
    
    # Cross-temporal analysis
    temporal_consistency: float         # Consistency between archive and bet (0-1)
    prediction_confidence: float        # Confidence in future projections (0-1)
    
    # Power evolution patterns
    power_evolution_trajectory: List[Tuple[datetime, Dict[str, float]]]
    transition_mechanisms: List[PowerTransition]
    adaptive_capacity: float           # System's capacity to adapt (0-1)
    
    # Validation metrics
    historical_accuracy: Optional[float]  # How accurate were past predictions
    predictive_coherence: float        # Internal coherence of predictions
    
    # Archaeological insights
    buried_power_traces: List[str]     # Hidden/suppressed power traces
    dominant_narratives: List[str]     # Openly celebrated power stories
    counter_narratives: List[str]      # Alternative power interpretations


@dataclass
class PowerArchaeology:
    """Archaeological analysis of power through legal texts"""
    temporal_layers: List['ArchaeologicalLayer']
    epoch_constructors: Dict[str, List[str]]  # Dominant constructors per epoch
    power_transitions: List['PowerTransitionEvent']
    construction_patterns: List[str]    # Recurring patterns of power construction
    
    # Excavation results
    excavation_depth: int              # How far back analysis goes
    layer_visibility: Dict[str, float] # Visibility of each layer
    stratigraphic_sequence: List[str]  # Chronological sequence of layers
    
    # Power continuities and breaks
    persistent_power_structures: List[str]  # Structures that survive transitions
    extinct_power_forms: List[str]     # Power forms that disappeared
    hybrid_formations: List[str]       # New combinations of old elements
    
    # Analytical insights
    construction_efficiency: Dict[str, float]  # Efficiency of different constructors
    power_concentration_trends: List[Tuple[datetime, float]]  # Concentration over time
    innovation_cycles: List[Tuple[datetime, str]]  # Periods of legal innovation


@dataclass
class FuturePowerProjection:
    """Projection of future power configurations"""
    projection_timeline: List[Tuple[int, PowerSignature]]  # (years_from_now, power_config)
    confidence_intervals: Dict[str, Tuple[float, float]]   # Uncertainty ranges
    
    # Scenario analysis
    base_scenario: Dict[str, Any]      # Most likely future scenario
    optimistic_scenario: Dict[str, Any]  # Best case for current power holders
    pessimistic_scenario: Dict[str, Any]  # Worst case for current power holders
    disruptive_scenarios: List[Dict[str, Any]]  # Potential disruptions
    
    # Betting analysis
    power_bets_placed: List[str]       # Who the text "bets" will have power
    hedge_mechanisms: List[str]        # How text hedges against uncertainty
    exit_strategies: List[str]         # Escape routes built into system
    
    # Validation framework
    prediction_testability: float     # How testable are the predictions (0-1)
    adaptation_triggers: List[str]    # What would trigger system adaptation
    failure_modes: List[str]          # How predictions could fail


@dataclass
class ArchaeologicalLayer:
    """Single layer in power archaeology"""
    layer_id: str
    epoch_start: datetime
    epoch_end: Optional[datetime]
    dominant_power_signature: PowerSignature
    
    # Layer characteristics
    layer_thickness: float            # How much legal "sediment" from this epoch
    preservation_quality: float       # How well preserved (0-1)
    contamination_level: float        # Mixing with other layers (0-1)
    
    # Power configuration
    power_distribution: Dict[str, float]  # Power by entity type
    institutional_architecture: List[str]  # Key institutions
    legitimacy_narrative: str         # How power justified itself
    
    # Evolution markers
    innovation_markers: List[str]     # Legal innovations from this period
    persistence_mechanisms: List[str] # How this layer tries to persist
    vulnerability_points: List[str]   # Where this layer is vulnerable


class GeneticBookAnalyzer:
    """
    Main analyzer implementing Genetic Book of the Dead concept
    
    Analyzes legal texts for their dual function as:
    - Archive of past power configurations (fossil record function)
    - Betting mechanism on future power configurations (prediction function)
    """
    
    def __init__(self):
        self.power_signatures_cache: Dict[str, List[PowerSignature]] = {}
        self.archaeological_data: Dict[str, PowerArchaeology] = {}
        self.prediction_accuracy_history: List[Tuple[datetime, float]] = []
        
        # Analysis parameters
        self.archaeological_depth = 50    # Default years to excavate
        self.prediction_horizon = 20      # Default years to project forward
        self.confidence_threshold = 0.7   # Minimum confidence for predictions
        
        # Power signature detection patterns
        self.power_markers = {
            PowerSignatureType.ECONOMIC_DOMINANCE: [
                'market forces', 'economic efficiency', 'private property',
                'free trade', 'competition', 'investor protection'
            ],
            PowerSignatureType.POLITICAL_AUTHORITY: [
                'state authority', 'public interest', 'democratic process',
                'legislative power', 'executive authority', 'sovereignty'
            ],
            PowerSignatureType.TECHNOLOGICAL_CONTROL: [
                'innovation', 'digital transformation', 'technological progress',
                'artificial intelligence', 'data governance', 'platform regulation'
            ],
            PowerSignatureType.INSTITUTIONAL_CAPTURE: [
                'regulatory expertise', 'industry knowledge', 'technical standards',
                'best practices', 'stakeholder input', 'consultation process'
            ]
        }
    
    def perform_dual_function_analysis(
        self, 
        legal_corpus: 'LegalCorpus',
        temporal_scope: 'TemporalScope'
    ) -> DualFunctionAnalysis:
        """
        Perform comprehensive dual function analysis of legal corpus
        
        Args:
            legal_corpus: Collection of legal texts to analyze
            temporal_scope: Time period for analysis
            
        Returns:
            Complete dual function analysis
        """
        # ARCHIVE FUNCTION: Reconstruct historical power from legal texts
        historical_power_reconstruction = self._reconstruct_historical_power(
            legal_corpus,
            temporal_scope.start_date,
            temporal_scope.mid_point
        )
        
        # BETTING FUNCTION: Extract future power projections
        future_power_projections = self._extract_future_projections(
            legal_corpus,
            temporal_scope.mid_point,
            temporal_scope.end_date
        )
        
        # VALIDATION: Check accuracy of historical predictions
        predictive_accuracy = self._validate_historical_predictions(
            historical_power_reconstruction,
            temporal_scope.mid_point
        )
        
        # COHERENCE ANALYSIS: Analyze consistency between archive and betting
        coherence_analysis = self._analyze_archive_betting_coherence(
            historical_power_reconstruction,
            future_power_projections
        )
        
        # POWER EVOLUTION: Trace power evolution patterns
        power_evolution = self._trace_power_evolution(
            historical_power_reconstruction,
            future_power_projections
        )
        
        # ARCHAEOLOGICAL INSIGHTS: Deep structural analysis
        archaeological_insights = self._generate_archaeological_insights(
            historical_power_reconstruction
        )
        
        return DualFunctionAnalysis(
            archive_function=historical_power_reconstruction,
            historical_power_signatures=self._extract_historical_signatures(legal_corpus),
            power_continuities=self._identify_power_continuities(
                historical_power_reconstruction, 
                future_power_projections
            ),
            betting_function=future_power_projections,
            future_power_projections=self._extract_future_signatures(legal_corpus),
            disruption_points=self._identify_disruption_points(future_power_projections),
            temporal_consistency=coherence_analysis['consistency_score'],
            prediction_confidence=future_power_projections.confidence_intervals.get('overall', (0.5, 0.5))[0],
            power_evolution_trajectory=power_evolution['trajectory'],
            transition_mechanisms=power_evolution['mechanisms'],
            adaptive_capacity=self._calculate_system_adaptive_capacity(legal_corpus),
            historical_accuracy=predictive_accuracy,
            predictive_coherence=coherence_analysis['internal_coherence'],
            buried_power_traces=archaeological_insights['buried_traces'],
            dominant_narratives=archaeological_insights['dominant_narratives'],
            counter_narratives=archaeological_insights['counter_narratives']
        )
    
    def analyze_legal_text_dual_function(self, legal_text: str) -> DualFunctionAnalysis:
        """
        Analyze single legal text for dual function
        
        Args:
            legal_text: Legal text to analyze
            
        Returns:
            Dual function analysis of single text
        """
        # Extract power signatures for past and future
        historical_signatures = self._extract_power_signature(legal_text, TemporalMode.PAST)
        future_signatures = self._extract_power_signature(legal_text, TemporalMode.FUTURE)
        
        # Calculate temporal consistency
        consistency_score = self._calculate_temporal_consistency(
            historical_signatures, 
            future_signatures
        )
        
        # Build simplified analysis for single text
        return self._build_single_text_analysis(
            legal_text,
            historical_signatures,
            future_signatures,
            consistency_score
        )
    
    def generate_power_archaeology_report(
        self, 
        legal_system: 'LegalSystem',
        archaeological_depth: int = None
    ) -> PowerArchaeology:
        """
        Generate comprehensive power archaeology report
        
        Args:
            legal_system: Legal system to analyze
            archaeological_depth: How many years to excavate (default: self.archaeological_depth)
            
        Returns:
            Complete power archaeology analysis
        """
        depth = archaeological_depth or self.archaeological_depth
        
        # Excavate temporal layers
        temporal_layers = self._excavate_temporal_layers(legal_system, depth)
        
        # Identify dominant constructors by epoch
        epoch_constructors = {}
        for layer in temporal_layers:
            constructors = self._identify_epoch_constructors(layer)
            epoch_constructors[layer.layer_id] = constructors
        
        # Analyze power transitions between epochs
        power_transitions = self._analyze_power_transitions(temporal_layers)
        
        # Identify construction patterns
        construction_patterns = self._identify_construction_patterns(temporal_layers)
        
        # Calculate construction efficiency
        construction_efficiency = self._calculate_construction_efficiency(
            epoch_constructors,
            temporal_layers
        )
        
        # Analyze power concentration trends
        concentration_trends = self._analyze_power_concentration_trends(temporal_layers)
        
        # Identify innovation cycles
        innovation_cycles = self._identify_innovation_cycles(temporal_layers)
        
        return PowerArchaeology(
            temporal_layers=temporal_layers,
            epoch_constructors=epoch_constructors,
            power_transitions=power_transitions,
            construction_patterns=construction_patterns,
            excavation_depth=depth,
            layer_visibility=self._calculate_layer_visibility(temporal_layers),
            stratigraphic_sequence=[layer.layer_id for layer in temporal_layers],
            persistent_power_structures=self._identify_persistent_structures(temporal_layers),
            extinct_power_forms=self._identify_extinct_forms(temporal_layers),
            hybrid_formations=self._identify_hybrid_formations(temporal_layers),
            construction_efficiency=construction_efficiency,
            power_concentration_trends=concentration_trends,
            innovation_cycles=innovation_cycles
        )
    
    def validate_prediction_accuracy(
        self,
        historical_corpus: 'LegalCorpus',
        validation_date: datetime,
        prediction_horizon: int = 10
    ) -> float:
        """
        Validate accuracy of past predictions made by legal texts
        
        Args:
            historical_corpus: Historical legal texts with predictions
            validation_date: Date to validate predictions against
            prediction_horizon: Years forward the predictions covered
            
        Returns:
            Accuracy score (0-1)
        """
        # Extract predictions made before validation_date
        historical_predictions = self._extract_historical_predictions(
            historical_corpus,
            validation_date - timedelta(days=365 * prediction_horizon),
            validation_date
        )
        
        # Get actual power configuration at validation_date
        actual_power_config = self._get_actual_power_distribution(validation_date)
        
        # Calculate prediction accuracy
        accuracy_scores = []
        for prediction in historical_predictions:
            accuracy = self._calculate_prediction_accuracy(
                prediction,
                actual_power_config
            )
            accuracy_scores.append(accuracy)
        
        overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Store for learning
        self.prediction_accuracy_history.append((validation_date, overall_accuracy))
        
        return overall_accuracy
    
    # Private implementation methods
    
    def _extract_power_signature(
        self, 
        legal_text: str, 
        temporal_mode: TemporalMode
    ) -> List[PowerSignature]:
        """Extract power signatures from legal text for specific temporal mode"""
        signatures = []
        
        # Temporal markers for different modes
        temporal_markers = {
            TemporalMode.PAST: ['was', 'had been', 'historically', 'traditionally', 'previously'],
            TemporalMode.PRESENT: ['is', 'are', 'currently', 'now', 'presently'],
            TemporalMode.FUTURE: ['will', 'shall', 'expected to', 'projected', 'anticipated']
        }
        
        # Look for each type of power signature
        for power_type, markers in self.power_markers.items():
            # Check if this power type is mentioned in the temporal context
            power_mentions = []
            for marker in markers:
                if marker.lower() in legal_text.lower():
                    # Look for temporal context around the marker
                    context = self._extract_context_around_marker(legal_text, marker, 50)
                    if any(temp_marker in context.lower() for temp_marker in temporal_markers[temporal_mode]):
                        power_mentions.append(marker)
            
            if power_mentions:
                # Create power signature for this type
                signature = self._build_power_signature(
                    power_type,
                    power_mentions,
                    legal_text,
                    temporal_mode
                )
                signatures.append(signature)
        
        return signatures
    
    def _build_power_signature(
        self,
        power_type: PowerSignatureType,
        power_mentions: List[str],
        legal_text: str,
        temporal_mode: TemporalMode
    ) -> PowerSignature:
        """Build complete power signature from detected elements"""
        
        # Extract entities associated with this power type
        power_entities = self._extract_power_entities(legal_text, power_mentions)
        
        # Calculate power magnitude based on text emphasis
        power_magnitude = self._calculate_power_magnitude(legal_text, power_mentions)
        
        # Extract temporal scope
        temporal_scope = self._extract_temporal_scope(legal_text, temporal_mode)
        
        # Extract additional features
        institutional_refs = self._extract_institutional_references(legal_text, power_mentions)
        procedural_advantages = self._extract_procedural_advantages(legal_text, power_mentions)
        enforcement_mechanisms = self._extract_enforcement_mechanisms(legal_text, power_mentions)
        
        return PowerSignature(
            signature_type=power_type,
            power_entities=power_entities,
            power_magnitude=power_magnitude,
            temporal_scope=temporal_scope,
            textual_markers=power_mentions,
            institutional_references=institutional_refs,
            procedural_advantages=procedural_advantages,
            exclusivity_level=self._calculate_exclusivity_level(legal_text, power_mentions),
            enforcement_mechanisms=enforcement_mechanisms,
            legitimacy_sources=self._extract_legitimacy_sources(legal_text, power_mentions),
            stability_indicators=self._extract_stability_indicators(legal_text, power_mentions),
            vulnerability_markers=self._extract_vulnerability_markers(legal_text, power_mentions),
            succession_mechanisms=self._extract_succession_mechanisms(legal_text, power_mentions)
        )
    
    def _reconstruct_historical_power(
        self,
        legal_corpus: 'LegalCorpus',
        start_date: datetime,
        end_date: datetime
    ) -> PowerArchaeology:
        """Reconstruct historical power configurations from legal corpus"""
        
        # Filter corpus by date range
        historical_texts = legal_corpus.filter_by_date_range(start_date, end_date)
        
        # Excavate temporal layers
        layers = []
        current_date = start_date
        layer_duration = timedelta(days=365 * 5)  # 5-year layers
        
        while current_date < end_date:
            layer_end = min(current_date + layer_duration, end_date)
            layer_texts = historical_texts.filter_by_date_range(current_date, layer_end)
            
            if layer_texts.texts:  # Only create layer if there are texts
                layer = self._create_archaeological_layer(
                    f"layer_{current_date.year}_{layer_end.year}",
                    current_date,
                    layer_end,
                    layer_texts
                )
                layers.append(layer)
            
            current_date = layer_end
        
        # Analyze transitions between layers
        transitions = self._analyze_layer_transitions(layers)
        
        # Identify construction patterns
        patterns = self._identify_construction_patterns(layers)
        
        return PowerArchaeology(
            temporal_layers=layers,
            epoch_constructors=self._map_epoch_constructors(layers),
            power_transitions=transitions,
            construction_patterns=patterns,
            excavation_depth=(end_date - start_date).days // 365,
            layer_visibility=self._calculate_layer_visibility(layers),
            stratigraphic_sequence=[layer.layer_id for layer in layers],
            persistent_power_structures=self._identify_persistent_structures(layers),
            extinct_power_forms=self._identify_extinct_forms(layers),
            hybrid_formations=self._identify_hybrid_formations(layers),
            construction_efficiency=self._calculate_construction_efficiency({}, layers),
            power_concentration_trends=self._analyze_power_concentration_trends(layers),
            innovation_cycles=self._identify_innovation_cycles(layers)
        )
    
    def _extract_future_projections(
        self,
        legal_corpus: 'LegalCorpus',
        start_date: datetime,
        projection_horizon: datetime
    ) -> FuturePowerProjection:
        """Extract future power projections from legal corpus"""
        
        # Filter texts that make future projections
        projection_texts = legal_corpus.filter_texts_with_future_references()
        
        # Extract projection timeline
        timeline = self._build_projection_timeline(projection_texts, projection_horizon)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_projection_confidence(projection_texts)
        
        # Build scenarios
        scenarios = self._build_future_scenarios(projection_texts)
        
        # Analyze betting patterns
        bets_placed = self._analyze_power_bets(projection_texts)
        hedge_mechanisms = self._identify_hedge_mechanisms(projection_texts)
        
        return FuturePowerProjection(
            projection_timeline=timeline,
            confidence_intervals=confidence_intervals,
            base_scenario=scenarios['base'],
            optimistic_scenario=scenarios['optimistic'],
            pessimistic_scenario=scenarios['pessimistic'],
            disruptive_scenarios=scenarios['disruptive'],
            power_bets_placed=bets_placed,
            hedge_mechanisms=hedge_mechanisms,
            exit_strategies=self._identify_exit_strategies(projection_texts),
            prediction_testability=self._calculate_prediction_testability(projection_texts),
            adaptation_triggers=self._identify_adaptation_triggers(projection_texts),
            failure_modes=self._identify_failure_modes(projection_texts)
        )
    
    # Utility methods for text analysis
    
    def _extract_context_around_marker(self, text: str, marker: str, window_size: int = 50) -> str:
        """Extract context around a text marker"""
        marker_pos = text.lower().find(marker.lower())
        if marker_pos == -1:
            return ""
        
        start = max(0, marker_pos - window_size)
        end = min(len(text), marker_pos + len(marker) + window_size)
        
        return text[start:end]
    
    def _extract_power_entities(self, text: str, power_mentions: List[str]) -> List[str]:
        """Extract entities that hold power based on text analysis"""
        # Simplified entity extraction - would implement NLP
        entities = []
        
        # Common power entity patterns
        entity_patterns = [
            r'\b(government|state|administration|authority)\b',
            r'\b(corporation|company|business|industry)\b',
            r'\b(court|judiciary|tribunal)\b',
            r'\b(legislature|parliament|congress)\b',
            r'\b(agency|commission|department)\b'
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates
    
    def _calculate_power_magnitude(self, text: str, power_mentions: List[str]) -> float:
        """Calculate magnitude of power based on text emphasis"""
        # Count power mentions and intensity words
        intensity_words = ['absolute', 'complete', 'total', 'full', 'exclusive', 'sole']
        qualifying_words = ['limited', 'restricted', 'conditional', 'partial', 'shared']
        
        intensity_count = sum(1 for word in intensity_words if word in text.lower())
        qualifying_count = sum(1 for word in qualifying_words if word in text.lower())
        mention_count = len(power_mentions)
        
        # Calculate magnitude (0-1)
        base_magnitude = min(1.0, mention_count / 5.0)  # Normalize by max expected mentions
        intensity_factor = 1.0 + (intensity_count * 0.2) - (qualifying_count * 0.1)
        
        return max(0.0, min(1.0, base_magnitude * intensity_factor))
    
    def _extract_temporal_scope(self, text: str, temporal_mode: TemporalMode) -> Tuple[datetime, Optional[datetime]]:
        """Extract temporal scope from text"""
        # Simplified temporal extraction - would implement sophisticated date parsing
        if temporal_mode == TemporalMode.PAST:
            return (datetime(1900, 1, 1), datetime.now())
        elif temporal_mode == TemporalMode.FUTURE:
            return (datetime.now(), None)
        else:
            return (datetime.now(), datetime.now())
    
    def __repr__(self) -> str:
        return f"GeneticBookAnalyzer(cache_size={len(self.power_signatures_cache)})"