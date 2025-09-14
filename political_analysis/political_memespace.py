"""
Political Memespace: Mapping Competitive Dynamics in Political Space
Adapts Legal-Memespace for political antagonisms using Lotka-Volterra equations
Based on: legal_memespace.memespace.LegalMemespace
Author: Ignacio Adrián Lerer
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.signal import find_peaks
import json

# Import base LegalMemespace
import sys
sys.path.append('..')
from legal_memespace.memespace import LegalMemespace, PhaseTransition

logger = logging.getLogger(__name__)

@dataclass
class PoliticalPhaseTransition:
    """Represents a detected phase transition in political space."""
    date: str
    coordinates_before: List[float]
    coordinates_after: List[float]
    magnitude: float
    transition_type: str
    affected_dimensions: List[int]
    statistical_significance: float
    political_event: str = ""
    actors_involved: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'date': self.date,
            'coordinates_before': self.coordinates_before,
            'coordinates_after': self.coordinates_after,
            'magnitude': self.magnitude,
            'transition_type': self.transition_type,
            'affected_dimensions': self.affected_dimensions,
            'statistical_significance': self.statistical_significance,
            'political_event': self.political_event,
            'actors_involved': self.actors_involved or []
        }

class PoliticalMemespace(LegalMemespace):
    """
    Adapts Legal-Memespace to map political antagonisms in 4-dimensional space.
    
    Political dimensions:
    - D1: Centralization vs Federalism (0=Central, 1=Federal)
    - D2: Buenos Aires vs Interior (0=Port, 1=Provinces) 
    - D3: Elite vs Popular (0=Elite, 1=Popular)
    - D4: Revolution vs Evolution (0=Gradual, 1=Rupture)
    """
    
    def __init__(self, n_dimensions: int = 4):
        """
        Initialize Political Memespace.
        
        Parameters:
        -----------
        n_dimensions : int
            Number of dimensions (default 4 for Argentine political space)
        """
        super().__init__(n_dimensions=n_dimensions)
        
        # Define political dimensions
        self.dimension_names = {
            0: 'Centralization vs Federalism',
            1: 'Buenos Aires vs Interior', 
            2: 'Elite vs Popular',
            3: 'Revolution vs Evolution'
        }
        
        self.dimension_labels = {
            0: ['Central', 'Federal'],
            1: ['Port', 'Interior'],
            2: ['Elite', 'Popular'], 
            3: ['Gradual', 'Rupture']
        }
        
        # Political-specific parameters
        self.political_keywords = {
            'centralization': ['estado', 'nación', 'central', 'unidad', 'gobierno nacional'],
            'federalism': ['provincia', 'federal', 'autonomía', 'descentralización'],
            'buenos_aires': ['puerto', 'capital', 'buenos aires', 'cosmopolita', 'europeo'],
            'interior': ['interior', 'provincial', 'regional', 'local', 'gaucho'],
            'elite': ['oligarquía', 'aristocracia', 'clase dirigente', 'ilustrados', 'minoría'],
            'popular': ['pueblo', 'masa', 'trabajadores', 'popular', 'mayoría'],
            'gradual': ['evolución', 'gradual', 'reforma', 'progresivo', 'constitucional'],
            'rupture': ['revolución', 'cambio radical', 'ruptura', 'transformación', 'quiebre']
        }
        
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words=None,  # Don't use stop words for small corpora
            ngram_range=(1, 2),
            min_df=1,  # Allow single occurrence terms
            max_df=0.95
        )
        
        # Track political trajectories
        self.political_trajectories = {}
        self.antagonism_history = []
        
    def calculate_political_coordinates(self, text: str) -> List[float]:
        """
        Calculate political coordinates for a text in 4D space.
        
        Parameters:
        -----------
        text : str
            Political text to analyze
            
        Returns:
        --------
        List[float]
            4D coordinates [centralization, buenos_aires, elite, revolution]
        """
        if not text or pd.isna(text):
            return [0.5, 0.5, 0.5, 0.5]  # Neutral position
        
        text_lower = text.lower()
        coordinates = []
        
        # D1: Centralization (0) vs Federalism (1)
        central_score = sum(1 for word in self.political_keywords['centralization'] if word in text_lower)
        federal_score = sum(1 for word in self.political_keywords['federalism'] if word in text_lower)
        d1 = federal_score / (central_score + federal_score + 1)  # +1 to avoid division by zero
        coordinates.append(d1)
        
        # D2: Buenos Aires (0) vs Interior (1)  
        ba_score = sum(1 for word in self.political_keywords['buenos_aires'] if word in text_lower)
        interior_score = sum(1 for word in self.political_keywords['interior'] if word in text_lower)
        d2 = interior_score / (ba_score + interior_score + 1)
        coordinates.append(d2)
        
        # D3: Elite (0) vs Popular (1)
        elite_score = sum(1 for word in self.political_keywords['elite'] if word in text_lower)
        popular_score = sum(1 for word in self.political_keywords['popular'] if word in text_lower)
        d3 = popular_score / (elite_score + popular_score + 1)
        coordinates.append(d3)
        
        # D4: Gradual (0) vs Revolution (1)
        gradual_score = sum(1 for word in self.political_keywords['gradual'] if word in text_lower)
        revolution_score = sum(1 for word in self.political_keywords['rupture'] if word in text_lower)
        d4 = revolution_score / (gradual_score + revolution_score + 1)
        coordinates.append(d4)
        
        return coordinates
    
    def map_political_positions(self, political_documents: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Map political actors to 4D coordinates based on their texts.
        
        Parameters:
        -----------
        political_documents : pd.DataFrame
            DataFrame with columns: document_id, author, year, text
            
        Returns:
        --------
        Dict[str, List[float]]
            Mapping from actor/document to 4D coordinates
        """
        logger.info(f"Mapping {len(political_documents)} political positions to 4D space")
        
        positions = {}
        
        for _, doc in political_documents.iterrows():
            coordinates = self.calculate_political_coordinates(doc.get('text', ''))
            
            # Store by document ID and author
            positions[doc['document_id']] = coordinates
            
            # Also aggregate by author if multiple documents
            author = doc.get('author', 'Unknown')
            if author in positions:
                # Average coordinates for multiple documents by same author
                existing = positions[author]
                positions[author] = [(e + c) / 2 for e, c in zip(existing, coordinates)]
            else:
                positions[author] = coordinates.copy()
        
        return positions
    
    def calculate_antagonism_distance(self, actor1_coords: List[float], 
                                    actor2_coords: List[float]) -> float:
        """
        Calculate antagonism distance between two political actors.
        
        Higher distance = deeper antagonism
        """
        if len(actor1_coords) != len(actor2_coords):
            raise ValueError("Coordinate vectors must have same length")
        
        # Euclidean distance in 4D space
        distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(actor1_coords, actor2_coords)))
        return distance
    
    def analyze_grieta_evolution(self, political_documents: pd.DataFrame, 
                               window_years: int = 10) -> List[Tuple[int, float]]:
        """
        Analyze the evolution of political polarization (grieta) over time.
        
        Parameters:
        -----------
        political_documents : pd.DataFrame
            Political documents with year column
        window_years : int
            Time window for calculating polarization
            
        Returns:
        --------
        List[Tuple[int, float]]
            Timeline of (year, max_antagonism_distance)
        """
        logger.info("Analyzing grieta evolution over time")
        
        grieta_evolution = []
        min_year = political_documents['year'].min()
        max_year = political_documents['year'].max()
        
        for year in range(min_year, max_year + 1, window_years):
            # Get documents in time window
            window_docs = political_documents[
                (political_documents['year'] >= year) & 
                (political_documents['year'] < year + window_years)
            ]
            
            if len(window_docs) < 2:
                continue
            
            # Calculate positions for this period
            positions = self.map_political_positions(window_docs)
            
            # Find maximum antagonism distance
            max_distance = 0
            coordinates_list = list(positions.values())
            
            for i in range(len(coordinates_list)):
                for j in range(i + 1, len(coordinates_list)):
                    distance = self.calculate_antagonism_distance(
                        coordinates_list[i], coordinates_list[j]
                    )
                    max_distance = max(max_distance, distance)
            
            grieta_evolution.append((year, max_distance))
        
        return grieta_evolution
    
    def find_political_attractors(self, positions: Dict[str, List[float]], 
                                n_attractors: int = 3) -> Dict[str, List[float]]:
        """
        Find stable political positions (attractors) in 4D space.
        
        These represent recurring political archetypes in Argentine history.
        """
        logger.info(f"Finding {n_attractors} political attractors")
        
        coordinates_array = np.array(list(positions.values()))
        
        # Use K-means clustering to find attractors
        kmeans = KMeans(n_clusters=n_attractors, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coordinates_array)
        centroids = kmeans.cluster_centers_
        
        # Name attractors based on their coordinates
        attractor_names = []
        for i, centroid in enumerate(centroids):
            name_parts = []
            for dim, coord in enumerate(centroid):
                if coord < 0.33:
                    name_parts.append(self.dimension_labels[dim][0])
                elif coord > 0.67:
                    name_parts.append(self.dimension_labels[dim][1])
                else:
                    name_parts.append('Moderate')
            
            attractor_name = f"Attractor_{i+1}_{'_'.join(name_parts[:2])}"
            attractor_names.append(attractor_name)
        
        attractors = {}
        for name, centroid in zip(attractor_names, centroids):
            attractors[name] = centroid.tolist()
        
        return attractors
    
    def detect_political_phase_transitions(self, political_documents: pd.DataFrame,
                                         window_size: int = 5) -> List[PoliticalPhaseTransition]:
        """
        Detect major phase transitions in political space.
        
        Identifies moments when political landscape fundamentally shifts.
        """
        logger.info("Detecting political phase transitions")
        
        # Sort documents by year
        sorted_docs = political_documents.sort_values('year')
        
        # Calculate rolling political center of mass
        transitions = []
        years = sorted_docs['year'].unique()
        
        for i in range(window_size, len(years) - window_size):
            year = years[i]
            
            # Get documents before and after this year
            before_docs = sorted_docs[
                (sorted_docs['year'] >= year - window_size) & 
                (sorted_docs['year'] < year)
            ]
            after_docs = sorted_docs[
                (sorted_docs['year'] >= year) & 
                (sorted_docs['year'] < year + window_size)
            ]
            
            if len(before_docs) < 2 or len(after_docs) < 2:
                continue
            
            # Calculate mean positions
            before_positions = self.map_political_positions(before_docs)
            after_positions = self.map_political_positions(after_docs)
            
            before_mean = np.mean(list(before_positions.values()), axis=0)
            after_mean = np.mean(list(after_positions.values()), axis=0)
            
            # Calculate magnitude of shift
            magnitude = np.linalg.norm(after_mean - before_mean)
            
            # Significant transition threshold
            if magnitude > 0.3:  # Configurable threshold
                # Identify affected dimensions
                affected_dims = []
                for dim in range(len(before_mean)):
                    if abs(after_mean[dim] - before_mean[dim]) > 0.2:
                        affected_dims.append(dim)
                
                # Create transition object
                transition = PoliticalPhaseTransition(
                    date=str(year),
                    coordinates_before=before_mean.tolist(),
                    coordinates_after=after_mean.tolist(),
                    magnitude=magnitude,
                    transition_type=self._classify_transition_type(before_mean, after_mean),
                    affected_dimensions=affected_dims,
                    statistical_significance=0.95,  # Simplified - would use proper stats
                    political_event=self._identify_historical_event(year)
                )
                
                transitions.append(transition)
        
        return transitions
    
    def calculate_political_fitness(self, positions: Dict[str, List[float]], 
                                  outcomes: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate political fitness based on position and historical outcomes.
        
        Actors closer to successful positions have higher fitness.
        """
        fitness_scores = {}
        
        # Identify successful vs unsuccessful actors
        successful_positions = []
        unsuccessful_positions = []
        
        for actor, outcome in outcomes.items():
            if actor in positions:
                if outcome in ['success', 'victory', 'power']:
                    successful_positions.append(positions[actor])
                elif outcome in ['failure', 'defeat', 'exile']:
                    unsuccessful_positions.append(positions[actor])
        
        if not successful_positions:
            return {actor: 0.5 for actor in positions}
        
        # Calculate mean successful position
        mean_successful = np.mean(successful_positions, axis=0)
        
        # Calculate fitness as inverse distance from successful center
        for actor, coords in positions.items():
            distance = np.linalg.norm(np.array(coords) - mean_successful)
            fitness = 1 / (1 + distance)  # Sigmoid-like function
            fitness_scores[actor] = fitness
        
        return fitness_scores
    
    def simulate_political_competition(self, positions: Dict[str, List[float]],
                                     interaction_matrix: np.ndarray,
                                     time_steps: int = 100) -> Dict[str, List[float]]:
        """
        Simulate political competition using Lotka-Volterra equations.
        
        Shows how political forces compete and evolve over time.
        """
        logger.info(f"Simulating political competition over {time_steps} steps")
        
        actors = list(positions.keys())
        n_actors = len(actors)
        
        if interaction_matrix.shape != (n_actors, n_actors):
            raise ValueError("Interaction matrix must match number of actors")
        
        # Initial populations (influence levels)
        initial_populations = np.ones(n_actors)
        
        def competition_dynamics(populations, t):
            """Lotka-Volterra competition equations."""
            dpdt = np.zeros(n_actors)
            
            for i in range(n_actors):
                growth_rate = 0.1  # Base growth rate
                competition_term = sum(interaction_matrix[i, j] * populations[j] 
                                     for j in range(n_actors))
                
                dpdt[i] = growth_rate * populations[i] * (1 - competition_term)
            
            return dpdt
        
        # Simulate competition
        t = np.linspace(0, time_steps, time_steps)
        solution = odeint(competition_dynamics, initial_populations, t)
        
        # Return final populations
        final_populations = solution[-1]
        results = {}
        
        for i, actor in enumerate(actors):
            results[actor] = final_populations[i]
        
        return results
    
    def _classify_transition_type(self, before_coords: np.ndarray, 
                                after_coords: np.ndarray) -> str:
        """Classify the type of political transition."""
        diff = after_coords - before_coords
        
        # Find dimension with largest change
        max_change_dim = np.argmax(np.abs(diff))
        max_change = diff[max_change_dim]
        
        dim_name = self.dimension_names[max_change_dim]
        direction = "toward " + (self.dimension_labels[max_change_dim][1] 
                               if max_change > 0 
                               else self.dimension_labels[max_change_dim][0])
        
        return f"{dim_name} shift {direction}"
    
    def _identify_historical_event(self, year: int) -> str:
        """Identify major historical events near given year."""
        events = {
            1810: "May Revolution",
            1816: "Independence Declaration", 
            1829: "Rosas Rise",
            1852: "Caseros Battle",
            1862: "National Organization",
            1880: "Federalization Buenos Aires",
            1890: "Revolution of the Park",
            1912: "Sáenz Peña Law",
            1930: "Uriburu Coup",
            1943: "GOU Coup", 
            1946: "Perón Election",
            1955: "Revolución Libertadora",
            1973: "Perón Return",
            1976: "Process Coup",
            1983: "Democratic Transition",
            1989: "Hyperinflation Crisis",
            2001: "Economic Crisis",
            2003: "Kirchner Election",
            2015: "Macri Election",
            2019: "Fernández Election",
            2023: "Milei Election"
        }
        
        # Find closest event
        closest_year = min(events.keys(), key=lambda x: abs(x - year))
        if abs(closest_year - year) <= 3:  # Within 3 years
            return events[closest_year]
        
        return f"Unknown event around {year}"
    
    def export_political_analysis(self, positions: Dict[str, List[float]],
                                attractors: Dict[str, List[float]],
                                transitions: List[PoliticalPhaseTransition],
                                filename: str) -> None:
        """Export complete political analysis to JSON."""
        export_data = {
            'dimensions': self.dimension_names,
            'positions': positions,
            'attractors': attractors,
            'phase_transitions': [t.to_dict() for t in transitions],
            'analysis_metadata': {
                'n_actors': len(positions),
                'n_attractors': len(attractors),
                'n_transitions': len(transitions),
                'coordinate_system': '4D Argentine Political Space'
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Political analysis exported to {filename}")