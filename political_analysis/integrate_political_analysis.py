"""
Integrated Political Analysis: Applying RootFinder and Legal-Memespace to Argentine Political Antagonisms
Combines genealogical tracing with spatial mapping for comprehensive political evolution analysis
Author: Ignacio Adrián Lerer
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import our adapted tools
from political_rootfinder import PoliticalRootFinder
from political_memespace import PoliticalMemespace

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedPoliticalAnalysis:
    """
    Integrates Political RootFinder and Political Memespace for comprehensive analysis
    of Argentine political antagonisms (1810-2025).
    """
    
    def __init__(self):
        self.rootfinder = PoliticalRootFinder(
            min_semantic_similarity=0.3,
            similarity_threshold=0.4
        )
        self.memespace = PoliticalMemespace(n_dimensions=4)
        
        # Results storage
        self.genealogies = {}
        self.positions = {}
        self.attractors = {}
        self.transitions = []
        self.integrated_results = {}
        
    def load_sample_political_documents(self) -> pd.DataFrame:
        """
        Load sample political documents for demonstration.
        In real usage, this would load actual historical documents.
        """
        logger.info("Loading sample political documents")
        
        # Sample documents representing key political moments
        documents = [
            {
                'document_id': 'Moreno_Plan_1810',
                'author': 'Mariano Moreno', 
                'year': 1810,
                'text': 'El pueblo tiene derecho a saber la conducta de sus representantes. La libertad de escribir es la base fundamental de la democracia. Buenos Aires debe liderar la revolución americana.',
                'political_position': [0.3, 0.2, 0.7, 0.8],  # Central, Port, Popular, Revolution
                'outcome': 'failure'  # Moreno was marginalized
            },
            {
                'document_id': 'Saavedra_Moderacion_1810',
                'author': 'Cornelio Saavedra',
                'year': 1810, 
                'text': 'Las provincias deben tener voz en el gobierno. La revolución debe ser gradual y prudente. El interior no puede ser ignorado por Buenos Aires.',
                'political_position': [0.7, 0.8, 0.6, 0.3],  # Federal, Interior, Popular, Gradual
                'outcome': 'success'  # Saavedra prevailed initially
            },
            {
                'document_id': 'Rivadavia_Unitario_1826',
                'author': 'Bernardino Rivadavia',
                'year': 1826,
                'text': 'La nación necesita un gobierno central fuerte. Las provincias deben subordinarse al poder nacional. La ilustración debe guiar al pueblo.',
                'political_position': [0.1, 0.1, 0.2, 0.4],  # Central, Port, Elite, Moderate
                'outcome': 'failure'  # Rivadavia resigned
            },
            {
                'document_id': 'Rosas_Federacion_1835',
                'author': 'Juan Manuel de Rosas',
                'year': 1835,
                'text': 'La federación es sagrada. Buenos Aires lidera pero respeta las provincias. El pueblo debe seguir a sus caudillos naturales. Orden y tradición.',
                'political_position': [0.6, 0.4, 0.8, 0.2],  # Federal, Mixed, Popular, Conservative
                'outcome': 'success'  # Rosas dominated for 20 years
            },
            {
                'document_id': 'Urquiza_Confederacion_1852',
                'author': 'Justo José de Urquiza', 
                'year': 1852,
                'text': 'Las provincias unidas bajo una constitución federal. El interior tiene los mismos derechos que Buenos Aires. Organización nacional.',
                'political_position': [0.8, 0.9, 0.5, 0.3],  # Federal, Interior, Mixed, Gradual
                'outcome': 'partial'  # Won against Rosas, lost to Mitre
            },
            {
                'document_id': 'Mitre_Organizacion_1862',
                'author': 'Bartolomé Mitre',
                'year': 1862,
                'text': 'Buenos Aires debe liderar la organización nacional. Progreso y civilización. Las instituciones liberales son el futuro de América.',
                'political_position': [0.3, 0.2, 0.3, 0.4],  # Central, Port, Elite, Moderate  
                'outcome': 'success'  # Became president, organized nation
            },
            {
                'document_id': 'Peron_Justicia_1945',
                'author': 'Juan Domingo Perón',
                'year': 1945,
                'text': 'Justicia social para los trabajadores. La patria es del pueblo, no de la oligarquía. Ni yanquis ni marxistas, peronistas.',
                'political_position': [0.5, 0.5, 0.9, 0.6],  # Mixed, Mixed, Popular, Moderate Revolution
                'outcome': 'success'  # Won elections, transformed Argentina
            },
            {
                'document_id': 'Antiperonismo_1955', 
                'author': 'Pedro Eugenio Aramburu',
                'year': 1955,
                'text': 'Libertar a la patria de la demagogia. Restituir las instituciones democráticas. La república no es compatible con el personalismo.',
                'political_position': [0.2, 0.2, 0.2, 0.7],  # Central, Port, Elite, Revolution
                'outcome': 'partial'  # Overthrew Perón but couldn\'t eliminate Peronism
            },
            {
                'document_id': 'Alfonsin_Democracia_1983',
                'author': 'Raúl Alfonsín', 
                'year': 1983,
                'text': 'Con la democracia se come, se cura y se educa. Nunca más a las violaciones de derechos humanos. República y constitución.',
                'political_position': [0.4, 0.4, 0.6, 0.3],  # Moderate Central, Mixed, Popular, Gradual
                'outcome': 'partial'  # Democratic transition but economic failure
            },
            {
                'document_id': 'Menem_Neoliberal_1990',
                'author': 'Carlos Menem',
                'year': 1990,
                'text': 'Cirugía mayor sin anestesia. Mercado libre y competencia. Relaciones carnales con Estados Unidos. Modernización del estado.',
                'political_position': [0.2, 0.3, 0.1, 0.8],  # Central, Port leaning, Elite, Revolution
                'outcome': 'success'  # Economic transformation, reelection
            },
            {
                'document_id': 'Kirchner_Popular_2003',
                'author': 'Néstor Kirchner',
                'year': 2003, 
                'text': 'Desendeudarnos y volver a creer. El estado presente en la economía. Derechos humanos como política de estado. Patria grande sudamericana.',
                'political_position': [0.6, 0.7, 0.8, 0.5],  # Federal, Interior, Popular, Moderate
                'outcome': 'success'  # Economic recovery, political realignment
            },
            {
                'document_id': 'Macri_Cambio_2015',
                'author': 'Mauricio Macri',
                'year': 2015,
                'text': 'Cambiemos para volver al mundo. Gradualismo y diálogo. Instituciones republicanas. Mercado y competencia responsable.',
                'political_position': [0.3, 0.2, 0.3, 0.2],  # Central, Port, Elite, Gradual
                'outcome': 'failure'  # Lost reelection after economic crisis
            },
            {
                'document_id': 'Milei_Liberal_2023',
                'author': 'Javier Milei',
                'year': 2023,
                'text': 'Viva la libertad carajo. Afuera el estado de todos lados. Dolarización y competencia de monedas. La casta política es el enemigo.',
                'political_position': [0.1, 0.2, 0.2, 0.9],  # Minimal State, Port, Elite, Total Rupture
                'outcome': 'success'  # Won presidency promising radical change
            }
        ]
        
        return pd.DataFrame(documents)
    
    def run_complete_analysis(self) -> Dict:
        """
        Run complete integrated analysis combining RootFinder and Memespace.
        """
        logger.info("Starting complete integrated political analysis")
        
        # 1. Load documents
        documents_df = self.load_sample_political_documents()
        logger.info(f"Loaded {len(documents_df)} political documents")
        
        # 2. Build semantic network for RootFinder
        semantic_network = self.rootfinder.build_semantic_network(documents_df)
        
        # 3. Trace genealogies for modern political movements
        modern_movements = [
            'Peron_Justicia_1945',
            'Macri_Cambio_2015', 
            'Milei_Liberal_2023'
        ]
        
        logger.info("Tracing political genealogies")
        for movement in modern_movements:
            genealogy = self.rootfinder.trace_political_genealogy(
                movement, semantic_network, max_depth=6
            )
            self.genealogies[movement] = genealogy
            logger.info(f"Traced genealogy for {movement}: {len(genealogy)} generations")
        
        # 4. Find common ancestors
        common_ancestors = {}
        movement_pairs = [
            ('Peron_Justicia_1945', 'Macri_Cambio_2015'),
            ('Peron_Justicia_1945', 'Milei_Liberal_2023'),
            ('Macri_Cambio_2015', 'Milei_Liberal_2023')
        ]
        
        for mov1, mov2 in movement_pairs:
            ancestor = self.rootfinder.find_common_political_ancestor(mov1, mov2, semantic_network)
            common_ancestors[f"{mov1}_vs_{mov2}"] = ancestor
            logger.info(f"Common ancestor of {mov1} and {mov2}: {ancestor}")
        
        # 5. Map political positions in 4D space  
        self.positions = self.memespace.map_political_positions(documents_df)
        logger.info(f"Mapped {len(self.positions)} political positions to 4D space")
        
        # 6. Analyze grieta evolution
        grieta_evolution = self.memespace.analyze_grieta_evolution(documents_df, window_years=20)
        logger.info(f"Analyzed grieta evolution: {len(grieta_evolution)} time points")
        
        # 7. Find political attractors
        self.attractors = self.memespace.find_political_attractors(self.positions, n_attractors=3)
        logger.info(f"Found {len(self.attractors)} political attractors")
        
        # 8. Detect phase transitions
        self.transitions = self.memespace.detect_political_phase_transitions(documents_df)
        logger.info(f"Detected {len(self.transitions)} major phase transitions")
        
        # 9. Calculate political fitness
        outcomes = {doc['author']: doc['outcome'] for _, doc in documents_df.iterrows()}
        fitness_scores = self.memespace.calculate_political_fitness(self.positions, outcomes)
        
        # 10. Integrate results
        self.integrated_results = self._integrate_genealogy_and_space()
        
        # Compile complete results
        results = {
            'genealogies': {k: [node.to_dict() for node in v] for k, v in self.genealogies.items()},
            'common_ancestors': common_ancestors,
            'positions_4d': self.positions,
            'political_attractors': self.attractors,
            'grieta_evolution': grieta_evolution,
            'phase_transitions': [t.to_dict() for t in self.transitions],
            'fitness_scores': fitness_scores,
            'integration_analysis': self.integrated_results,
            'metadata': {
                'n_documents': len(documents_df),
                'timespan': f"{documents_df['year'].min()}-{documents_df['year'].max()}",
                'n_genealogies': len(self.genealogies),
                'n_attractors': len(self.attractors),
                'n_transitions': len(self.transitions)
            }
        }
        
        logger.info("Complete integrated analysis finished")
        return results
    
    def _integrate_genealogy_and_space(self) -> Dict:
        """
        Integrate genealogical findings with spatial analysis.
        
        Key questions:
        - Do genealogically related movements cluster in 4D space?
        - Do movements with common ancestors have similar trajectories?
        - Can spatial position predict genealogical inheritance strength?
        """
        logger.info("Integrating genealogy and spatial analysis")
        
        integration_results = {}
        
        # 1. Correlation between genealogical and spatial distance
        genealogy_distances = []
        spatial_distances = []
        
        for movement, genealogy in self.genealogies.items():
            if len(genealogy) >= 2:
                # Calculate genealogical depth
                genealogical_distance = len(genealogy)
                
                # Calculate spatial distance from political center
                if movement in self.positions:
                    spatial_distance = np.linalg.norm(
                        np.array(self.positions[movement]) - [0.5, 0.5, 0.5, 0.5]
                    )
                    
                    genealogy_distances.append(genealogical_distance)
                    spatial_distances.append(spatial_distance)
        
        if len(genealogy_distances) >= 2:
            correlation, p_value = pearsonr(genealogy_distances, spatial_distances)
            integration_results['genealogy_space_correlation'] = {
                'correlation': correlation,
                'p_value': p_value,
                'interpretation': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'
            }
        
        # 2. Attractor genealogical analysis
        attractor_genealogies = {}
        for attractor_name, coords in self.attractors.items():
            # Find closest political actor to this attractor
            min_distance = float('inf')
            closest_actor = None
            
            for actor, actor_coords in self.positions.items():
                if actor in [g.split('_')[0] for g in self.genealogies.keys()]:  # Only genealogically traced actors
                    distance = np.linalg.norm(np.array(actor_coords) - np.array(coords))
                    if distance < min_distance:
                        min_distance = distance
                        closest_actor = actor
            
            attractor_genealogies[attractor_name] = {
                'closest_actor': closest_actor,
                'distance': min_distance
            }
        
        integration_results['attractor_genealogies'] = attractor_genealogies
        
        # 3. Phase transition genealogical impact
        transition_impacts = []
        for transition in self.transitions:
            transition_year = int(transition.date)
            
            # Find genealogies that span this transition
            spanning_genealogies = []
            for movement, genealogy in self.genealogies.items():
                years_in_genealogy = [node.year for node in genealogy]
                if min(years_in_genealogy) <= transition_year <= max(years_in_genealogy):
                    spanning_genealogies.append(movement)
            
            transition_impacts.append({
                'transition_year': transition_year,
                'affected_genealogies': spanning_genealogies,
                'transition_magnitude': transition.magnitude,
                'political_event': transition.political_event
            })
        
        integration_results['transition_impacts'] = transition_impacts
        
        # 4. Predictive model assessment
        # Can we predict genealogical inheritance from spatial position?
        inheritance_strengths = []
        spatial_positions = []
        
        for movement, genealogy in self.genealogies.items():
            inheritance_strength = self.rootfinder.calculate_inheritance_strength(genealogy)
            if movement in self.positions:
                inheritance_strengths.append(inheritance_strength)
                spatial_positions.append(self.positions[movement])
        
        if len(inheritance_strengths) >= 2:
            # Simple predictive correlation
            position_norms = [np.linalg.norm(pos) for pos in spatial_positions]
            pred_correlation, pred_p_value = pearsonr(inheritance_strengths, position_norms)
            
            integration_results['inheritance_prediction'] = {
                'correlation': pred_correlation,
                'p_value': pred_p_value,
                'predictive_power': 'Good' if abs(pred_correlation) > 0.6 else 'Moderate' if abs(pred_correlation) > 0.3 else 'Poor'
            }
        
        return integration_results
    
    def generate_visualizations(self, results: Dict, output_dir: str = 'visualizations/'):
        """Generate comprehensive visualizations of the integrated analysis."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Generating visualizations in {output_dir}")
        
        # 1. Political genealogy tree
        self._plot_genealogy_tree(results, f"{output_dir}political_genealogy_tree.png")
        
        # 2. 4D political space (3D projection)
        self._plot_4d_political_space(results, f"{output_dir}political_space_4d.png")
        
        # 3. Grieta evolution timeline
        self._plot_grieta_evolution(results, f"{output_dir}grieta_evolution.png")
        
        # 4. Integration correlation analysis
        self._plot_integration_analysis(results, f"{output_dir}integration_analysis.png")
        
        logger.info("All visualizations generated successfully")
    
    def _plot_genealogy_tree(self, results: Dict, filename: str):
        """Plot political genealogy tree."""
        fig, axes = plt.subplots(1, len(results['genealogies']), figsize=(15, 8))
        if len(results['genealogies']) == 1:
            axes = [axes]
        
        for i, (movement, genealogy) in enumerate(results['genealogies'].items()):
            ax = axes[i]
            
            # Create simple genealogy visualization
            generations = [node['generation'] for node in genealogy]
            years = [node['year'] for node in genealogy]
            
            ax.scatter(generations, years, s=100, alpha=0.7)
            ax.plot(generations, years, '--', alpha=0.5)
            
            # Add labels
            for j, node in enumerate(genealogy):
                if j % 2 == 0:  # Label every other node to avoid crowding
                    ax.annotate(node['author'], 
                              (node['generation'], node['year']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8)
            
            ax.set_xlabel('Generation (0=Modern)')
            ax.set_ylabel('Year')
            ax.set_title(f"Genealogy: {movement.split('_')[0]}")
            ax.invert_xaxis()  # Modern (0) on left, ancient on right
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_4d_political_space(self, results: Dict, filename: str):
        """Plot 4D political space as 3D projection."""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates (use first 3 dimensions for 3D plot)
        actors = list(results['positions_4d'].keys())
        coords = np.array(list(results['positions_4d'].values()))
        
        # Color code by time period
        colors = []
        for actor in actors:
            if '1810' in actor or '1826' in actor or '1835' in actor:
                colors.append('red')  # Early period
            elif '1852' in actor or '1862' in actor:
                colors.append('orange')  # Organization period
            elif '1945' in actor or '1955' in actor:
                colors.append('blue')  # Peronist era
            elif '1983' in actor or '1990' in actor or '2003' in actor:
                colors.append('green')  # Democratic era
            else:
                colors.append('purple')  # Contemporary
        
        # Plot points
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                           c=colors, s=100, alpha=0.7)
        
        # Add attractors
        attractor_coords = np.array(list(results['political_attractors'].values()))
        ax.scatter(attractor_coords[:, 0], attractor_coords[:, 1], attractor_coords[:, 2],
                  marker='*', s=300, c='black', label='Attractors')
        
        ax.set_xlabel('Centralization ← → Federalism')
        ax.set_ylabel('Buenos Aires ← → Interior') 
        ax.set_zlabel('Elite ← → Popular')
        ax.set_title('Argentine Political Space (4D→3D Projection)')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_grieta_evolution(self, results: Dict, filename: str):
        """Plot evolution of political polarization over time."""
        grieta_data = results['grieta_evolution']
        years = [point[0] for point in grieta_data]
        distances = [point[1] for point in grieta_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, distances, 'o-', linewidth=2, markersize=6)
        
        # Highlight major transitions
        for transition in results['phase_transitions']:
            plt.axvline(x=int(transition['date']), color='red', linestyle='--', alpha=0.7)
            plt.text(int(transition['date']), max(distances) * 0.9, 
                    transition['political_event'], 
                    rotation=90, fontsize=8)
        
        plt.xlabel('Year')
        plt.ylabel('Maximum Political Distance (Grieta Depth)')
        plt.title('Evolution of Political Polarization in Argentina (1810-2025)')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_integration_analysis(self, results: Dict, filename: str):
        """Plot correlation between genealogical and spatial analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        integration = results['integration_analysis']
        
        # Plot 1: Genealogy-Space Correlation (if available)
        if 'genealogy_space_correlation' in integration:
            corr_data = integration['genealogy_space_correlation']
            axes[0, 0].bar(['Genealogical Distance', 'Spatial Distance'], [1, corr_data['correlation']])
            axes[0, 0].set_title(f"Genealogy-Space Correlation: {corr_data['correlation']:.3f}")
            axes[0, 0].set_ylabel('Normalized Values')
        
        # Plot 2: Attractor Analysis
        if 'attractor_genealogies' in integration:
            attractor_names = list(integration['attractor_genealogies'].keys())
            distances = [integration['attractor_genealogies'][name]['distance'] 
                        for name in attractor_names]
            axes[0, 1].bar(range(len(attractor_names)), distances)
            axes[0, 1].set_xticks(range(len(attractor_names)))
            axes[0, 1].set_xticklabels([name.split('_')[1] for name in attractor_names], rotation=45)
            axes[0, 1].set_title('Distance to Political Attractors')
            axes[0, 1].set_ylabel('Distance')
        
        # Plot 3: Phase Transition Timeline
        if 'transition_impacts' in integration:
            transitions = integration['transition_impacts']
            years = [t['transition_year'] for t in transitions]
            magnitudes = [t['transition_magnitude'] for t in transitions]
            
            axes[1, 0].scatter(years, magnitudes, s=100, alpha=0.7)
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Transition Magnitude')
            axes[1, 0].set_title('Political Phase Transitions')
            
            # Add event labels
            for i, transition in enumerate(transitions):
                axes[1, 0].annotate(transition['political_event'][:10], 
                                   (years[i], magnitudes[i]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8)
        
        # Plot 4: Inheritance Prediction (if available)
        if 'inheritance_prediction' in integration:
            pred_data = integration['inheritance_prediction']
            axes[1, 1].bar(['Actual', 'Predicted'], [1, abs(pred_data['correlation'])])
            axes[1, 1].set_title(f"Inheritance Predictability: {pred_data['predictive_power']}")
            axes[1, 1].set_ylabel('Correlation Strength')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, results: Dict, filename: str = 'political_analysis_results.json'):
        """Export complete results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to {filename}")

def main():
    """
    Main execution function demonstrating the integrated political analysis.
    """
    print("="*80)
    print("INTEGRATED POLITICAL ANALYSIS: ROOTFINDER + MEMESPACE")
    print("Argentine Political Antagonisms (1810-2025)")
    print("="*80)
    
    # Initialize analyzer
    analyzer = IntegratedPoliticalAnalysis()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Print key findings
    print("\n🔍 KEY FINDINGS:")
    print(f"📊 Analyzed {results['metadata']['n_documents']} political documents")
    print(f"🌳 Traced {results['metadata']['n_genealogies']} political genealogies") 
    print(f"📍 Identified {results['metadata']['n_attractors']} stable political attractors")
    print(f"🔄 Detected {results['metadata']['n_transitions']} major phase transitions")
    
    # Display common ancestors
    print("\n👨‍👩‍👧‍👦 COMMON POLITICAL ANCESTORS:")
    for pair, ancestor in results['common_ancestors'].items():
        print(f"   {pair}: {ancestor}")
    
    # Display attractors
    print("\n📍 POLITICAL ATTRACTORS:")
    for name, coords in results['political_attractors'].items():
        print(f"   {name}: {[f'{c:.2f}' for c in coords]}")
    
    # Display grieta evolution
    grieta_current = results['grieta_evolution'][-1][1] if results['grieta_evolution'] else 0
    grieta_start = results['grieta_evolution'][0][1] if results['grieta_evolution'] else 0
    print(f"\n📈 GRIETA EVOLUTION:")
    print(f"   Start ({results['grieta_evolution'][0][0]}): {grieta_start:.3f}")
    print(f"   Current ({results['grieta_evolution'][-1][0]}): {grieta_current:.3f}")
    print(f"   Change: {((grieta_current - grieta_start) / grieta_start * 100):+.1f}%")
    
    # Integration results
    if 'genealogy_space_correlation' in results['integration_analysis']:
        corr = results['integration_analysis']['genealogy_space_correlation']['correlation']
        print(f"\n🔗 INTEGRATION ANALYSIS:")
        print(f"   Genealogy-Space Correlation: {corr:.3f}")
        print(f"   Interpretation: {results['integration_analysis']['genealogy_space_correlation']['interpretation']}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    analyzer.generate_visualizations(results)
    
    # Export results
    print("💾 Exporting results...")
    analyzer.export_results(results)
    
    print("\n✅ Analysis complete! Check 'political_analysis_results.json' and 'visualizations/' folder")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()