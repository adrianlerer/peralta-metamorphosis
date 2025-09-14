"""
Visualization Module for Paper 11
Comprehensive plotting functions for political actor network analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoliticalVisualization:
    """
    Comprehensive visualization class for political actor analysis.
    Supports static and interactive plots for network analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), 
                 style: str = 'seaborn-v0_8',
                 color_palette: str = 'husl'):
        """
        Initialize visualization class.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size
        style : str
            Matplotlib style
        color_palette : str
            Seaborn color palette
        """
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette(color_palette)
        
        # Color schemes for different visualizations
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'neutral': '#C73E1D',
            'lopez_rega': '#8B0000',
            'milei': '#FF4500',
            'highlight': '#FFD700'
        }
        
        # Political dimension categories
        self.dimension_categories = {
            'Economic Policy': ['market_orientation', 'state_intervention', 'fiscal_policy'],
            'Social Issues': ['social_liberalism', 'traditional_values', 'civil_rights'],
            'Political System': ['democracy_support', 'institutional_trust', 'populism_level'],
            'International Relations': ['nationalism', 'international_cooperation', 'sovereignty']
        }
    
    def plot_similarity_matrix(self, similarity_matrix: np.ndarray, 
                             actor_names: List[str],
                             title: str = "Political Actor Similarity Matrix",
                             save_path: Optional[str] = None,
                             interactive: bool = True) -> Optional[go.Figure]:
        """
        Plot similarity matrix as heatmap.
        
        Parameters:
        -----------
        similarity_matrix : np.ndarray
            Similarity matrix
        actor_names : list
            Names of political actors
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        interactive : bool
            Create interactive Plotly plot
            
        Returns:
        --------
        plotly.graph_objects.Figure or None
        """
        if interactive:
            # Interactive Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=actor_names,
                y=actor_names,
                colorscale='RdYlBu_r',
                zmin=0,
                zmax=1,
                text=np.round(similarity_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 8},
                hoverongaps=False,
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title="Political Actors",
                yaxis_title="Political Actors",
                width=800,
                height=800,
                font=dict(size=10)
            )
            
            # Highlight López Rega and Milei if present
            lopez_indices = [i for i, name in enumerate(actor_names) if 'López Rega' in str(name)]
            milei_indices = [i for i, name in enumerate(actor_names) if 'Milei' in str(name)]
            
            if lopez_indices and milei_indices:
                # Add rectangle annotations to highlight the comparison
                for lopez_idx in lopez_indices:
                    for milei_idx in milei_indices:
                        fig.add_shape(
                            type="rect",
                            x0=milei_idx-0.5, y0=lopez_idx-0.5,
                            x1=milei_idx+0.5, y1=lopez_idx+0.5,
                            line=dict(color="red", width=3)
                        )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            # Static matplotlib heatmap
            plt.figure(figsize=self.figsize)
            
            mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
            sns.heatmap(similarity_matrix, 
                       annot=True, 
                       fmt='.3f',
                       cmap='RdYlBu_r',
                       square=True,
                       mask=mask,
                       cbar_kws={"shrink": .8},
                       xticklabels=actor_names,
                       yticklabels=actor_names)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Political Actors', fontsize=12)
            plt.ylabel('Political Actors', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return None
    
    def plot_multidimensional_breakdown(self, similarities_by_category: Dict[str, float],
                                      title: str = "López Rega-Milei Multidimensional Similarity",
                                      save_path: Optional[str] = None,
                                      interactive: bool = True) -> Optional[go.Figure]:
        """
        Plot multidimensional similarity breakdown.
        
        Parameters:
        -----------
        similarities_by_category : dict
            Similarities by dimension category
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        interactive : bool
            Create interactive Plotly plot
            
        Returns:
        --------
        plotly.graph_objects.Figure or None
        """
        categories = list(similarities_by_category.keys())
        similarities = list(similarities_by_category.values())
        
        if interactive:
            # Interactive bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=similarities,
                    text=[f'{s:.3f}' for s in similarities],
                    textposition='auto',
                    marker_color=[
                        self.colors['primary'] if s >= 0.5 else self.colors['secondary']
                        for s in similarities
                    ],
                    hovertemplate='<b>%{x}</b><br>Similarity: %{y:.3f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title="Political Dimensions",
                yaxis_title="Similarity Score",
                yaxis=dict(range=[0, 1]),
                showlegend=False,
                width=800,
                height=500
            )
            
            # Add horizontal line at 0.5
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                         annotation_text="Neutral Similarity (0.5)")
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            # Static matplotlib plot
            plt.figure(figsize=self.figsize)
            
            colors = [self.colors['primary'] if s >= 0.5 else self.colors['secondary']
                     for s in similarities]
            
            bars = plt.bar(categories, similarities, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, sim in zip(bars, similarities):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{sim:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, 
                       label='Neutral Similarity (0.5)')
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Political Dimensions', fontsize=12)
            plt.ylabel('Similarity Score', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return None
    
    def plot_network_analysis(self, similarity_matrix: np.ndarray,
                            actor_names: List[str],
                            threshold: float = 0.7,
                            layout_algorithm: str = 'spring',
                            title: str = "Political Actor Network",
                            save_path: Optional[str] = None,
                            interactive: bool = True) -> Optional[go.Figure]:
        """
        Plot network analysis of political actors.
        
        Parameters:
        -----------
        similarity_matrix : np.ndarray
            Similarity matrix
        actor_names : list
            Names of political actors
        threshold : float
            Similarity threshold for edges
        layout_algorithm : str
            Network layout algorithm
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        interactive : bool
            Create interactive Plotly plot
            
        Returns:
        --------
        plotly.graph_objects.Figure or None
        """
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, actor in enumerate(actor_names):
            G.add_node(i, name=actor)
        
        # Add edges based on similarity threshold
        for i in range(len(actor_names)):
            for j in range(i+1, len(actor_names)):
                if similarity_matrix[i, j] >= threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        # Calculate layout
        if layout_algorithm == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout_algorithm == 'circular':
            pos = nx.circular_layout(G)
        elif layout_algorithm == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Calculate centrality measures
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        if interactive:
            # Interactive Plotly network
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(G.edges[edge]['weight'])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                  line=dict(width=2, color='#888'),
                                  hoverinfo='none',
                                  mode='lines')
            
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                actor_name = G.nodes[node]['name']
                node_text.append(actor_name)
                
                # Node size based on centrality
                size = 20 + centrality[node] * 40
                node_sizes.append(size)
                
                # Color coding
                if 'López Rega' in actor_name:
                    node_colors.append(self.colors['lopez_rega'])
                elif 'Milei' in actor_name:
                    node_colors.append(self.colors['milei'])
                else:
                    node_colors.append(self.colors['primary'])
                
                # Hover info
                connections = len(list(G.neighbors(node)))
                info = f"Actor: {actor_name}<br>"
                info += f"Connections: {connections}<br>"
                info += f"Centrality: {centrality[node]:.3f}<br>"
                info += f"Betweenness: {betweenness[node]:.3f}"
                node_info.append(info)
            
            node_trace = go.Scatter(x=node_x, y=node_y,
                                  mode='markers+text',
                                  hoverinfo='text',
                                  text=node_text,
                                  hovertext=node_info,
                                  textposition="middle center",
                                  marker=dict(
                                      size=node_sizes,
                                      color=node_colors,
                                      line=dict(width=2, color='white')
                                  ))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title={
                                   'text': f'{title}<br>(Similarity threshold: {threshold})',
                                   'x': 0.5,
                                   'xanchor': 'center'
                               },
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text=f"Network with {len(G.nodes())} actors and {len(G.edges())} connections",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(color='gray', size=12)
                               )],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               width=800,
                               height=600))
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            # Static matplotlib network
            plt.figure(figsize=self.figsize)
            
            # Draw edges
            edges = G.edges()
            nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.5, width=2)
            
            # Node colors
            node_colors = []
            for node in G.nodes():
                actor_name = G.nodes[node]['name']
                if 'López Rega' in actor_name:
                    node_colors.append(self.colors['lopez_rega'])
                elif 'Milei' in actor_name:
                    node_colors.append(self.colors['milei'])
                else:
                    node_colors.append(self.colors['primary'])
            
            # Node sizes based on centrality
            node_sizes = [300 + centrality[node] * 1000 for node in G.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=node_sizes, alpha=0.8)
            
            # Draw labels
            labels = {node: G.nodes[node]['name'] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
            
            plt.title(f'{title}\n(Similarity threshold: {threshold})', 
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            
            # Add legend
            lopez_patch = mpatches.Patch(color=self.colors['lopez_rega'], label='López Rega')
            milei_patch = mpatches.Patch(color=self.colors['milei'], label='Milei')
            other_patch = mpatches.Patch(color=self.colors['primary'], label='Other Actors')
            plt.legend(handles=[lopez_patch, milei_patch, other_patch], loc='upper right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return None
    
    def plot_bootstrap_results(self, bootstrap_results: Dict,
                             title: str = "Bootstrap Validation Results",
                             save_path: Optional[str] = None,
                             interactive: bool = True) -> Optional[go.Figure]:
        """
        Plot bootstrap validation results.
        
        Parameters:
        -----------
        bootstrap_results : dict
            Bootstrap validation results
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        interactive : bool
            Create interactive Plotly plot
            
        Returns:
        --------
        plotly.graph_objects.Figure or None
        """
        if 'lopez_rega_milei' not in bootstrap_results:
            logger.warning("No López Rega-Milei bootstrap results found")
            return None
        
        results = bootstrap_results['lopez_rega_milei']
        bootstrap_similarities = results['bootstrap_similarities']
        original_similarity = results['original_similarity']
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
        
        if interactive:
            # Interactive histogram with confidence intervals
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=bootstrap_similarities,
                nbinsx=50,
                name='Bootstrap Distribution',
                opacity=0.7,
                marker_color=self.colors['primary'],
                hovertemplate='Similarity: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ))
            
            # Original value line
            fig.add_vline(x=original_similarity, line_dash="solid", line_color="red",
                         annotation_text=f"Original: {original_similarity:.3f}")
            
            # Confidence interval
            fig.add_vrect(x0=ci_lower, x1=ci_upper, 
                         fillcolor="yellow", opacity=0.3,
                         annotation_text=f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Similarity Score",
                yaxis_title="Frequency",
                showlegend=False,
                width=800,
                height=500
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            # Static matplotlib histogram
            plt.figure(figsize=self.figsize)
            
            plt.hist(bootstrap_similarities, bins=50, alpha=0.7, 
                    color=self.colors['primary'], edgecolor='black')
            
            plt.axvline(original_similarity, color='red', linestyle='-', linewidth=2,
                       label=f'Original: {original_similarity:.3f}')
            
            plt.axvspan(ci_lower, ci_upper, alpha=0.3, color='yellow',
                       label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
            
            plt.xlabel('Similarity Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(title, fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return None
    
    def plot_pca_analysis(self, pca_results: Dict, actor_names: List[str],
                         title: str = "PCA Analysis of Political Actors",
                         save_path: Optional[str] = None,
                         interactive: bool = True) -> Optional[go.Figure]:
        """
        Plot PCA analysis results.
        
        Parameters:
        -----------
        pca_results : dict
            PCA results including components and explained variance
        actor_names : list
            Names of political actors
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        interactive : bool
            Create interactive Plotly plot
            
        Returns:
        --------
        plotly.graph_objects.Figure or None
        """
        components = pca_results['components']
        explained_variance = pca_results['explained_variance_ratio']
        
        if interactive:
            # Interactive scatter plot
            colors = []
            for name in actor_names:
                if 'López Rega' in name:
                    colors.append(self.colors['lopez_rega'])
                elif 'Milei' in name:
                    colors.append(self.colors['milei'])
                else:
                    colors.append(self.colors['primary'])
            
            fig = go.Figure(data=go.Scatter(
                x=components[:, 0],
                y=components[:, 1],
                mode='markers+text',
                text=actor_names,
                textposition="top center",
                marker=dict(
                    size=12,
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': f'{title}<br>PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%}',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title=f"First Principal Component ({explained_variance[0]:.1%})",
                yaxis_title=f"Second Principal Component ({explained_variance[1]:.1%})",
                showlegend=False,
                width=800,
                height=600
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            # Static matplotlib plot
            plt.figure(figsize=self.figsize)
            
            colors = []
            for name in actor_names:
                if 'López Rega' in name:
                    colors.append(self.colors['lopez_rega'])
                elif 'Milei' in name:
                    colors.append(self.colors['milei'])
                else:
                    colors.append(self.colors['primary'])
            
            scatter = plt.scatter(components[:, 0], components[:, 1], 
                                c=colors, s=100, alpha=0.8, edgecolors='white', linewidth=2)
            
            # Add labels
            for i, name in enumerate(actor_names):
                plt.annotate(name, (components[i, 0], components[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, ha='left')
            
            plt.xlabel(f'First Principal Component ({explained_variance[0]:.1%})', fontsize=12)
            plt.ylabel(f'Second Principal Component ({explained_variance[1]:.1%})', fontsize=12)
            plt.title(f'{title}\nExplained Variance: PC1={explained_variance[0]:.1%}, PC2={explained_variance[1]:.1%}', 
                     fontsize=14, fontweight='bold')
            
            # Add legend
            lopez_patch = mpatches.Patch(color=self.colors['lopez_rega'], label='López Rega')
            milei_patch = mpatches.Patch(color=self.colors['milei'], label='Milei')
            other_patch = mpatches.Patch(color=self.colors['primary'], label='Other Actors')
            plt.legend(handles=[lopez_patch, milei_patch, other_patch])
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return None
    
    def create_dashboard(self, analysis_results: Dict,
                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Parameters:
        -----------
        analysis_results : dict
            Complete analysis results
        save_path : str, optional
            Path to save dashboard
            
        Returns:
        --------
        plotly.graph_objects.Figure : Interactive dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Similarity Matrix', 'Multidimensional Analysis', 
                           'Network Analysis', 'Bootstrap Validation'),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Add similarity heatmap (simplified)
        if 'similarity_matrix' in analysis_results:
            similarity_matrix = analysis_results['similarity_matrix']
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    colorscale='RdYlBu_r',
                    showscale=False
                ),
                row=1, col=1
            )
        
        # Add multidimensional analysis
        if 'multidimensional_similarities' in analysis_results:
            similarities = analysis_results['multidimensional_similarities']
            categories = list(similarities.keys())
            values = list(similarities.values())
            
            fig.add_trace(
                go.Bar(x=categories, y=values, showlegend=False),
                row=1, col=2
            )
        
        # Add network metrics
        if 'network_metrics' in analysis_results:
            metrics = analysis_results['network_metrics']
            fig.add_trace(
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Add bootstrap distribution
        if 'bootstrap_results' in analysis_results:
            bootstrap_data = analysis_results['bootstrap_results']['lopez_rega_milei']['bootstrap_similarities']
            fig.add_trace(
                go.Histogram(x=bootstrap_data, showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Paper 11: Political Actor Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def export_for_gephi(self, similarity_matrix: np.ndarray, 
                        actor_names: List[str],
                        threshold: float = 0.7,
                        output_path: str = "gephi_export.gexf"):
        """
        Export network data for Gephi visualization.
        
        Parameters:
        -----------
        similarity_matrix : np.ndarray
            Similarity matrix
        actor_names : list
            Names of political actors
        threshold : float
            Similarity threshold for edges
        output_path : str
            Output file path
        """
        # Create network graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for i, actor in enumerate(actor_names):
            G.add_node(i, label=actor, name=actor)
        
        # Add edges based on similarity threshold
        for i in range(len(actor_names)):
            for j in range(i+1, len(actor_names)):
                if similarity_matrix[i, j] >= threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        # Export to GEXF format for Gephi
        nx.write_gexf(G, output_path)
        logger.info(f"Network exported to Gephi format: {output_path}")
    
    def save_all_plots(self, analysis_results: Dict, output_dir: str = "plots"):
        """
        Save all plots to specified directory.
        
        Parameters:
        -----------
        analysis_results : dict
            Complete analysis results
        output_dir : str
            Output directory for plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Similarity matrix
        if 'similarity_matrix' in analysis_results and 'actor_names' in analysis_results:
            self.plot_similarity_matrix(
                analysis_results['similarity_matrix'],
                analysis_results['actor_names'],
                save_path=f"{output_dir}/similarity_matrix.png"
            )
        
        # Multidimensional analysis
        if 'multidimensional_similarities' in analysis_results:
            self.plot_multidimensional_breakdown(
                analysis_results['multidimensional_similarities'],
                save_path=f"{output_dir}/multidimensional_analysis.png"
            )
        
        # Network analysis
        if 'similarity_matrix' in analysis_results and 'actor_names' in analysis_results:
            self.plot_network_analysis(
                analysis_results['similarity_matrix'],
                analysis_results['actor_names'],
                save_path=f"{output_dir}/network_analysis.png"
            )
        
        # Bootstrap results
        if 'bootstrap_results' in analysis_results:
            self.plot_bootstrap_results(
                analysis_results['bootstrap_results'],
                save_path=f"{output_dir}/bootstrap_validation.png"
            )
        
        # PCA analysis
        if 'pca_results' in analysis_results and 'actor_names' in analysis_results:
            self.plot_pca_analysis(
                analysis_results['pca_results'],
                analysis_results['actor_names'],
                save_path=f"{output_dir}/pca_analysis.png"
            )
        
        # Dashboard
        dashboard = self.create_dashboard(analysis_results)
        dashboard.write_html(f"{output_dir}/dashboard.html")
        
        logger.info(f"All plots saved to {output_dir}")

# Utility functions for custom visualizations
def create_correlation_matrix_plot(data: pd.DataFrame, 
                                  dimensions: List[str],
                                  method: str = 'pearson') -> go.Figure:
    """
    Create correlation matrix plot for political dimensions.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Political actors data
    dimensions : list
        Political dimensions to analyze
    method : str
        Correlation method
        
    Returns:
    --------
    plotly.graph_objects.Figure : Correlation matrix plot
    """
    corr_matrix = data[dimensions].corr(method=method)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Political Dimensions Correlation Matrix",
        xaxis_title="Dimensions",
        yaxis_title="Dimensions",
        width=600,
        height=600
    )
    
    return fig

def plot_actor_radar_chart(actor_data: pd.Series, 
                          dimensions: List[str],
                          actor_name: str) -> go.Figure:
    """
    Create radar chart for individual political actor.
    
    Parameters:
    -----------
    actor_data : pd.Series
        Actor's political dimension scores
    dimensions : list
        Political dimensions
    actor_name : str
        Name of the actor
        
    Returns:
    --------
    plotly.graph_objects.Figure : Radar chart
    """
    values = [actor_data[dim] for dim in dimensions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=dimensions + [dimensions[0]],
        fill='toself',
        name=actor_name,
        line=dict(color='rgb(32, 146, 230)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title=f"Political Profile: {actor_name}",
        showlegend=True
    )
    
    return fig