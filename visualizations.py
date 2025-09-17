#!/usr/bin/env python3
"""
Argentine Legal Evolution Dataset - Visualization Suite
Version 1.0.0-beta
Author: Adrian Lerer  
Date: September 2025

This script generates comprehensive visualizations for the Argentine Legal Evolution Dataset,
supporting the empirical analysis of "The Extended Phenotype of Law" research.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
FIGURE_SIZE = (12, 8)
DPI = 300
OUTPUT_DIR = "visualizations"

# Create output directory
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load datasets
print("Loading Argentine Legal Evolution Dataset...")

evolution_cases = pd.read_csv("evolution_cases.csv", parse_dates=['fecha_inicio', 'fecha_fin'])
velocity_metrics = pd.read_csv("velocity_metrics.csv")
transplants_tracking = pd.read_csv("transplants_tracking.csv", parse_dates=['introduction_date', 'modification_date'])
crisis_periods = pd.read_csv("crisis_periods.csv", parse_dates=['start_date', 'end_date'])
innovations_exported = pd.read_csv("innovations_exported.csv", parse_dates=['origin_date'])

print(f"Loaded datasets:")
print(f"- Evolution Cases: {len(evolution_cases)} observations")
print(f"- Velocity Metrics: {len(velocity_metrics)} observations") 
print(f"- Transplants: {len(transplants_tracking)} observations")
print(f"- Crisis Periods: {len(crisis_periods)} observations")
print(f"- Innovations Exported: {len(innovations_exported)} observations")
print()

# ================================================================================
# 1. TEMPORAL EVOLUTION TIMELINE
# ================================================================================

def create_evolution_timeline():
    """Create comprehensive timeline of legal evolution cases."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Prepare data
    evolution_cases['decade'] = (evolution_cases['fecha_inicio'].dt.year // 10) * 10
    evolution_cases['duration_years'] = (evolution_cases['fecha_fin'] - evolution_cases['fecha_inicio']).dt.days / 365.25
    
    # Top panel: Cases by decade and success
    decade_success = evolution_cases.groupby(['decade', 'exito']).size().unstack(fill_value=0)
    decade_success.plot(kind='bar', stacked=True, ax=ax1, color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
    ax1.set_title('Legal Evolution Cases by Decade and Success Level', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Decade')
    ax1.set_ylabel('Number of Cases')
    ax1.legend(title='Success Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Bottom panel: Survival distribution
    survival_data = evolution_cases.dropna(subset=['supervivencia_anos'])
    ax2.hist(survival_data['supervivencia_anos'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(survival_data['supervivencia_anos'].mean(), color='red', linestyle='--', 
                label=f'Mean: {survival_data["supervivencia_anos"].mean():.1f} years')
    ax2.axvline(survival_data['supervivencia_anos'].median(), color='orange', linestyle='--',
                label=f'Median: {survival_data["supervivencia_anos"].median():.1f} years')
    ax2.set_title('Distribution of Legal Innovation Survival Times', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Survival (Years)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/evolution_timeline.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("✓ Created evolution timeline visualization")

# ================================================================================
# 2. CRISIS ACCELERATION ANALYSIS
# ================================================================================

def create_crisis_acceleration_plot():
    """Visualize legal change acceleration during crisis periods."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel 1: Acceleration factor by crisis severity
    severity_order = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
    crisis_clean = crisis_periods.dropna(subset=['acceleration_factor', 'severity_level'])
    
    sns.boxplot(data=crisis_clean, x='severity_level', y='acceleration_factor', 
                order=severity_order, ax=ax1)
    ax1.set_title('Legal Change Acceleration by Crisis Severity', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Crisis Severity')
    ax1.set_ylabel('Acceleration Factor (x normal velocity)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Panel 2: Crisis timeline
    crisis_periods['duration'] = (crisis_periods['end_date'] - crisis_periods['start_date']).dt.days
    crisis_periods['year'] = crisis_periods['start_date'].dt.year
    
    scatter_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Very High': 'darkred', 'Extreme': 'black'}
    for severity in severity_order:
        data = crisis_periods[crisis_periods['severity_level'] == severity]
        if not data.empty:
            ax2.scatter(data['year'], data['duration'], 
                       label=severity, alpha=0.7, s=60,
                       c=[scatter_colors.get(severity, 'blue')])
    
    ax2.set_title('Crisis Duration vs Year by Severity', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Crisis Duration (Days)')
    ax2.legend(title='Severity')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Legal changes during crises
    crisis_changes = crisis_periods.dropna(subset=['legal_changes_count'])
    ax3.scatter(crisis_changes['acceleration_factor'], crisis_changes['legal_changes_count'], 
                alpha=0.6, s=80, c='purple')
    ax3.set_title('Acceleration vs Total Legal Changes', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Acceleration Factor')
    ax3.set_ylabel('Total Legal Changes')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation
    if len(crisis_changes) > 2:
        correlation = crisis_changes['acceleration_factor'].corr(crisis_changes['legal_changes_count'])
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 4: Emergency decrees vs regular laws
    crisis_legislation = crisis_periods.dropna(subset=['emergency_decrees', 'new_laws'])
    ax4.scatter(crisis_legislation['emergency_decrees'], crisis_legislation['new_laws'], 
                alpha=0.6, s=80, c='orange')
    ax4.set_title('Emergency Decrees vs New Laws During Crises', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Emergency Decrees')
    ax4.set_ylabel('New Laws')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    if len(crisis_legislation) > 2:
        z = np.polyfit(crisis_legislation['emergency_decrees'], crisis_legislation['new_laws'], 1)
        p = np.poly1d(z)
        ax4.plot(crisis_legislation['emergency_decrees'], p(crisis_legislation['emergency_decrees']), 
                 "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/crisis_acceleration.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("✓ Created crisis acceleration analysis")

# ================================================================================
# 3. TRANSPLANT SUCCESS PATTERNS
# ================================================================================

def create_transplant_analysis():
    """Analyze patterns of legal transplant success and adaptation."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel 1: Success by origin legal family
    transplant_success = transplants_tracking.groupby(['origin_legal_family', 'success_level']).size().unstack(fill_value=0)
    transplant_success_pct = transplant_success.div(transplant_success.sum(axis=1), axis=0) * 100
    
    transplant_success_pct.plot(kind='bar', stacked=True, ax=ax1, 
                               color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    ax1.set_title('Transplant Success Rate by Origin Legal Family', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Origin Legal Family')
    ax1.set_ylabel('Percentage')
    ax1.legend(title='Success Level', bbox_to_anchor=(1.05, 1))
    ax1.tick_params(axis='x', rotation=45)
    
    # Panel 2: Adaptation requirements vs success
    adaptation_success = pd.crosstab(transplants_tracking['adaptation_required'], 
                                   transplants_tracking['success_level'], normalize='index') * 100
    
    sns.heatmap(adaptation_success, annot=True, fmt='.1f', ax=ax2, cmap='RdYlGn')
    ax2.set_title('Success Rate by Adaptation Requirements', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Success Level')
    ax2.set_ylabel('Adaptation Required')
    
    # Panel 3: Survival vs mutations
    transplant_clean = transplants_tracking.dropna(subset=['survival_years', 'mutations_count'])
    ax3.scatter(transplant_clean['mutations_count'], transplant_clean['survival_years'], 
                alpha=0.6, s=60, c='green')
    ax3.set_title('Transplant Survival vs Mutations', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Mutations')
    ax3.set_ylabel('Survival (Years)')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Resistance levels
    resistance_counts = transplants_tracking['local_resistance_level'].value_counts()
    wedges, texts, autotexts = ax4.pie(resistance_counts.values, labels=resistance_counts.index, 
                                       autopct='%1.1f%%', startangle=90)
    ax4.set_title('Distribution of Local Resistance Levels', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/transplant_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("✓ Created transplant success analysis")

# ================================================================================
# 4. VELOCITY TRENDS OVER TIME
# ================================================================================

def create_velocity_trends():
    """Analyze velocity of legal change over time."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Extract year ranges and convert to numeric for plotting
    velocity_metrics['period_start'] = velocity_metrics['period'].str.split('-').str[0].astype(float)
    velocity_metrics['period_end'] = velocity_metrics['period'].str.split('-').str[-1].astype(float)
    velocity_metrics['period_mid'] = (velocity_metrics['period_start'] + velocity_metrics['period_end']) / 2
    
    # Panel 1: Reform frequency over time
    reform_freq = velocity_metrics[velocity_metrics['metric_type'] == 'Reform_Frequency'].copy()
    if not reform_freq.empty:
        ax1.plot(reform_freq['period_mid'], reform_freq['value'], 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_title('Legal Reform Frequency Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Reforms per Year')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(reform_freq['period_mid'], reform_freq['value'], 1)
        p = np.poly1d(z)
        ax1.plot(reform_freq['period_mid'], p(reform_freq['period_mid']), "r--", alpha=0.8, 
                 label=f'Trend (slope: {z[0]:.3f})')
        ax1.legend()
    
    # Panel 2: Innovation speed in financial sector
    innovation_speed = velocity_metrics[velocity_metrics['metric_type'] == 'Innovation_Speed'].copy()
    if not innovation_speed.empty:
        bars = ax2.bar(innovation_speed['period_mid'], innovation_speed['value'], 
                       width=3, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_title('Financial Innovation Regulatory Response Speed', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Days (Innovation to Regulation)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, innovation_speed['value']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                     f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/velocity_trends.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("✓ Created velocity trends analysis")

# ================================================================================
# 5. INNOVATION EXPORT SUCCESS NETWORK
# ================================================================================

def create_export_network():
    """Create network visualization of Argentine legal innovation exports."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel 1: Export success by legal area
    export_area = innovations_exported.groupby('legal_area')['success_level'].value_counts().unstack(fill_value=0)
    export_area_pct = export_area.div(export_area.sum(axis=1), axis=0) * 100
    
    export_area_pct.plot(kind='barh', stacked=True, ax=ax1, 
                        color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
    ax1.set_title('Export Success by Legal Area', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Percentage')
    ax1.legend(title='Success Level', bbox_to_anchor=(1.05, 1))
    
    # Panel 2: Regional influence over time
    innovations_exported['decade'] = (innovations_exported['origin_date'].dt.year // 10) * 10
    influence_time = innovations_exported.groupby(['decade', 'regional_influence']).size().unstack(fill_value=0)
    
    influence_time.plot(kind='area', stacked=True, ax=ax2, alpha=0.7)
    ax2.set_title('Regional Influence of Exports Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Decade')
    ax2.set_ylabel('Number of Innovations')
    ax2.legend(title='Regional Influence')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/export_network.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("✓ Created export success network analysis")

# ================================================================================
# 6. MUTATION PATTERNS ANALYSIS
# ================================================================================

def create_mutation_analysis():
    """Analyze mutation patterns in legal evolution."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel 1: Mutations by selection type
    mutation_selection = evolution_cases.groupby('tipo_seleccion')['mutaciones_identificadas'].mean()
    bars1 = ax1.bar(mutation_selection.index, mutation_selection.values, 
                    color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    ax1.set_title('Average Mutations by Selection Type', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Selection Type')
    ax1.set_ylabel('Average Mutations')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars1, mutation_selection.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 2: Mutations vs survival
    mutation_survival = evolution_cases.dropna(subset=['mutaciones_identificadas', 'supervivencia_anos'])
    ax2.scatter(mutation_survival['mutaciones_identificadas'], mutation_survival['supervivencia_anos'],
                alpha=0.6, s=60, c='purple')
    ax2.set_title('Mutations vs Survival Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Mutations')
    ax2.set_ylabel('Survival (Years)')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation
    if len(mutation_survival) > 2:
        correlation = mutation_survival['mutaciones_identificadas'].corr(mutation_survival['supervivencia_anos'])
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 3: Mutation distribution
    mutations_dist = evolution_cases['mutaciones_identificadas'].dropna()
    ax3.hist(mutations_dist, bins=range(0, int(mutations_dist.max()) + 2), 
             alpha=0.7, color='orange', edgecolor='black')
    ax3.set_title('Distribution of Mutation Counts', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Mutations')
    ax3.set_ylabel('Frequency')
    ax3.axvline(mutations_dist.mean(), color='red', linestyle='--', 
                label=f'Mean: {mutations_dist.mean():.1f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Success rate by mutation level
    evolution_cases['mutation_category'] = pd.cut(evolution_cases['mutaciones_identificadas'], 
                                                 bins=[-1, 0, 2, 5, float('inf')], 
                                                 labels=['None', 'Low (1-2)', 'Medium (3-5)', 'High (6+)'])
    
    mutation_success = pd.crosstab(evolution_cases['mutation_category'], 
                                  evolution_cases['exito'], normalize='index') * 100
    
    mutation_success.plot(kind='bar', ax=ax4, color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
    ax4.set_title('Success Rate by Mutation Level', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mutation Level')
    ax4.set_ylabel('Percentage')
    ax4.legend(title='Success Level')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/mutation_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("✓ Created mutation patterns analysis")

# ================================================================================
# 7. INTERACTIVE DASHBOARD (PLOTLY)
# ================================================================================

def create_interactive_dashboard():
    """Create interactive dashboard using Plotly."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Evolution Cases Timeline', 'Crisis Acceleration', 
                       'Transplant Success Rates', 'Export Influence'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Evolution timeline
    decade_counts = evolution_cases.groupby('decade').size()
    fig.add_trace(
        go.Scatter(x=decade_counts.index, y=decade_counts.values,
                  mode='lines+markers', name='Evolution Cases',
                  line=dict(width=3, color='blue')),
        row=1, col=1
    )
    
    # Crisis acceleration
    crisis_clean = crisis_periods.dropna(subset=['acceleration_factor', 'severity_level'])
    fig.add_trace(
        go.Scatter(x=crisis_clean['severity_level'], y=crisis_clean['acceleration_factor'],
                  mode='markers', name='Crisis Acceleration',
                  marker=dict(size=10, color='red')),
        row=1, col=2
    )
    
    # Transplant success
    success_counts = transplants_tracking['success_level'].value_counts()
    fig.add_trace(
        go.Bar(x=success_counts.index, y=success_counts.values,
               name='Transplant Success', marker_color='green'),
        row=2, col=1
    )
    
    # Export influence
    influence_counts = innovations_exported['regional_influence'].value_counts()
    fig.add_trace(
        go.Bar(x=influence_counts.index, y=influence_counts.values,
               name='Export Influence', marker_color='orange'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Argentine Legal Evolution Dataset - Interactive Dashboard",
        title_x=0.5,
        showlegend=False,
        height=800
    )
    
    # Save interactive dashboard
    fig.write_html(f'{OUTPUT_DIR}/interactive_dashboard.html')
    
    print("✓ Created interactive dashboard")

# ================================================================================
# 8. SUMMARY STATISTICS VISUALIZATION
# ================================================================================

def create_summary_statistics():
    """Create comprehensive summary statistics visualization."""
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    
    # Dataset overview
    dataset_sizes = [len(evolution_cases), len(transplants_tracking), 
                    len(crisis_periods), len(innovations_exported)]
    dataset_names = ['Evolution\nCases', 'Transplants', 'Crisis\nPeriods', 'Innovations\nExported']
    
    bars1 = ax1.bar(dataset_names, dataset_sizes, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    ax1.set_title('Dataset Overview', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Observations')
    
    for bar, val in zip(bars1, dataset_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # Success rates comparison
    evo_success = (evolution_cases['exito'] == 'Exitoso').mean() * 100
    transplant_success = (transplants_tracking['success_level'] == 'High').mean() * 100
    export_success = (innovations_exported['success_level'].isin(['High', 'Very High'])).mean() * 100
    
    success_rates = [evo_success, transplant_success, export_success]
    success_names = ['Endogenous\nEvolution', 'Legal\nTransplants', 'Innovation\nExports']
    
    bars2 = ax2.bar(success_names, success_rates, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
    ax2.set_title('Success Rates Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 100)
    
    for bar, val in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Temporal distribution
    evolution_cases['year'] = evolution_cases['fecha_inicio'].dt.year
    yearly_counts = evolution_cases['year'].value_counts().sort_index()
    
    ax3.plot(yearly_counts.index, yearly_counts.values, color='purple', linewidth=2)
    ax3.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='purple')
    ax3.set_title('Legal Innovations Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Number of Innovations')
    ax3.grid(True, alpha=0.3)
    
    # Legal areas distribution
    area_counts = evolution_cases['area_derecho'].value_counts().head(8)
    ax4.barh(range(len(area_counts)), area_counts.values, color='skyblue', alpha=0.7)
    ax4.set_yticks(range(len(area_counts)))
    ax4.set_yticklabels([area.replace('Derecho ', '') for area in area_counts.index])
    ax4.set_title('Cases by Legal Area (Top 8)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Cases')
    
    # Crisis impact
    crisis_severity_counts = crisis_periods['severity_level'].value_counts()
    wedges, texts, autotexts = ax5.pie(crisis_severity_counts.values, 
                                      labels=crisis_severity_counts.index,
                                      autopct='%1.1f%%', startangle=90,
                                      colors=['lightgreen', 'yellow', 'orange', 'red', 'darkred'])
    ax5.set_title('Crisis Severity Distribution', fontsize=12, fontweight='bold')
    
    # Survival analysis
    survival_data = evolution_cases['supervivencia_anos'].dropna()
    ax6.hist(survival_data, bins=15, alpha=0.7, color='teal', edgecolor='black')
    ax6.axvline(survival_data.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {survival_data.mean():.1f} years')
    ax6.axvline(survival_data.median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {survival_data.median():.1f} years')
    ax6.set_title('Legal Innovation Survival Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Survival (Years)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/summary_statistics.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("✓ Created summary statistics visualization")

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Execute all visualizations."""
    
    print("Creating Argentine Legal Evolution Dataset Visualizations...")
    print("=" * 60)
    
    # Create all visualizations
    create_evolution_timeline()
    create_crisis_acceleration_plot()
    create_transplant_analysis() 
    create_velocity_trends()
    create_export_network()
    create_mutation_analysis()
    create_interactive_dashboard()
    create_summary_statistics()
    
    print("\n" + "=" * 60)
    print("✓ All visualizations completed successfully!")
    print(f"✓ Output saved to '{OUTPUT_DIR}/' directory")
    print("\nGenerated files:")
    print("- evolution_timeline.png")
    print("- crisis_acceleration.png") 
    print("- transplant_analysis.png")
    print("- velocity_trends.png")
    print("- export_network.png")
    print("- mutation_analysis.png")
    print("- summary_statistics.png")
    print("- interactive_dashboard.html")
    print("\nVisualization suite ready for academic publication and presentation.")

if __name__ == "__main__":
    main()