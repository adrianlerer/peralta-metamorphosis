#!/usr/bin/env python3
"""
Crear visualización resumen del análisis Proyecto Gollan
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_summary_visualization():
    """Crear visualización resumen de resultados clave"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PROYECTO GOLLAN 2130-D-2025: ANÁLISIS INTEGRAL\nPolitical Similarity Framework (PSF)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Scores Principales
    ax1.set_title('EVALUACIÓN GENERAL', fontweight='bold', fontsize=14)
    
    categories = ['Constitucionalidad', 'Gobernanza', 'Innovación*', 'Score General']
    scores = [0.188, 0.213, 0.588, 0.330]  # Innovación invertida para visualización
    colors = ['#FF4444', '#FF4444', '#FF4444', '#FF4444']  # Todo rojo por ser problemático
    
    bars = ax1.bar(categories, scores, color=colors, alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Umbral Aceptable')
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Umbral Crítico')
    
    # Añadir valores
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Score (0-1)')
    ax1.legend()
    ax1.text(0.5, -0.15, '*Innovación: Score real -0.412 (invertido para visualización)', 
             ha='center', transform=ax1.transAxes, fontsize=10, style='italic')
    
    # 2. Violaciones Constitucionales
    ax2.set_title('VIOLACIONES ART. 19 CN', fontweight='bold', fontsize=14)
    
    violations = ['Principio\nLegalidad', 'Esfera\nPrivada', 'Carga\nPrueba', 'Exceso\nRegulatorio']
    violation_scores = [0.80, 0.70, 0.90, 0.85]
    
    colors_viol = plt.cm.Reds(np.array(violation_scores))
    bars2 = ax2.bar(violations, violation_scores, color=colors_viol)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Severidad Violación (0-1)')
    
    # Añadir valores
    for bar, score in zip(bars2, violation_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Dimensiones Políticas 4D
    ax3.set_title('ESPACIO POLÍTICO 4D', fontweight='bold', fontsize=14)
    
    dimensions = ['Centralización', 'Puerto-Interior', 'Elite-Popular', 'Continuidad']
    dim_values = [0.85, 0.75, 0.65, 0.45]
    
    # Gráfico radar simplificado en barras
    bars3 = ax3.bar(dimensions, dim_values, 
                   color=['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98'])
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Score (0-1)')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Añadir valores
    for bar, value in zip(bars3, dim_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Impacto en Sectores
    ax4.set_title('IMPACTO NEGATIVO POR SECTOR', fontweight='bold', fontsize=14)
    
    sectors = ['PyMEs', 'Startups', 'Grandes\nEmpresas', 'Investigación']
    impacts = [0.90, 0.95, 0.30, 0.70]
    
    colors_impact = ['#FF4444' if x > 0.7 else '#FFA500' if x > 0.4 else '#90EE90' for x in impacts]
    bars4 = ax4.bar(sectors, impacts, color=colors_impact)
    ax4.set_ylim(0, 1)
    ax4.set_ylabel('Impacto Negativo (0-1)')
    
    # Añadir valores
    for bar, impact in zip(bars4, impacts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{impact:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Añadir texto de veredicto
    fig.text(0.5, 0.02, 
             '🔴 VEREDICTO: INCONSTITUCIONAL Y CONTRAPRODUCENTE - RECHAZO INTEGRAL RECOMENDADO', 
             ha='center', va='bottom', fontsize=14, fontweight='bold', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2))
    
    return fig

def main():
    """Función principal"""
    print("📊 Creando visualización resumen Proyecto Gollan...")
    
    # Crear visualización
    fig = create_summary_visualization()
    
    # Guardar
    plt.savefig('/home/user/webapp/proyecto_gollan_resumen_visual.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Visualización guardada: proyecto_gollan_resumen_visual.png")
    
    # Mostrar estadísticas clave
    print("\n🎯 ESTADÍSTICAS CLAVE:")
    print("=" * 50)
    print(f"⚖️ Constitucionalidad: 0.188/1.000 (CRÍTICO)")
    print(f"🏛️ Gobernanza: 0.213/1.000 (AUTORITARIO)")  
    print(f"💡 Innovación: -0.412 (MUY NEGATIVO)")
    print(f"📊 Violación Art. 19: 4/4 principios violados")
    print(f"🔴 Recomendación: RECHAZO INTEGRAL")

if __name__ == "__main__":
    main()