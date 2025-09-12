#!/usr/bin/env python3
"""
Quick script to rename framework references
"""

import os
import re

def rename_in_file(file_path):
    """Renombrar referencias en archivo"""
    
    replacements = {
        'López Rega': 'Actor Referencia A',
        'Lopez Rega': 'Actor Referencia A', 
        'lópez rega': 'actor referencia A',
        'lopez rega': 'actor referencia A',
        'LÓPEZ REGA': 'ACTOR REFERENCIA A',
        'LOPEZ REGA': 'ACTOR REFERENCIA A',
        'Milei': 'Actor Referencia B',
        'MILEI': 'ACTOR REFERENCIA B',
        'milei': 'actor referencia B',
        'López Rega-Milei': 'Political Similarity Framework',
        'Lopez Rega-Milei': 'Political Similarity Framework',
        'lópez rega-milei': 'political similarity framework',
        'lopez rega-milei': 'political similarity framework',
        'lopez_rega_similarity': 'political_similarity_index',
        'López_Rega_similarity': 'political_similarity_index',
        'Lopez_Rega_similarity': 'political_similarity_index',
        'José López Rega': 'Actor Histórico A',
        'Jose Lopez Rega': 'Actor Histórico A',
        'Javier Milei': 'Actor Contemporáneo B',
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

# Archivos a procesar
files = [
    '/home/user/webapp/expanded_corpus_comprehensive_analysis.py',
    '/home/user/webapp/proyecto_gollan_political_analysis.py',
    '/home/user/webapp/ANÁLISIS_PROYECTO_GOLLAN_IA_2130-D-2025.md',
    '/home/user/webapp/EXPANDED_CORPUS_ANALYSIS_SUMMARY.md',
    '/home/user/webapp/COMPREHENSIVE_ANALYSIS_SUMMARY.md',
    '/home/user/webapp/comprehensive_expanded_corpus_analysis_results.json',
    '/home/user/webapp/proyecto_gollan_analisis_politico_completo.json'
]

print("🔄 Renombrando referencias...")
changed = 0

for file_path in files:
    if os.path.exists(file_path):
        if rename_in_file(file_path):
            print(f"✅ {os.path.basename(file_path)}")
            changed += 1
        else:
            print(f"➡️ {os.path.basename(file_path)} (no changes)")
    else:
        print(f"❌ {os.path.basename(file_path)} (not found)")

print(f"\n🏆 Completado: {changed} archivos modificados")
print("📋 Framework renombrado a: Political Similarity Framework (PSF)")