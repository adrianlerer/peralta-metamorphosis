#!/usr/bin/env python3
"""
Script para renombrar el framework y eliminar referencias específicas
Cambio de López Rega-Milei Framework a Political Similarity Framework (PSF)
"""

import os
import re
import json

def rename_framework_in_file(file_path):
    """Renombrar referencias en un archivo específico"""
    
    # Mapeo de términos a cambiar
    replacements = {
        # Referencias directas
        r'López Rega': 'Actor Referencia A',
        r'Lopez Rega': 'Actor Referencia A', 
        r'lópez rega': 'actor referencia A',
        r'lopez rega': 'actor referencia A',
        r'LÓPEZ REGA': 'ACTOR REFERENCIA A',
        r'LOPEZ REGA': 'ACTOR REFERENCIA A',
        
        r'Milei': 'Actor Referencia B',
        r'MILEI': 'ACTOR REFERENCIA B',
        r'milei': 'actor referencia B',
        
        # Framework names
        r'López Rega-Milei similarity': 'Political Similarity Index (PSI)',
        r'Lopez Rega-Milei similarity': 'Political Similarity Index (PSI)',
        r'lópez rega-milei similarity': 'political similarity index (PSI)',
        r'lopez rega-milei similarity': 'political similarity index (PSI)',
        
        r'López Rega-Milei Framework': 'Political Similarity Framework (PSF)',
        r'Lopez Rega-Milei Framework': 'Political Similarity Framework (PSF)',
        
        # Columnas y variables
        r'lopez_rega_similarity': 'political_similarity_index',
        r'López_Rega_similarity': 'political_similarity_index',
        r'Lopez_Rega_similarity': 'political_similarity_index',
        
        # Descripciones técnicas
        r'López Rega-Milei Overall Similarity': 'Political Similarity Index (PSI) Score',
        r'Lopez Rega-Milei Overall Similarity': 'Political Similarity Index (PSI) Score',
        
        # Análisis específicos
        r'López Rega-Milei Multi-Dimensional Similarity': 'Multi-Dimensional Political Similarity Analysis',
        r'Lopez Rega-Milei Multi-Dimensional Similarity': 'Multi-Dimensional Political Similarity Analysis',
        
        # Referencias en texto
        r'Most Similar to López Rega': 'Highest Political Similarity Score',
        r'Most Similar to Lopez Rega': 'Highest Political Similarity Score',
        r'Similar to López Rega': 'High Political Similarity',
        r'Similar to Lopez Rega': 'High Political Similarity',
        
        # Nombres específicos en contexto
        r'José López Rega': 'Actor Histórico A',
        r'Jose Lopez Rega': 'Actor Histórico A',
        r'Javier Milei': 'Actor Contemporáneo B',
        
        # Referencias académicas
        r'López Rega-Milei comparison framework': 'Political Actor Comparison Framework',
        r'Lopez Rega-Milei comparison framework': 'Political Actor Comparison Framework',
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Aplicar reemplazos
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # Solo escribir si hubo cambios
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def update_political_dataset():
    """Actualizar el dataset político para usar nombres genéricos"""
    
    # Crear versión actualizada del dataset
    # Framework actualizado - no necesitamos crear código aquí
    print("Framework dataset will be updated separately")
    
    return updated_dataset_code

def main():
    """Función principal para renombrar el framework"""
    
    print("🔄 RENOMBRANDO FRAMEWORK POLÍTICO")
    print("=" * 50)
    print("Cambios principales:")
    print("• López Rega → Actor Histórico A / Actor Referencia A")  
    print("• Milei → Actor Contemporáneo B / Actor Referencia B")
    print("• López Rega-Milei Framework → Political Similarity Framework (PSF)")
    print("• lopez_rega_similarity → political_similarity_index")
    print("=" * 50)
    
    # Archivos a procesar
    files_to_process = [
        '/home/user/webapp/expanded_corpus_comprehensive_analysis.py',
        '/home/user/webapp/proyecto_gollan_political_analysis.py',
        '/home/user/webapp/ANÁLISIS_PROYECTO_GOLLAN_IA_2130-D-2025.md',
        '/home/user/webapp/EXPANDED_CORPUS_ANALYSIS_SUMMARY.md',
        '/home/user/webapp/COMPREHENSIVE_ANALYSIS_SUMMARY.md',
        '/home/user/webapp/Paper11_CORRECTED_COMPLETE.md'
    ]
    
    # Archivos JSON a procesar
    json_files = [
        '/home/user/webapp/comprehensive_expanded_corpus_analysis_results.json',
        '/home/user/webapp/proyecto_gollan_analisis_politico_completo.json'
    ]
    
    files_changed = 0
    
    # Procesar archivos de texto
    for file_path in files_to_process:
        if os.path.exists(file_path):
            print(f"📝 Processing: {os.path.basename(file_path)}")
            if rename_framework_in_file(file_path):
                files_changed += 1
                print(f"   ✅ Updated")
            else:
                print(f"   ➡️ No changes needed")
        else:
            print(f"   ❌ File not found: {file_path}")
    
    # Procesar archivos JSON
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"📊 Processing JSON: {os.path.basename(json_file)}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = f.read()
                
                # Aplicar reemplazos a JSON como texto
                original_data = data
                
                replacements = {
                    'López Rega': 'Actor Referencia A',
                    'Lopez Rega': 'Actor Referencia A',
                    'Milei': 'Actor Referencia B',
                    'lopez_rega_similarity': 'political_similarity_index',
                    'López Rega-Milei': 'Political Similarity Framework',
                    'Lopez Rega-Milei': 'Political Similarity Framework'
                }
                
                for old, new in replacements.items():
                    data = data.replace(old, new)
                
                if data != original_data:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        f.write(data)
                    files_changed += 1
                    print(f"   ✅ JSON Updated")
                else:
                    print(f"   ➡️ JSON No changes needed")
                    
            except Exception as e:
                print(f"   ❌ Error processing JSON {json_file}: {e}")
    
    print(f"\n🏆 PROCESO COMPLETADO")
    print(f"📊 Archivos modificados: {files_changed}")
    print(f"🎯 Framework renombrado exitosamente")
    print(f"\n📋 NUEVO FRAMEWORK:")
    print(f"   • Political Similarity Framework (PSF)")
    print(f"   • Political Similarity Index (PSI)")  
    print(f"   • Multi-Dimensional Political Analysis")
    print(f"   • Generic Reference Actors A & B")

if __name__ == "__main__":
    main()
'''