#!/usr/bin/env python3
"""
Quick Enhanced Universal Framework v3.0 - Presupuesto Uruguay 2025-2029
Análisis optimizado para PDF grande con Reality Filter
"""

import pdfplumber
import json
import re
from datetime import datetime
from pathlib import Path

def quick_extract_key_content(pdf_path: str, max_pages: int = 20) -> dict:
    """Extracción rápida de contenido clave de las primeras páginas"""
    content = {
        'key_articles': [],
        'budget_figures': [],
        'sections': [],
        'sample_text': '',
        'total_pages': 0,
        'processed_pages': 0
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            content['total_pages'] = len(pdf.pages)
            pages_to_process = min(max_pages, len(pdf.pages))
            
            for i in range(pages_to_process):
                page = pdf.pages[i]
                page_text = page.extract_text()
                
                if page_text and i < 5:  # Solo guardar texto de primeras 5 páginas
                    content['sample_text'] += page_text + '\n'
                
                if page_text:
                    # Artículos clave
                    articles = re.findall(r'Artículo\s+\d+[.-].*', page_text, re.IGNORECASE)
                    content['key_articles'].extend(articles[:3])  # Solo primeros 3 por página
                    
                    # Cifras importantes
                    figures = re.findall(r'[UY]?\$?\s*[\d,]+\.?\d*\s*(?:millones?|mil\s+millones?)', page_text)
                    content['budget_figures'].extend(figures[:5])  # Solo primeras 5 por página
                    
                    # Secciones
                    sections = re.findall(r'SECCIÓN\s+[IVX]+|CAPÍTULO\s+\d+|TÍTULO\s+[IVX]+', page_text, re.IGNORECASE)
                    content['sections'].extend(sections)
                
                content['processed_pages'] = i + 1
                
                # Progreso cada 5 páginas
                if (i + 1) % 5 == 0:
                    print(f"   Procesadas {i+1}/{pages_to_process} páginas...")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return content

def analyze_uruguay_budget_quick():
    """Análisis rápido del presupuesto uruguayo"""
    
    print("🇺🇾 ENHANCED UNIVERSAL FRAMEWORK v3.0 - PRESUPUESTO URUGUAY 2025-2029")
    print("📊 ANÁLISIS RÁPIDO CON REALITY FILTER")
    print("=" * 70)
    
    pdf_file = Path("/home/user/webapp/Presupuesto_Uruguay_2025-2029.pdf")
    
    if not pdf_file.exists():
        print(f"❌ No se encontró: {pdf_file}")
        return
    
    print(f"📁 Archivo: {pdf_file.name}")
    print(f"📊 Tamaño: {pdf_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Extracción rápida
    print("\n🔍 Extrayendo contenido clave...")
    content = quick_extract_key_content(str(pdf_file), max_pages=15)
    
    print(f"✅ Procesadas: {content['processed_pages']}/{content['total_pages']} páginas")
    print(f"📋 Artículos detectados: {len(content['key_articles'])}")
    print(f"💰 Cifras detectadas: {len(content['budget_figures'])}")
    print(f"📖 Secciones detectadas: {len(content['sections'])}")
    
    # Análisis con Reality Filter
    print("\n🎯 APLICANDO REALITY FILTER:")
    
    text = content['sample_text'].lower()
    
    # Variables críticas detectadas
    variables_criticas = []
    
    # 1. Período plurianual
    years_found = re.findall(r'202[5-9]', content['sample_text'])
    if len(set(years_found)) >= 4:
        variables_criticas.append("✅ PERÍODO PLURIANUAL: 2025-2029 confirmado")
    else:
        variables_criticas.append("⚠️ PERÍODO PLURIANUAL: Verificar cobertura años")
    
    # 2. Ajuste inflación
    if any(term in text for term in ['inflación', 'ajuste', 'índice', 'deflactor']):
        variables_criticas.append("✅ AJUSTE INFLACIÓN: Mecanismo detectado")
    else:
        variables_criticas.append("🚨 AJUSTE INFLACIÓN: NO DETECTADO - CRÍTICO")
    
    # 3. Base temporal
    if '2025' in text and ('enero' in text or 'base' in text):
        variables_criticas.append("✅ BASE TEMPORAL: Enero 2025 identificada")
    else:
        variables_criticas.append("⚠️ BASE TEMPORAL: Clarificar fecha base")
    
    # 4. Cifras significativas
    if len(content['budget_figures']) > 5:
        variables_criticas.append("✅ CIFRAS PRESUPUESTARIAS: Múltiples montos detectados")
    else:
        variables_criticas.append("⚠️ CIFRAS PRESUPUESTARIAS: Pocas cifras detectadas")
    
    # 5. Marco institucional
    institutions = ['ministerio', 'mef', 'opp', 'bcu', 'presidencia']
    inst_found = sum(1 for inst in institutions if inst in text)
    if inst_found >= 2:
        variables_criticas.append("✅ MARCO INSTITUCIONAL: Instituciones identificadas")
    else:
        variables_criticas.append("⚠️ MARCO INSTITUCIONAL: Verificar instituciones")
    
    # Mostrar análisis
    print()
    for variable in variables_criticas:
        print(f"   {variable}")
    
    # Reality Filter Warnings
    print("\n🚨 REALITY FILTER - ADVERTENCIAS CRÍTICAS:")
    
    reality_warnings = [
        "🔴 INFLACIÓN ACUMULADA: Calcular impacto 5 años (2025-2029)",
        "🔴 INCERTIDUMBRE TEMPORAL: Proyecciones años 2027-2029 alta incertidumbre",  
        "🔴 CONTEXTO URUGUAYO: Considerar dependencia commodities y vecinos",
        "🔴 SOSTENIBILIDAD FISCAL: Analizar ratio deuda/PIB realista",
        "🔴 CAPACIDAD EJECUCIÓN: Evaluar viabilidad implementación quinquenal"
    ]
    
    for warning in reality_warnings:
        print(f"   {warning}")
    
    # Recomendaciones específicas
    print("\n📋 RECOMENDACIONES ENHANCED FRAMEWORK v3.0:")
    
    recommendations = [
        "",
        "1. AJUSTE INFLACIÓN OBLIGATORIO:",
        "   • Usar deflactor BCU (meta 3-7% anual)",
        "   • Calcular inflación acumulada 2025-2029", 
        "   • Expresar todo en pesos constantes enero 2025",
        "",
        "2. ANÁLISIS TEMPORAL DINÁMICO:",
        "   • Años 2025-2026: Proyecciones más confiables",
        "   • Años 2027-2029: Incluir márgenes incertidumbre",
        "   • Comparar con series históricas uruguayas",
        "",
        "3. CONTEXTO ECONÓMICO ESPECÍFICO:",
        "   • Precio soja/carne (exports clave Uruguay)", 
        "   • Crecimiento Argentina/Brasil (socios comerciales)",
        "   • Tipo cambio real (competitividad)",
        "   • Restricciones balance pagos",
        "",
        "4. VARIABLES CRÍTICAS MONITOREO:",
        "   • Gasto público como % PIB",
        "   • Inversión pública sostenibilidad",
        "   • Déficit/superávit fiscal realista", 
        "   • Deuda pública trayectoria"
    ]
    
    for rec in recommendations:
        print(rec)
    
    # Evaluación general
    has_inflation = any('inflación' in var.lower() for var in variables_criticas)
    has_period = '2025-2029' in str(content)
    
    print(f"\n🏆 EVALUACIÓN GENERAL:")
    if not has_inflation:
        assessment = "🚨 CRÍTICO: Sin mecanismo ajuste inflación - Análisis temporal inválido"
    elif len(content['budget_figures']) < 3:
        assessment = "⚠️ LIMITADO: Pocas cifras detectadas - Requiere análisis más profundo"
    else:
        assessment = "✅ ANALIZABLE: Documento viable para análisis con Reality Filter"
    
    print(f"   {assessment}")
    
    # Guardar resumen
    summary = {
        'document': 'Presupuesto Nacional Uruguay 2025-2029',
        'analysis_date': datetime.now().isoformat(),
        'pages_processed': content['processed_pages'],
        'total_pages': content['total_pages'],
        'variables_detected': variables_criticas,
        'reality_warnings': reality_warnings,
        'assessment': assessment,
        'key_figures_sample': content['budget_figures'][:10],
        'key_articles_sample': content['key_articles'][:5]
    }
    
    with open('/home/user/webapp/uruguay_budget_quick_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Análisis guardado en: uruguay_budget_quick_analysis.json")
    
    # Guardar muestra de contenido
    with open('/home/user/webapp/uruguay_budget_sample_content.txt', 'w', encoding='utf-8') as f:
        f.write("# PRESUPUESTO NACIONAL URUGUAY 2025-2029 - MUESTRA CONTENIDO\n\n")
        f.write(f"Páginas analizadas: {content['processed_pages']}/{content['total_pages']}\n\n")
        f.write("## ARTÍCULOS DETECTADOS:\n")
        for art in content['key_articles'][:10]:
            f.write(f"- {art}\n")
        f.write("\n## CIFRAS DETECTADAS:\n")
        for fig in content['budget_figures'][:15]:
            f.write(f"- {fig}\n")
        f.write(f"\n## MUESTRA TEXTO (Primeras páginas):\n\n")
        f.write(content['sample_text'][:3000])
        f.write("\n\n[... contenido truncado para análisis rápido ...]")
    
    print(f"📄 Muestra contenido guardada en: uruguay_budget_sample_content.txt")
    
    return summary

if __name__ == "__main__":
    analyze_uruguay_budget_quick()