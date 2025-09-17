#!/usr/bin/env python3
"""
Enhanced Universal Framework v3.0 - Analizador Reformas Tributarias Uruguay 2025-2029
Reality Filter aplicado a política tributaria específica
"""

import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Any
import json

class UruguayanTaxReformAnalyzer:
    """Analizador especializado en reformas tributarias Uruguay"""
    
    def __init__(self):
        self.tax_keywords = [
            'impuesto', 'tributo', 'tasa', 'contribución', 'arancel',
            'IVA', 'IRAE', 'IRPF', 'IP', 'IMEBA', 'IMESI',
            'alícuota', 'tarifa', 'exoneración', 'exención', 
            'gravamen', 'deducción', 'crédito fiscal',
            'base imponible', 'hecho generador'
        ]
        
        self.tax_changes_patterns = [
            r'sustitú[iy]ese.*art[íi]culo.*\d+.*ley.*\d+.*impuesto',
            r'modif[íi]case.*art[íi]culo.*\d+.*tributo',
            r'cr[ée]ase.*impuesto.*\w+',
            r'establ[ée]cese.*tasa.*\d+',
            r'al[íi]cuota.*\d+.*por.*ciento',
            r'exon[ée]rase.*impuesto',
            r'grav[áa]rase.*con.*tasa'
        ]
        
        self.uruguayan_taxes = {
            'IVA': 'Impuesto al Valor Agregado',
            'IRAE': 'Impuesto a las Rentas de Actividades Económicas', 
            'IRPF': 'Impuesto a las Rentas de Personas Físicas',
            'IP': 'Impuesto al Patrimonio',
            'IMEBA': 'Impuesto a los Bienes de Activo Fijo',
            'IMESI': 'Impuesto Específico Interno',
            'IAE': 'Impuesto a los Actos Económicos',
            'COFIS': 'Contribución para el Financiamiento de la Seguridad Social'
        }
    
    def extract_tax_sections(self, pdf_path: str, max_pages: int = 50) -> Dict[str, Any]:
        """Extrae secciones relacionadas con impuestos"""
        tax_content = {
            'tax_articles': [],
            'tax_modifications': [],
            'new_taxes': [],
            'rate_changes': [],
            'exemptions': [],
            'total_pages_scanned': 0,
            'tax_pages': []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_scan = min(max_pages, len(pdf.pages))
                
                for i in range(pages_to_scan):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    
                    if not text:
                        continue
                    
                    text_lower = text.lower()
                    
                    # Verificar si la página contiene contenido tributario
                    has_tax_content = any(keyword in text_lower for keyword in self.tax_keywords)
                    
                    if has_tax_content:
                        tax_content['tax_pages'].append({
                            'page': i + 1,
                            'text': text[:2000]  # Primeros 2000 caracteres
                        })
                        
                        # Buscar artículos específicos sobre impuestos
                        articles = re.findall(r'Art[íi]culo\s+\d+[.-].*?(?=Art[íi]culo|\Z)', text, re.DOTALL | re.IGNORECASE)
                        for article in articles:
                            if any(keyword in article.lower() for keyword in self.tax_keywords):
                                tax_content['tax_articles'].append({
                                    'page': i + 1,
                                    'content': article[:500]  # Primeros 500 caracteres
                                })
                        
                        # Buscar modificaciones tributarias
                        for pattern in self.tax_changes_patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            if matches:
                                tax_content['tax_modifications'].extend([{
                                    'page': i + 1,
                                    'type': 'modification',
                                    'content': match
                                } for match in matches])
                        
                        # Buscar nuevos impuestos
                        new_tax_patterns = [
                            r'cr[ée]ase.*(?:impuesto|tributo|tasa).*?(?:\.|;)',
                            r'establ[ée]cese.*(?:impuesto|tributo|tasa).*?(?:\.|;)'
                        ]
                        for pattern in new_tax_patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            if matches:
                                tax_content['new_taxes'].extend([{
                                    'page': i + 1,
                                    'content': match
                                } for match in matches])
                        
                        # Buscar cambios de alícuotas
                        rate_patterns = [
                            r'al[íi]cuota.*?(?:\d+(?:\.\d+)?)\s*(?:por\s*ciento|%)',
                            r'tasa.*?(?:\d+(?:\.\d+)?)\s*(?:por\s*ciento|%)'
                        ]
                        for pattern in rate_patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            if matches:
                                tax_content['rate_changes'].extend([{
                                    'page': i + 1,
                                    'content': match
                                } for match in matches])
                        
                        # Buscar exoneraciones
                        exemption_patterns = [
                            r'exon[ée]r[ae]se.*?(?:impuesto|tributo).*?(?:\.|;)',
                            r'exent[oó].*?(?:impuesto|tributo).*?(?:\.|;)'
                        ]
                        for pattern in exemption_patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            if matches:
                                tax_content['exemptions'].extend([{
                                    'page': i + 1,
                                    'content': match
                                } for match in matches])
                    
                    tax_content['total_pages_scanned'] = i + 1
                    
                    # Progreso cada 10 páginas
                    if (i + 1) % 10 == 0:
                        print(f"   Escaneadas {i+1}/{pages_to_scan} páginas...")
        
        except Exception as e:
            print(f"Error escaneando impuestos: {e}")
        
        return tax_content
    
    def analyze_tax_impact(self, tax_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el impacto de las reformas tributarias detectadas"""
        
        impact_analysis = {
            'summary': {
                'tax_articles_found': len(tax_content['tax_articles']),
                'modifications_found': len(tax_content['tax_modifications']),
                'new_taxes_found': len(tax_content['new_taxes']),
                'rate_changes_found': len(tax_content['rate_changes']),
                'exemptions_found': len(tax_content['exemptions']),
                'tax_pages_found': len(tax_content['tax_pages'])
            },
            'impact_assessment': {
                'revenue_impact': 'PENDING_DETAILED_ANALYSIS',
                'economic_impact': 'REQUIRES_QUANTIFICATION',
                'social_impact': 'NEEDS_EVALUATION',
                'competitiveness_impact': 'TO_BE_MEASURED'
            },
            'critical_changes': [],
            'reality_filter_warnings': []
        }
        
        # Análisis específico por tipo de cambio
        if tax_content['new_taxes']:
            impact_analysis['critical_changes'].append({
                'type': 'NEW_TAXES',
                'count': len(tax_content['new_taxes']),
                'impact': 'POTENTIAL_REVENUE_INCREASE',
                'concern': 'BUSINESS_BURDEN_INCREASE'
            })
            impact_analysis['reality_filter_warnings'].append(
                "NUEVO IMPUESTO DETECTADO: Evaluar impacto competitividad económica"
            )
        
        if tax_content['rate_changes']:
            impact_analysis['critical_changes'].append({
                'type': 'RATE_CHANGES', 
                'count': len(tax_content['rate_changes']),
                'impact': 'REVENUE_MODIFICATION',
                'concern': 'TAX_BURDEN_CHANGE'
            })
            impact_analysis['reality_filter_warnings'].append(
                "CAMBIOS ALÍCUOTAS: Verificar impacto inflación y poder adquisitivo"
            )
        
        if tax_content['exemptions']:
            impact_analysis['critical_changes'].append({
                'type': 'TAX_EXEMPTIONS',
                'count': len(tax_content['exemptions']),
                'impact': 'POTENTIAL_REVENUE_DECREASE',
                'benefit': 'SECTOR_STIMULUS'
            })
        
        # Reality Filter específico Uruguay
        impact_analysis['reality_filter_warnings'].extend([
            "CONTEXTO URUGUAYO: Verificar impacto en competitividad regional",
            "PRESIÓN FISCAL: Evaluar carga tributaria total vs. capacidad contributiva",
            "EVASIÓN FISCAL: Considerar impacto en informalidad económica",
            "INVERSIÓN EXTRANJERA: Analizar efectos en atracción de capitales"
        ])
        
        return impact_analysis

def main():
    """Ejecuta análisis completo reformas tributarias Uruguay"""
    
    print("🇺🇾 ANALIZADOR REFORMAS TRIBUTARIAS URUGUAY 2025-2029")
    print("📊 Enhanced Universal Framework v3.0 - Reality Filter Tributario")
    print("=" * 70)
    
    pdf_file = Path("/home/user/webapp/Presupuesto_Uruguay_2025-2029.pdf")
    
    if not pdf_file.exists():
        print(f"❌ No se encontró: {pdf_file}")
        return
    
    print(f"📁 Analizando: {pdf_file.name}")
    
    # Análisis tributario
    analyzer = UruguayanTaxReformAnalyzer()
    
    print("\n🔍 Escaneando reformas tributarias...")
    tax_content = analyzer.extract_tax_sections(str(pdf_file), max_pages=100)
    
    print(f"\n📊 RESULTADOS ESCANEO TRIBUTARIO:")
    print(f"   📄 Páginas escaneadas: {tax_content['total_pages_scanned']}")
    print(f"   📋 Artículos tributarios: {len(tax_content['tax_articles'])}")
    print(f"   🔄 Modificaciones detectadas: {len(tax_content['tax_modifications'])}")
    print(f"   🆕 Nuevos impuestos: {len(tax_content['new_taxes'])}")
    print(f"   📈 Cambios alícuotas: {len(tax_content['rate_changes'])}")
    print(f"   ✅ Exoneraciones: {len(tax_content['exemptions'])}")
    print(f"   📑 Páginas con contenido tributario: {len(tax_content['tax_pages'])}")
    
    # Análisis de impacto
    print("\n🎯 ANALIZANDO IMPACTO REFORMAS...")
    impact_analysis = analyzer.analyze_tax_impact(tax_content)
    
    print(f"\n🏆 EVALUACIÓN IMPACTO TRIBUTARIO:")
    summary = impact_analysis['summary']
    if summary['new_taxes_found'] > 0:
        print(f"   🆕 NUEVOS IMPUESTOS: {summary['new_taxes_found']} detectados")
    if summary['rate_changes_found'] > 0:
        print(f"   📈 CAMBIOS ALÍCUOTAS: {summary['rate_changes_found']} detectados")
    if summary['modifications_found'] > 0:
        print(f"   🔄 MODIFICACIONES: {summary['modifications_found']} detectadas")
    if summary['exemptions_found'] > 0:
        print(f"   ✅ EXONERACIONES: {summary['exemptions_found']} detectadas")
    
    # Mostrar cambios críticos
    if impact_analysis['critical_changes']:
        print(f"\n🚨 CAMBIOS CRÍTICOS IDENTIFICADOS:")
        for change in impact_analysis['critical_changes']:
            print(f"   🔴 {change['type']}: {change['count']} casos")
            print(f"      Impacto: {change['impact']}")
            if 'concern' in change:
                print(f"      Preocupación: {change['concern']}")
    
    # Reality Filter warnings
    if impact_analysis['reality_filter_warnings']:
        print(f"\n⚠️ REALITY FILTER - ADVERTENCIAS:")
        for warning in impact_analysis['reality_filter_warnings']:
            print(f"   ⚠️ {warning}")
    
    # Guardar resultados
    results = {
        'analysis_timestamp': '2024-09-16T23:59:00',
        'document': 'Presupuesto Uruguay 2025-2029',
        'tax_content_found': tax_content,
        'impact_analysis': impact_analysis
    }
    
    output_file = '/home/user/webapp/uruguay_tax_reform_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Análisis guardado en: {output_file}")
    
    # Guardar contenido tributario para revisión
    if tax_content['tax_pages']:
        content_file = '/home/user/webapp/uruguay_tax_content_extracted.txt'
        with open(content_file, 'w', encoding='utf-8') as f:
            f.write("# CONTENIDO TRIBUTARIO EXTRAÍDO - PRESUPUESTO URUGUAY 2025-2029\n\n")
            
            if tax_content['tax_articles']:
                f.write("## ARTÍCULOS TRIBUTARIOS DETECTADOS:\n\n")
                for i, article in enumerate(tax_content['tax_articles'][:10], 1):
                    f.write(f"### Artículo {i} (Página {article['page']}):\n")
                    f.write(f"{article['content']}\n\n")
            
            if tax_content['new_taxes']:
                f.write("## NUEVOS IMPUESTOS DETECTADOS:\n\n")
                for i, tax in enumerate(tax_content['new_taxes'], 1):
                    f.write(f"{i}. (Página {tax['page']}) {tax['content']}\n")
            
            if tax_content['rate_changes']:
                f.write("\n## CAMBIOS DE ALÍCUOTAS DETECTADOS:\n\n")
                for i, change in enumerate(tax_content['rate_changes'], 1):
                    f.write(f"{i}. (Página {change['page']}) {change['content']}\n")
            
            if tax_content['exemptions']:
                f.write("\n## EXONERACIONES DETECTADAS:\n\n")
                for i, exemption in enumerate(tax_content['exemptions'], 1):
                    f.write(f"{i}. (Página {exemption['page']}) {exemption['content']}\n")
        
        print(f"📄 Contenido tributario guardado en: {content_file}")
    
    # Recomendación final
    total_changes = (len(tax_content['new_taxes']) + 
                    len(tax_content['rate_changes']) + 
                    len(tax_content['modifications']))
    
    print(f"\n🎯 EVALUACIÓN GENERAL:")
    if total_changes == 0:
        print("   ℹ️ NO SE DETECTARON REFORMAS TRIBUTARIAS SIGNIFICATIVAS")
        print("   📝 Nota: Puede requerir análisis más profundo de Tomos específicos")
    elif total_changes < 5:
        print("   ✅ REFORMAS TRIBUTARIAS MENORES DETECTADAS")
        print("   📊 Requiere cuantificación impacto específico")
    else:
        print("   🚨 REFORMAS TRIBUTARIAS SIGNIFICATIVAS DETECTADAS")
        print("   ⚠️ ANÁLISIS IMPACTO ECONÓMICO CRÍTICO REQUERIDO")

if __name__ == "__main__":
    main()