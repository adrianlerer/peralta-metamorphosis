#!/usr/bin/env python3
"""
Enhanced Universal Framework v3.0 - Analizador Presupuesto Nacional Uruguay 2025-2029
Reality Filter aplicado con análisis dinámico de variables presupuestarias
"""

import pdfplumber
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sys
from pathlib import Path

class CriticalityLevel(Enum):
    BLOCKING = "BLOCKING"
    CRITICAL = "CRITICAL" 
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

class VariableType(Enum):
    TEMPORAL = "TEMPORAL"
    MONETARY = "MONETARY"
    INFLATION = "INFLATION"
    POLICY = "POLICY"
    INSTITUTIONAL = "INSTITUTIONAL"
    FISCAL = "FISCAL"
    PLURIANNUAL = "PLURIANNUAL"
    INTERNATIONAL = "INTERNATIONAL"

@dataclass
class BudgetVariable:
    name: str
    type: VariableType
    criticality: CriticalityLevel
    description: str
    current_value: Any
    requires_adjustment: bool
    inflation_impact: bool
    temporal_analysis_required: bool
    reality_filter_notes: str

@dataclass
class UruguayanBudgetAnalysis:
    document_id: str
    analysis_timestamp: str
    budget_period: str
    total_pages: int
    variables_detected: List[BudgetVariable]
    inflation_adjustments_required: List[str]
    temporal_inconsistencies: List[str]
    reality_filter_warnings: List[str]
    key_figures: Dict[str, Any]
    pluriannual_analysis: Dict[str, Any]
    recommendations: List[str]
    overall_assessment: str

class EnhancedUruguayanBudgetAnalyzer:
    """Analizador especializado para Presupuesto Nacional Uruguay 2025-2029"""
    
    def __init__(self):
        self.uruguayan_context = {
            'currency': 'UYU (Peso Uruguayo)',
            'inflation_target': 'BCU Target Range 3-7%',
            'fiscal_year': 'Calendario (Enero-Diciembre)',
            'budget_period': '2025-2029 (Plurianual)',
            'government_term': 'Período constitucional completo'
        }
        
        self.reality_filter_checks = {
            'inflation_adjustment': 'Obligatorio para comparaciones temporales',
            'currency_stability': 'Verificar impacto devaluación/apreciación',
            'debt_sustainability': 'Análisis ratios deuda/PIB realistic',
            'growth_projections': 'Validar con tendencias históricas',
            'external_factors': 'Considerar contexto internacional'
        }
    
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extrae contenido completo del PDF del presupuesto"""
        content = {
            'pages': [],
            'total_pages': 0,
            'budget_figures': [],
            'articles': [],
            'sections': [],
            'tables_detected': 0,
            'full_text': ''
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                content['total_pages'] = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        content['pages'].append({
                            'number': i + 1,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
                        content['full_text'] += page_text + '\n'
                        
                        # Detectar artículos
                        articles = re.findall(r'Artículo\s+\d+[.-].*', page_text, re.IGNORECASE)
                        content['articles'].extend(articles)
                        
                        # Detectar cifras presupuestarias
                        figures = re.findall(r'[UY]?\$?\s*[\d,]+\.?\d*\s*(?:millones?|mil\s+millones?)?', page_text)
                        content['budget_figures'].extend(figures)
                        
                        # Detectar secciones
                        sections = re.findall(r'SECCIÓN\s+[IVX]+|CAPÍTULO\s+\d+|TÍTULO\s+[IVX]+', page_text, re.IGNORECASE)
                        content['sections'].extend(sections)
                    
                    # Contar tablas
                    tables = page.extract_tables()
                    if tables:
                        content['tables_detected'] += len(tables)
        
        except Exception as e:
            print(f"Error extrayendo PDF: {str(e)}")
            return content
        
        print(f"✅ PDF procesado: {content['total_pages']} páginas, {len(content['articles'])} artículos, {content['tables_detected']} tablas")
        return content
    
    def scan_temporal_variables(self, content: Dict[str, Any]) -> List[BudgetVariable]:
        """Escanea variables temporales críticas para análisis plurianual"""
        variables = []
        text = content['full_text']
        
        # Variable período plurianual
        period_mentions = re.findall(r'202[5-9]', text)
        years_covered = len(set(period_mentions))
        
        variables.append(BudgetVariable(
            name="Período Presupuestario Plurianual",
            type=VariableType.PLURIANNUAL,
            criticality=CriticalityLevel.HIGH,
            description="Presupuesto quinquenal 2025-2029 para período gobierno completo",
            current_value=f"Años detectados: {years_covered}, menciones: {len(period_mentions)}",
            requires_adjustment=False,
            inflation_impact=True,
            temporal_analysis_required=True,
            reality_filter_notes="CRÍTICO: Proyecciones 5 años requieren ajuste inflación anual obligatorio"
        ))
        
        # Variable base de cálculo
        base_date_mentions = re.findall(r'1º?\s*de\s*enero\s*de\s*2025|valores?\s*de\s*2025', text, re.IGNORECASE)
        variables.append(BudgetVariable(
            name="Base Temporal Cálculos",
            type=VariableType.TEMPORAL,
            criticality=CriticalityLevel.CRITICAL if len(base_date_mentions) == 0 else CriticalityLevel.MEDIUM,
            description="Fecha base para valores presupuestarios y ajustes posteriores",
            current_value=f"Menciones fecha base: {len(base_date_mentions)}",
            requires_adjustment=len(base_date_mentions) == 0,
            inflation_impact=True,
            temporal_analysis_required=True,
            reality_filter_notes="Valores deben estar expresados en moneda constante para análisis válido"
        ))
        
        return variables
    
    def scan_monetary_variables(self, content: Dict[str, Any]) -> List[BudgetVariable]:
        """Escanea variables monetarias y fiscales críticas"""
        variables = []
        text = content['full_text']
        
        # Cifras presupuestarias detectadas
        budget_figures = content['budget_figures']
        large_figures = [f for f in budget_figures if any(keyword in f.lower() for keyword in ['millones', 'mil millones'])]
        
        variables.append(BudgetVariable(
            name="Cifras Presupuestarias Principales",
            type=VariableType.MONETARY,
            criticality=CriticalityLevel.HIGH,
            description="Montos principales del presupuesto nacional uruguayo",
            current_value=f"Cifras detectadas: {len(budget_figures)}, grandes cifras: {len(large_figures)}",
            requires_adjustment=True,
            inflation_impact=True,
            temporal_analysis_required=True,
            reality_filter_notes="OBLIGATORIO: Todas las cifras requieren análisis inflación acumulada 2025-2029"
        ))
        
        # Variable ajuste por inflación
        inflation_mentions = re.findall(r'ajust[ae].*inflación|índice.*precio|IPC|deflactor', text, re.IGNORECASE)
        variables.append(BudgetVariable(
            name="Mecanismo Ajuste Inflación",
            type=VariableType.INFLATION,
            criticality=CriticalityLevel.BLOCKING if len(inflation_mentions) == 0 else CriticalityLevel.HIGH,
            description="Metodología ajuste por inflación para período plurianual",
            current_value=f"Menciones ajuste inflación: {len(inflation_mentions)}",
            requires_adjustment=len(inflation_mentions) == 0,
            inflation_impact=True,
            temporal_analysis_required=True,
            reality_filter_notes="BLOQUEANTE: Sin ajuste inflación, análisis temporal será inválido"
        ))
        
        return variables
    
    def scan_institutional_variables(self, content: Dict[str, Any]) -> List[BudgetVariable]:
        """Escanea variables institucionales y de política fiscal"""
        variables = []
        text = content['full_text']
        
        # Variable instituciones involucradas
        institutions = re.findall(r'Ministerio|Presidencia|BCU|Banco Central|MEF|OPP', text, re.IGNORECASE)
        variables.append(BudgetVariable(
            name="Marco Institucional Presupuestario",
            type=VariableType.INSTITUTIONAL,
            criticality=CriticalityLevel.MEDIUM,
            description="Instituciones responsables ejecución presupuesto plurianual",
            current_value=f"Instituciones mencionadas: {len(set(institutions))}",
            requires_adjustment=False,
            inflation_impact=False,
            temporal_analysis_required=False,
            reality_filter_notes="Verificar coordinación institucional para ejecución quinquenal"
        ))
        
        # Variable política fiscal
        fiscal_terms = re.findall(r'déficit|superávit|deuda|gasto público|inversión pública', text, re.IGNORECASE)
        variables.append(BudgetVariable(
            name="Política Fiscal Plurianual",
            type=VariableType.FISCAL,
            criticality=CriticalityLevel.HIGH,
            description="Estrategia fiscal para período gobierno 2025-2029",
            current_value=f"Términos fiscales detectados: {len(fiscal_terms)}",
            requires_adjustment=True,
            inflation_impact=True,
            temporal_analysis_required=True,
            reality_filter_notes="Analizar sostenibilidad fiscal en contexto inflacionario uruguayo"
        ))
        
        return variables
    
    def apply_reality_filter(self, variables: List[BudgetVariable], content: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Aplica Reality Filter para detectar inconsistencias críticas"""
        warnings = []
        adjustments_required = []
        
        # Check 1: Análisis inflación obligatorio
        inflation_vars = [v for v in variables if v.inflation_impact]
        if len(inflation_vars) > 0:
            adjustments_required.append(
                "CRÍTICO: Todas las cifras monetarias requieren ajuste por inflación acumulada 2025-2029"
            )
        
        # Check 2: Período plurianual
        pluriannual_vars = [v for v in variables if v.type == VariableType.PLURIANNUAL]
        if len(pluriannual_vars) > 0:
            warnings.append(
                "ANÁLISIS PLURIANUAL: Proyecciones 5 años requieren consideración incertidumbre económica"
            )
        
        # Check 3: Cifras realistas
        budget_figures = content.get('budget_figures', [])
        if len(budget_figures) > 10:
            warnings.append(
                "VERIFICAR REALISMO: Múltiples cifras presupuestarias requieren validación con contexto económico uruguayo"
            )
        
        # Check 4: Variables bloqueantes
        blocking_vars = [v for v in variables if v.criticality == CriticalityLevel.BLOCKING]
        for var in blocking_vars:
            adjustments_required.append(f"BLOQUEANTE: {var.name} - {var.reality_filter_notes}")
        
        return warnings, adjustments_required
    
    def generate_reality_based_recommendations(self, analysis_result: UruguayanBudgetAnalysis) -> List[str]:
        """Genera recomendaciones basadas en Reality Filter"""
        recommendations = []
        
        # Recomendaciones temporales
        recommendations.extend([
            "",
            "🎯 RECOMENDACIONES REALITY FILTER - PRESUPUESTO URUGUAY 2025-2029:",
            "",
            "1. AJUSTE INFLACIÓN OBLIGATORIO:",
            "   • Aplicar deflactor BCU para toda cifra temporal comparativa",
            "   • Usar meta inflación 3-7% anual (rango objetivo BCU)",
            "   • Calcular inflación acumulada período 2025-2029",
            "   • Expresar resultados en pesos constantes base enero 2025",
            "",
            "2. ANÁLISIS PLURIANUAL REALISTA:",
            "   • Considerar incertidumbre creciente años 2027-2029", 
            "   • Incluir escenarios alternativos (optimista/pesimista/base)",
            "   • Validar proyecciones con series históricas uruguayas",
            "   • Analizar impacto choques externos (commodities, Argentina, Brasil)",
            "",
            "3. CONTEXTO ECONÓMICO URUGUAYO:",
            "   • Verificar coherencia con Plan Quinquenal de Desarrollo",
            "   • Analizar sostenibilidad deuda pública (ratio deuda/PIB)",
            "   • Considerar restricciones balance de pagos",
            "   • Evaluar capacidad institucional ejecución quinquenal",
            "",
            "4. VARIABLES CRÍTICAS MONITOREO:",
            "   • Tipo de cambio real (competitividad externa)",
            "   • Precio commodities (soja, carne, arroz - exportaciones clave)",
            "   • Crecimiento PIB Argentina y Brasil (socios comerciales)",
            "   • Tasas interés internacionales (costo financiamiento)",
        ])
        
        return recommendations
    
    def analyze_budget_document(self, pdf_path: str) -> UruguayanBudgetAnalysis:
        """Análisis integral del presupuesto uruguayo con Enhanced Framework v3.0"""
        
        print("🇺🇾 ENHANCED UNIVERSAL FRAMEWORK v3.0 - PRESUPUESTO NACIONAL URUGUAY")
        print("=" * 80)
        
        # Extraer contenido PDF
        print("📄 Extrayendo contenido PDF...")
        content = self.extract_pdf_content(pdf_path)
        
        if content['total_pages'] == 0:
            print("❌ Error: No se pudo extraer contenido del PDF")
            return None
        
        # Análisis dinámico de variables
        print("🔍 Aplicando escaneo dinámico de variables...")
        all_variables = []
        all_variables.extend(self.scan_temporal_variables(content))
        all_variables.extend(self.scan_monetary_variables(content))
        all_variables.extend(self.scan_institutional_variables(content))
        
        # Aplicar Reality Filter
        print("🎯 Aplicando Reality Filter...")
        warnings, adjustments = self.apply_reality_filter(all_variables, content)
        
        # Detectar inconsistencias temporales
        temporal_issues = [
            var.reality_filter_notes for var in all_variables 
            if var.temporal_analysis_required and var.requires_adjustment
        ]
        
        # Análisis plurianual
        pluriannual_analysis = {
            'period': '2025-2029',
            'duration_years': 5,
            'government_term': 'Período constitucional completo',
            'inflation_impact': 'Alto - Requiere ajuste anual acumulativo',
            'uncertainty_level': 'Creciente hacia años finales 2027-2029'
        }
        
        # Figuras clave detectadas
        key_figures = {
            'total_articles': len(content['articles']),
            'budget_figures_detected': len(content['budget_figures']),
            'sections_detected': len(set(content['sections'])),
            'tables_detected': content['tables_detected'],
            'pages_analyzed': content['total_pages']
        }
        
        # Generar análisis
        analysis = UruguayanBudgetAnalysis(
            document_id="Uruguay-Budget-2025-2029",
            analysis_timestamp=datetime.now().isoformat(),
            budget_period="2025-2029 (Quinquenal)",
            total_pages=content['total_pages'],
            variables_detected=all_variables,
            inflation_adjustments_required=adjustments,
            temporal_inconsistencies=temporal_issues,
            reality_filter_warnings=warnings,
            key_figures=key_figures,
            pluriannual_analysis=pluriannual_analysis,
            recommendations=[],
            overall_assessment="PENDIENTE ANÁLISIS DETALLADO"
        )
        
        # Generar recomendaciones
        analysis.recommendations = self.generate_reality_based_recommendations(analysis)
        
        # Evaluación general
        blocking_vars = len([v for v in all_variables if v.criticality == CriticalityLevel.BLOCKING])
        critical_vars = len([v for v in all_variables if v.criticality == CriticalityLevel.CRITICAL])
        
        if blocking_vars > 0:
            analysis.overall_assessment = "REQUIERE CORRECCIONES BLOQUEANTES - ANÁLISIS INFLACIÓN CRÍTICO"
        elif critical_vars > 2:
            analysis.overall_assessment = "ANÁLISIS VIABLE CON CORRECCIONES CRÍTICAS REQUERIDAS"
        else:
            analysis.overall_assessment = "DOCUMENTO ANALIZABLE - APLICAR REALITY FILTER COMPLETO"
        
        return analysis

def main():
    """Ejecuta análisis completo presupuesto uruguayo"""
    
    pdf_file = Path("/home/user/webapp/Presupuesto_Uruguay_2025-2029.pdf")
    
    if not pdf_file.exists():
        print(f"❌ Error: No se encontró {pdf_file}")
        return
    
    print("🇺🇾 INICIANDO ANÁLISIS PRESUPUESTO NACIONAL URUGUAY 2025-2029")
    print(f"📁 Archivo: {pdf_file}")
    print(f"📊 Tamaño: {pdf_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Ejecutar análisis
    analyzer = EnhancedUruguayanBudgetAnalyzer()
    analysis = analyzer.analyze_budget_document(str(pdf_file))
    
    if not analysis:
        print("❌ Error en análisis")
        return
    
    # Mostrar resultados
    print(f"\n🏆 ANÁLISIS COMPLETADO")
    print(f"📊 Variables detectadas: {len(analysis.variables_detected)}")
    print(f"⚠️ Advertencias Reality Filter: {len(analysis.reality_filter_warnings)}")
    print(f"🔧 Ajustes requeridos: {len(analysis.inflation_adjustments_required)}")
    print(f"🎯 Evaluación: {analysis.overall_assessment}")
    
    # Mostrar variables críticas
    critical_vars = [v for v in analysis.variables_detected if v.criticality in [CriticalityLevel.BLOCKING, CriticalityLevel.CRITICAL]]
    if critical_vars:
        print(f"\n🚨 VARIABLES CRÍTICAS ({len(critical_vars)}):")
        for var in critical_vars:
            print(f"   🔴 {var.name}: {var.reality_filter_notes}")
    
    # Guardar resultados
    output_file = '/home/user/webapp/uruguay_budget_analysis_enhanced_v3.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        result_dict = asdict(analysis)
        json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Análisis guardado en: {output_file}")
    
    # Guardar contenido extraído para análisis posterior
    content_file = '/home/user/webapp/uruguay_budget_extracted_content.txt'
    try:
        content = analyzer.extract_pdf_content(str(pdf_file))
        with open(content_file, 'w', encoding='utf-8') as f:
            f.write("# PRESUPUESTO NACIONAL URUGUAY 2025-2029 - CONTENIDO EXTRAÍDO\n\n")
            f.write(f"Total páginas: {content['total_pages']}\n")
            f.write(f"Artículos detectados: {len(content['articles'])}\n")
            f.write(f"Tablas detectadas: {content['tables_detected']}\n\n")
            f.write("## CONTENIDO COMPLETO:\n\n")
            f.write(content['full_text'])
        
        print(f"📄 Contenido extraído guardado en: {content_file}")
    except Exception as e:
        print(f"⚠️ Error guardando contenido: {str(e)}")

if __name__ == "__main__":
    main()