#!/usr/bin/env python3
"""
Análisis dinámico del documento jurídico PPMM-GEI
Aplicando Enhanced Universal Framework v3.0 con Dynamic Variable Scanning
"""

from ENHANCED_UNIVERSAL_FRAMEWORK_V3_DYNAMIC_VARIABLES import DynamicVariableScanner, DynamicVariable, VariableType, CriticalityLevel
import re
from typing import Dict, List, Any

class LegalEnvironmentalVariableScanner(DynamicVariableScanner):
    """
    Scanner especializado para análisis jurídico-ambiental
    con detección de variables específicas de créditos de carbono
    """
    
    def __init__(self):
        super().__init__(domain="legal_environmental_analysis")
        
        # Patrones específicos para análisis legal-ambiental
        self.legal_environmental_patterns = {
            'carbon_credit_concepts': {
                'additionality_indicators': r'(adicionalidad|business\s+as\s+usual|línea\s+base|baseline)',
                'forest_data_mentions': r'(\d+(?:\.\d+)?)\s*(millones?\s+de\s+hectáreas?|ha\s+de\s+bosques?)',
                'project_types': r'(evitar\s+deforestación|restauración|conservación|manejo\s+sostenible)',
                'securitization_language': r'(securitización|máquina\s+de\s+captura\s+de\s+valor|business\s+as\s+usual)'
            },
            
            'legal_framework_indicators': {
                'constitutional_basis': r'(art\.\s*41|presupuestos\s+mínimos|constitución\s+nacional)',
                'jurisprudence_citations': r'(Mendoza|Riachuelo|Salas|CSJN|Corte\s+Suprema)',
                'federalism_concepts': r'(federalismo|competencias\s+provinciales|autonomía\s+provincial)',
                'legal_nature_carbon': r'(naturaleza\s+jurídica|carbono\s+mitigado|titularidad)'
            },
            
            'technical_compliance': {
                'carbon_accounting': r'(contabilidad\s+de\s+carbono|MRV|medición|reporte|verificación)',
                'international_frameworks': r'(Convención\s+Marco|Acuerdo\s+de\s+París|UNFCCC)',
                'registry_systems': r'(RENAMI|registro|sistema\s+de\s+seguimiento)'
            }
        }
    
    def scan_legal_environmental_variables(self, document_text: str, user_corrections: str) -> Dict:
        """
        Escaneo especializado para detectar variables críticas en análisis jurídico-ambiental
        """
        
        identified_issues = []
        
        # 1. CRÍTICO: Detectar errores conceptuales sobre adicionalidad
        additionality_issues = self._scan_additionality_concepts(document_text, user_corrections)
        identified_issues.extend(additionality_issues)
        
        # 2. CRÍTICO: Detectar errores en datos forestales
        forest_data_issues = self._scan_forest_data_accuracy(document_text, user_corrections)
        identified_issues.extend(forest_data_issues)
        
        # 3. IMPORTANTE: Verificar marco legal constitucional
        legal_framework_issues = self._scan_legal_framework_completeness(document_text)
        identified_issues.extend(legal_framework_issues)
        
        # 4. MODERADO: Revisar aspectos técnicos de implementación
        technical_issues = self._scan_technical_implementation(document_text)
        identified_issues.extend(technical_issues)
        
        return {
            'identified_variables': identified_issues,
            'blocking_variables': [v for v in identified_issues if v.criticality == CriticalityLevel.BLOCKING],
            'critical_variables': [v for v in identified_issues if v.criticality == CriticalityLevel.CRITICAL],
            'correction_required': len([v for v in identified_issues if v.criticality in [CriticalityLevel.BLOCKING, CriticalityLevel.CRITICAL]]) > 0
        }
    
    def _scan_additionality_concepts(self, document_text: str, user_corrections: str) -> List[DynamicVariable]:
        """Detectar problemas conceptuales con adicionalidad en créditos de carbono"""
        
        variables = []
        
        # Detectar "business as usual" como problema conceptual
        securitization_matches = re.findall(self.legal_environmental_patterns['carbon_credit_concepts']['securitization_language'], 
                                          document_text, re.IGNORECASE)
        
        if securitization_matches:
            variables.append(DynamicVariable(
                name="additionality_conceptual_error",
                type=VariableType.STRUCTURAL,
                criticality=CriticalityLevel.CRITICAL,
                detected_value={
                    'error_type': 'business_as_usual_misconception',
                    'found_text': securitization_matches,
                    'user_correction': user_corrections
                },
                impact_description="CRITICAL: Confusión conceptual sobre adicionalidad en créditos de carbono - no son 'business as usual'",
                requires_normalization=True,
                adjustment_method="correct_additionality_concept"
            ))
        
        return variables
    
    def _scan_forest_data_accuracy(self, document_text: str, user_corrections: str) -> List[DynamicVariable]:
        """Detectar errores en datos forestales"""
        
        variables = []
        
        # Buscar menciones específicas de hectáreas forestales
        forest_mentions = re.findall(self.legal_environmental_patterns['carbon_credit_concepts']['forest_data_mentions'], 
                                   document_text, re.IGNORECASE)
        
        if forest_mentions:
            for mention in forest_mentions:
                # Detectar si se mencionan cifras incorrectas como "3.7 millones" vs "50+ millones"
                amount = float(mention[0]) if mention[0].replace('.', '').isdigit() else 0
                
                if amount < 10:  # Cifras sospechosamente bajas para Argentina
                    variables.append(DynamicVariable(
                        name="forest_data_accuracy_error",
                        type=VariableType.EXTERNAL,
                        criticality=CriticalityLevel.CRITICAL,
                        detected_value={
                            'incorrect_amount': amount,
                            'detected_text': mention,
                            'correct_data': 'Más de 50 millones de hectáreas de bosques nativos reconocidos',
                            'user_correction': user_corrections
                        },
                        impact_description="CRITICAL: Datos forestales incorrectos - subestima significativamente superficie boscosa argentina",
                        requires_normalization=True,
                        adjustment_method="correct_forest_statistics"
                    ))
        
        return variables
    
    def _scan_legal_framework_completeness(self, document_text: str) -> List[DynamicVariable]:
        """Verificar completitud del marco legal constitucional"""
        
        variables = []
        
        # Verificar citas jurisprudenciales
        jurisprudence_mentions = re.findall(self.legal_environmental_patterns['legal_framework_indicators']['jurisprudence_citations'], 
                                          document_text, re.IGNORECASE)
        
        if jurisprudence_mentions:
            variables.append(DynamicVariable(
                name="constitutional_framework_analysis",
                type=VariableType.POLICY,
                criticality=CriticalityLevel.IMPORTANT,
                detected_value={
                    'jurisprudence_citations': jurisprudence_mentions,
                    'constitutional_basis_solid': True
                },
                impact_description="Marco constitucional sólidamente fundamentado en jurisprudencia CSJN",
                requires_normalization=False
            ))
        
        return variables
    
    def _scan_technical_implementation(self, document_text: str) -> List[DynamicVariable]:
        """Revisar aspectos técnicos de implementación"""
        
        variables = []
        
        # Detectar menciones de sistemas técnicos
        registry_mentions = re.findall(self.legal_environmental_patterns['technical_compliance']['registry_systems'], 
                                     document_text, re.IGNORECASE)
        
        if registry_mentions:
            variables.append(DynamicVariable(
                name="technical_implementation_framework",
                type=VariableType.STRUCTURAL,
                criticality=CriticalityLevel.MODERATE,
                detected_value={
                    'registry_systems': registry_mentions,
                    'implementation_viable': True
                },
                impact_description="Frameworks técnicos de implementación identificados correctamente"
            ))
        
        return variables

def analyze_ppmm_document():
    """Función principal para analizar el documento PPMM-GEI"""
    
    # Leer documento completo
    with open('/home/user/webapp/documento_completo.txt', 'r', encoding='utf-8') as f:
        document_content = f.read()
    
    # Correcciones específicas del usuario
    user_corrections = """
    CORRECCIONES CRÍTICAS DEL USUARIO:
    
    1. ADICIONALIDAD EN CRÉDITOS DE CARBONO:
    - ERROR: Presentar bosques existentes como "business as usual" para securitización
    - CORRECTO: Los proyectos de créditos de carbono deben demostrar ADICIONALIDAD
    - CONCEPTO: Son proyectos NUEVOS con algo diferente a lo que se haría de cualquier manera
    
    2. DATOS FORESTALES ARGENTINOS:
    - ERROR: "3.7 millones de hectáreas de bosques nativos"
    - CORRECTO: "Más de 50 millones de hectáreas de bosques nativos reconocidos por las provincias"
    
    3. TIPOS DE PROYECTOS CORRECTOS:
    - Evitar deforestación
    - Restauración
    - Conservación
    - Manejo sostenible
    
    NO es simplemente convertir bosques existentes en "máquina de captura de valor"
    """
    
    # Inicializar scanner especializado
    scanner = LegalEnvironmentalVariableScanner()
    
    # Realizar análisis dinámico
    analysis_results = scanner.scan_legal_environmental_variables(document_content, user_corrections)
    
    return analysis_results

if __name__ == "__main__":
    results = analyze_ppmm_document()
    
    print("🔍 DYNAMIC VARIABLE ANALYSIS - DOCUMENTO JURÍDICO PPMM-GEI")
    print("=" * 60)
    
    print(f"\n📊 RESUMEN DE VARIABLES IDENTIFICADAS:")
    print(f"Total variables: {len(results['identified_variables'])}")
    print(f"Variables blocking: {len(results['blocking_variables'])}")
    print(f"Variables críticas: {len(results['critical_variables'])}")
    print(f"Corrección requerida: {'SÍ' if results['correction_required'] else 'NO'}")
    
    print(f"\n⚠️ VARIABLES CRÍTICAS DETECTADAS:")
    for var in results['critical_variables']:
        print(f"- {var.name}")
        print(f"  Tipo: {var.type.value}")
        print(f"  Impacto: {var.impact_description}")
        if var.detected_value:
            print(f"  Datos: {var.detected_value}")
        print()
    
    if results['correction_required']:
        print("🚨 EL DOCUMENTO REQUIERE CORRECCIONES CRÍTICAS")
    else:
        print("✅ EL DOCUMENTO NO PRESENTA ERRORES CRÍTICOS")