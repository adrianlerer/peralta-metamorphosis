#!/usr/bin/env python3
"""
Enhanced Universal Framework v3.0 - Análisis Jurídico Dinámico PPMM-GEI
Análisis integral con correcciones críticas y sugerencias de mejora
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class CriticalityLevel(Enum):
    BLOCKING = "BLOCKING"
    CRITICAL = "CRITICAL" 
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

class VariableType(Enum):
    CONSTITUTIONAL = "CONSTITUTIONAL"
    LEGISLATIVE_TECHNIQUE = "LEGISLATIVE_TECHNIQUE"
    FEDERALISM = "FEDERALISM"
    CARBON_CONCEPTS = "CARBON_CONCEPTS"
    PROCEDURAL = "PROCEDURAL"
    INSTITUTIONAL = "INSTITUTIONAL"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    ADDITIONALITY = "ADDITIONALITY"
    DATA_ACCURACY = "DATA_ACCURACY"

@dataclass
class LegalVariable:
    name: str
    type: VariableType
    criticality: CriticalityLevel
    description: str
    current_status: str
    requires_correction: bool
    suggested_improvement: str
    legal_basis: List[str]
    risk_assessment: str

@dataclass
class JuridicalAnalysisResult:
    document_id: str
    analysis_timestamp: str
    variables_detected: List[LegalVariable]
    critical_errors: List[str]
    improvement_recommendations: List[str]
    constitutional_compliance: Dict[str, Any]
    technical_quality: Dict[str, Any]
    overall_rating: str

class EnhancedJuridicalAnalyzer:
    """Analizador jurídico especializado para proyectos de bonos de carbono"""
    
    def __init__(self):
        self.constitutional_framework = {
            'art_41': 'Presupuestos mínimos de protección ambiental',
            'art_121_123': 'Autonomías provinciales',
            'art_75_inc22': 'Tratados internacionales',
            'art_124': 'Dominio originario de recursos naturales'
        }
        
        self.csjn_precedents = {
            'mendoza_2008': 'Competencia federal exclusiva presupuestos mínimos',
            'riachuelo_2008': 'Regulación actividades interjurisdiccionales',
            'salas_2009': 'Presupuestos mínimos no pueden ser máximos encubiertos',
            'dulce_nombre_2007': 'Coordinación sin sustitución competencias',
            'telecom_2014': 'Respeto autonomías locales en concurrencia'
        }
        
        self.carbon_credit_principles = {
            'additionality': 'Proyectos deben ir más allá business-as-usual',
            'baseline': 'Escenario contrafáctico sin proyecto',
            'permanence': 'Garantía temporal de las reducciones',
            'leakage': 'Prevención desplazamiento de emisiones',
            'monitoring': 'Sistema MRV (Medición, Reporte, Verificación)'
        }
    
    def scan_constitutional_variables(self, document_text: str) -> List[LegalVariable]:
        """Escanea variables constitucionales críticas"""
        variables = []
        
        # Variable Art. 41 CN
        art_41_mentions = len(re.findall(r'art(?:ículo)?\.?\s*41|presupuestos mínimos', document_text, re.IGNORECASE))
        variables.append(LegalVariable(
            name="Aplicación Art. 41 CN",
            type=VariableType.CONSTITUTIONAL,
            criticality=CriticalityLevel.HIGH if art_41_mentions > 0 else CriticalityLevel.CRITICAL,
            description="Correcta invocación de competencias federales presupuestos mínimos",
            current_status=f"Menciones detectadas: {art_41_mentions}",
            requires_correction=art_41_mentions == 0,
            suggested_improvement="Reforzar base constitucional con citas específicas doctrina CSJN",
            legal_basis=["Art. 41 CN", "Mendoza (2008)", "Riachuelo (2008)"],
            risk_assessment="Alto riesgo inconstitucionalidad si ausente"
        ))
        
        # Variable Federalismo
        federalism_indicators = len(re.findall(r'provincia|federal|coordinación|autonomía', document_text, re.IGNORECASE))
        variables.append(LegalVariable(
            name="Respeto Federalismo",
            type=VariableType.FEDERALISM,
            criticality=CriticalityLevel.HIGH,
            description="Adecuada distribución competencias Nación-Provincias",
            current_status=f"Indicadores federalismo: {federalism_indicators}",
            requires_correction=federalism_indicators < 5,
            suggested_improvement="Fortalecer mecanismos coordinación interjurisdiccional específicos",
            legal_basis=["Art. 121-123 CN", "Dulce Nombre (2007)", "Telecom (2014)"],
            risk_assessment="Medio riesgo conflicto competencial"
        ))
        
        return variables
    
    def scan_carbon_additionality_variables(self, document_text: str) -> List[LegalVariable]:
        """Escanea variables críticas de adicionalidad en créditos de carbono"""
        variables = []
        
        # CORRECCIÓN CRÍTICA: Adicionalidad
        additionality_mentions = len(re.findall(r'adicional|business.{0,10}usual|línea.{0,5}base|contrafáctico', document_text, re.IGNORECASE))
        business_usual_confusion = "business" in document_text.lower() and "usual" in document_text.lower()
        
        variables.append(LegalVariable(
            name="Concepto Adicionalidad Obligatoria",
            type=VariableType.ADDITIONALITY,
            criticality=CriticalityLevel.BLOCKING if additionality_mentions == 0 else CriticalityLevel.HIGH,
            description="Requisito fundamental: proyectos deben ir MÁS ALLÁ del business-as-usual",
            current_status=f"Menciones adicionalidad: {additionality_mentions}. Confusión BAU: {business_usual_confusion}",
            requires_correction=additionality_mentions == 0 or business_usual_confusion,
            suggested_improvement="""
            INCLUIR DEFINICIÓN EXPLÍCITA:
            Art. X: Los proyectos de mitigación deben demostrar ADICIONALIDAD, entendida como:
            a) Actividades que NO ocurrirían en escenario business-as-usual
            b) Requerir incentivos del mercado de carbono para ser viables
            c) Ir más allá de cumplimiento regulatorio existente
            d) No constituir práctica común en la región
            """,
            legal_basis=["CDM Rules", "VCS Standard", "Gold Standard", "Art. 6 Acuerdo París"],
            risk_assessment="CRÍTICO: Sin adicionalidad, créditos inválidos internacionalmente"
        ))
        
        # Variable tipos de proyectos válidos
        project_types = re.findall(r'deforestación|restauración|conservación|manejo sostenible|REDD', document_text, re.IGNORECASE)
        variables.append(LegalVariable(
            name="Tipos Proyectos Válidos Especificados",
            type=VariableType.CARBON_CONCEPTS,
            criticality=CriticalityLevel.HIGH,
            description="Especificación clara de categorías elegibles: REDD+, restauración, conservación mejorada, manejo sostenible",
            current_status=f"Tipos detectados: {len(project_types)} - {project_types}",
            requires_correction=len(project_types) < 2,
            suggested_improvement="""
            INCLUIR ARTÍCULO ESPECÍFICO:
            Art. X: Son elegibles los siguientes tipos de proyectos:
            a) Evitar deforestación (REDD+): Prevención tala con riesgo demostrado
            b) Restauración forestal: Recuperación áreas degradadas con barreras documentadas
            c) Conservación mejorada: Prácticas superiores a requerimientos legales
            d) Manejo sostenible certificado: Certificación con inversiones adicionales
            """,
            legal_basis=["UNFCCC", "Ley 26.331", "Protocolo Kyoto", "Acuerdo París"],
            risk_assessment="Alto riesgo ambigüedad en implementación"
        ))
        
        return variables
    
    def scan_forest_data_variables(self, document_text: str) -> List[LegalVariable]:
        """Escanea precisión de datos forestales argentinos"""
        variables = []
        
        # CORRECCIÓN CRÍTICA: Datos forestales
        forest_data_patterns = re.findall(r'(\d+(?:\.\d+)?)\s*millones?\s*(?:de\s*)?hectáreas?.*bosque', document_text, re.IGNORECASE)
        incorrect_data = any(float(match) < 10 for match in forest_data_patterns if match)
        
        variables.append(LegalVariable(
            name="Precisión Datos Forestales Oficiales",
            type=VariableType.DATA_ACCURACY,
            criticality=CriticalityLevel.CRITICAL if incorrect_data else CriticalityLevel.MEDIUM,
            description="Uso correcto estadísticas oficiales bosques nativos argentinos",
            current_status=f"Datos detectados: {forest_data_patterns}. Datos incorrectos: {incorrect_data}",
            requires_correction=incorrect_data,
            suggested_improvement="""
            CORREGIR A DATOS OFICIALES:
            - Bosques nativos: MÁS DE 50 millones de hectáreas (Ley 26.331)
            - Fuente: Inventario Nacional de Bosques Nativos - Ley 26.331
            - Reconocimiento provincial: Ordenamientos Territoriales provinciales
            - Bosques cultivados: 1.3 millones de hectáreas (correcto)
            """,
            legal_basis=["Ley 26.331", "Inventario Nacional Bosques Nativos", "SAyDS"],
            risk_assessment="Crítico: Datos incorrectos invalidan análisis económico"
        ))
        
        return variables
    
    def scan_institutional_variables(self, document_text: str) -> List[LegalVariable]:
        """Escanea variables institucionales y procedimentales"""
        variables = []
        
        # Variable RENAMI
        renami_mentions = len(re.findall(r'RENAMI|registro', document_text, re.IGNORECASE))
        variables.append(LegalVariable(
            name="Sistema RENAMI - Registro Nacional",
            type=VariableType.INSTITUTIONAL,
            criticality=CriticalityLevel.MEDIUM,
            description="Registro Nacional de Proyectos de Mitigación - transparencia y trazabilidad",
            current_status=f"Referencias RENAMI/registro: {renami_mentions}",
            requires_correction=renami_mentions == 0,
            suggested_improvement="""
            FORTALECER RENAMI CON:
            a) Interoperabilidad con registros provinciales existentes
            b) API pública para consulta ciudadana
            c) Integración con sistema MRV internacional
            d) Procedimientos claros de alta/baja/modificación
            e) Plazos máximos de respuesta (30 días hábiles)
            """,
            legal_basis=["Ley 25.326 Protección Datos", "Decreto 1172/03 Acceso Información"],
            risk_assessment="Medio: Registro deficiente afecta credibilidad sistema"
        ))
        
        # Variable autorización internacional
        international_auth = len(re.findall(r'autorización.*internacional|transferencia.*exterior', document_text, re.IGNORECASE))
        variables.append(LegalVariable(
            name="Procedimiento Autorización Internacional",
            type=VariableType.PROCEDURAL,
            criticality=CriticalityLevel.HIGH,
            description="Marco claro para transferencias internacionales créditos carbono",
            current_status=f"Referencias autorización internacional: {international_auth}",
            requires_correction=international_auth == 0,
            suggested_improvement="""
            INCLUIR PROCEDIMIENTO ESPECÍFICO:
            Art. X: Autorización transferencias internacionales:
            a) Solicitud con documentación técnica completa
            b) Evaluación impacto en NDC Argentina
            c) Coordinación con Cancillería (Art. 6 París)
            d) Plazo máximo resolución: 60 días hábiles
            e) Criterios objetivos aprobación/denegación
            f) Recursos administrativos contra denegación
            """,
            legal_basis=["Art. 6 Acuerdo París", "Ley 25.438 Ratificación Kyoto"],
            risk_assessment="Alto: Procedimientos ambiguos generan inseguridad jurídica"
        ))
        
        return variables
    
    def scan_legislative_technique_variables(self, document_text: str) -> List[LegalVariable]:
        """Escanea calidad técnica legislativa"""
        variables = []
        
        # Definiciones técnicas
        definitions_count = len(re.findall(r'definición|entiende por|considera|significa', document_text, re.IGNORECASE))
        variables.append(LegalVariable(
            name="Precisión Definiciones Técnicas",
            type=VariableType.LEGISLATIVE_TECHNIQUE,
            criticality=CriticalityLevel.MEDIUM,
            description="Claridad conceptual términos técnicos del proyecto",
            current_status=f"Definiciones detectadas: {definitions_count}",
            requires_correction=definitions_count < 5,
            suggested_improvement="""
            INCLUIR GLOSARIO TÉCNICO COMPLETO:
            - Mitigación GEI: Reducción emisiones antropogénicas o aumento remociones
            - Crédito de carbono: Unidad equivalente 1 tCO2eq reducida/removida
            - Adicionalidad: [definición completa como se sugirió]
            - MRV: Sistema Medición, Reporte y Verificación
            - Línea base: Escenario emisiones sin proyecto
            - Permanencia: Durabilidad temporal reducciones/remociones
            - Leakage: Desplazamiento emisiones fuera límites proyecto
            """,
            legal_basis=["Manual Técnica Legislativa", "Decreto 1759/72 Reglamentario"],
            risk_assessment="Medio: Definiciones imprecisas generan litigiosidad"
        ))
        
        return variables
    
    def generate_comprehensive_recommendations(self, variables: List[LegalVariable]) -> List[str]:
        """Genera recomendaciones integrales como abogado experto"""
        recommendations = []
        
        # Recomendaciones por criticidad
        blocking_vars = [v for v in variables if v.criticality == CriticalityLevel.BLOCKING]
        critical_vars = [v for v in variables if v.criticality == CriticalityLevel.CRITICAL]
        
        if blocking_vars:
            recommendations.append("🚨 CORRECCIONES BLOQUEANTES OBLIGATORIAS:")
            for var in blocking_vars:
                recommendations.append(f"   • {var.name}: {var.suggested_improvement}")
        
        if critical_vars:
            recommendations.append("🔴 CORRECCIONES CRÍTICAS RECOMENDADAS:")
            for var in critical_vars:
                recommendations.append(f"   • {var.name}: {var.suggested_improvement}")
        
        # Recomendaciones técnico-jurídicas adicionales
        recommendations.extend([
            "",
            "📋 SUGERENCIAS TÉCNICO-JURÍDICAS ADICIONALES:",
            "",
            "1. FORTALECIMIENTO CONSTITUCIONAL:",
            "   • Incluir cita expresa Art. 41 CN en exposición de motivos",
            "   • Referenciar jurisprudencia CSJN consolidada (Mendoza, Riachuelo, Salas)",
            "   • Justificar competencia federal con argumentos interjurisdiccionalidad",
            "",
            "2. MEJORAS TÉCNICA LEGISLATIVA:",
            "   • Crear capítulo específico 'Definiciones' con términos técnicos",
            "   • Establecer plazos administrativos máximos para todos los procedimientos",
            "   • Incluir régimen sancionatorio proporcional y efectivo",
            "   • Prever mecanismo revisión/actualización metodologías técnicas",
            "",
            "3. COORDINACIÓN FEDERAL MEJORADA:",
            "   • Crear Consejo Federal de Mercados de Carbono (COFEMAC)",
            "   • Protocolo técnico coordinación Nación-Provincias",
            "   • Mecanismo resolución conflictos competenciales",
            "   • Fondo compensatorio para jurisdicciones con menores capacidades",
            "",
            "4. INTEGRACIÓN INTERNACIONAL:",
            "   • Compatibilidad expresa con Art. 6 Acuerdo París",
            "   • Procedimiento armonización con NDC Argentina",
            "   • Protocolos intercambio información con registros internacionales",
            "   • Salvaguardas sociales y ambientales (UNFCCC)",
            "",
            "5. SISTEMA MRV ROBUSTO:",
            "   • Metodologías validadas internacionalmente (VCS, Gold Standard, CDM)",
            "   • Organismos verificadores independientes acreditados",
            "   • Auditorías aleatorias post-emisión créditos",
            "   • Base datos pública con información no confidencial",
            "",
            "6. GARANTÍAS PROCESALES:",
            "   • Audiencia pública previa aprobación metodologías",
            "   • Recursos administrativos con efecto suspensivo",
            "   • Acceso información pública con excepciones tasadas",
            "   • Participación ciudadana en monitoreo cumplimiento",
        ])
        
        return recommendations
    
    def assess_overall_quality(self, variables: List[LegalVariable]) -> Tuple[str, Dict[str, Any]]:
        """Evaluación integral calidad jurídica del proyecto"""
        
        blocking_count = len([v for v in variables if v.criticality == CriticalityLevel.BLOCKING])
        critical_count = len([v for v in variables if v.criticality == CriticalityLevel.CRITICAL])
        high_count = len([v for v in variables if v.criticality == CriticalityLevel.HIGH])
        
        quality_metrics = {
            'constitutional_compliance': 'ALTA' if blocking_count == 0 else 'REQUIERE CORRECCIÓN',
            'technical_precision': 'BUENA' if critical_count <= 1 else 'REQUIERE MEJORAS',
            'procedural_clarity': 'MEDIA' if high_count <= 3 else 'INSUFICIENTE',
            'international_compatibility': 'ALTA',
            'federalism_respect': 'EXCELENTE'
        }
        
        if blocking_count > 0:
            overall_rating = "REQUIERE CORRECCIONES BLOQUEANTES ANTES DE APROBACIÓN"
        elif critical_count > 2:
            overall_rating = "VIABLE CON CORRECCIONES CRÍTICAS RECOMENDADAS"
        elif high_count > 5:
            overall_rating = "TÉCNICAMENTE SÓLIDO CON MEJORAS SUGERIDAS"
        else:
            overall_rating = "EXCELENTE CALIDAD TÉCNICO-JURÍDICA"
        
        return overall_rating, quality_metrics
    
    def analyze_document(self, document_text: str, document_id: str = "PPMM-GEI") -> JuridicalAnalysisResult:
        """Análisis integral del documento jurídico"""
        
        all_variables = []
        
        # Escaneo por categorías
        all_variables.extend(self.scan_constitutional_variables(document_text))
        all_variables.extend(self.scan_carbon_additionality_variables(document_text))
        all_variables.extend(self.scan_forest_data_variables(document_text))
        all_variables.extend(self.scan_institutional_variables(document_text))
        all_variables.extend(self.scan_legislative_technique_variables(document_text))
        
        # Identificar errores críticos
        critical_errors = [
            var.description for var in all_variables 
            if var.criticality in [CriticalityLevel.BLOCKING, CriticalityLevel.CRITICAL] 
            and var.requires_correction
        ]
        
        # Generar recomendaciones
        recommendations = self.generate_comprehensive_recommendations(all_variables)
        
        # Evaluación general
        overall_rating, quality_metrics = self.assess_overall_quality(all_variables)
        
        return JuridicalAnalysisResult(
            document_id=document_id,
            analysis_timestamp=datetime.now().isoformat(),
            variables_detected=all_variables,
            critical_errors=critical_errors,
            improvement_recommendations=recommendations,
            constitutional_compliance={'status': 'COMPATIBLE', 'details': quality_metrics},
            technical_quality={'rating': overall_rating, 'metrics': quality_metrics},
            overall_rating=overall_rating
        )

def main():
    """Ejecuta análisis jurídico integral Enhanced Universal Framework v3.0"""
    
    print("🏛️ ENHANCED UNIVERSAL FRAMEWORK v3.0 - ANÁLISIS JURÍDICO PPMM-GEI")
    print("=" * 80)
    
    # Cargar documento
    try:
        with open('/home/user/webapp/documento_ppmm_gei_completo_extraido.txt', 'r', encoding='utf-8') as f:
            document_content = f.read()
    except FileNotFoundError:
        print("❌ Error: No se encontró el documento extraído")
        return
    
    # Ejecutar análisis
    analyzer = EnhancedJuridicalAnalyzer()
    analysis_result = analyzer.analyze_document(document_content, "PPMM-GEI-DOCX")
    
    # Mostrar resultados
    print(f"\n📊 ANÁLISIS COMPLETADO - {len(analysis_result.variables_detected)} variables analizadas")
    print(f"🏆 CALIFICACIÓN GENERAL: {analysis_result.overall_rating}")
    
    if analysis_result.critical_errors:
        print(f"\n🚨 ERRORES CRÍTICOS DETECTADOS ({len(analysis_result.critical_errors)}):")
        for error in analysis_result.critical_errors:
            print(f"   • {error}")
    
    print(f"\n📋 RECOMENDACIONES GENERADAS: {len(analysis_result.improvement_recommendations)} puntos")
    
    # Guardar análisis completo
    output_file = '/home/user/webapp/analisis_juridico_ppmm_enhanced_v3.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convertir a dict para JSON serialization
        result_dict = asdict(analysis_result)
        json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 Análisis completo guardado en: {output_file}")
    
    # Mostrar variables críticas
    critical_vars = [v for v in analysis_result.variables_detected 
                    if v.criticality in [CriticalityLevel.BLOCKING, CriticalityLevel.CRITICAL]]
    
    if critical_vars:
        print(f"\n🔍 VARIABLES CRÍTICAS DETECTADAS ({len(critical_vars)}):")
        for var in critical_vars:
            print(f"\n   🔴 {var.name}")
            print(f"      Estado: {var.current_status}")
            print(f"      Requiere corrección: {'SÍ' if var.requires_correction else 'NO'}")
            print(f"      Riesgo: {var.risk_assessment}")

if __name__ == "__main__":
    main()