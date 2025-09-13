#!/usr/bin/env python3
"""
Auditoría Crítica: ¿Está el TGA introduciendo sesgo pro-colectivista?
Análisis específico de los sesgos ideológicos en la aplicación TGA

Revisar si las "mejoras metodológicas" están sesgando hacia perspectivas estatistas/regulatorias
"""

import json
from typing import Dict, List

class CollectivistBiasAudit:
    """
    Auditar si el análisis TGA está introduciendo sesgo pro-colectivista/estatista
    """
    
    def __init__(self):
        # Cargar resultados TGA para auditoría
        with open('tga_enhanced_analysis_gollan_20250913_022913.json', 'r', encoding='utf-8') as f:
            self.tga_results = json.load(f)
    
    def audit_perspective_balance(self) -> Dict:
        """
        AUDITORÍA 1: ¿Las perspectivas múltiples están balanceadas o sesgadas?
        """
        print("=== AUDITORÍA 1: BALANCE DE PERSPECTIVAS ===")
        
        perspectives = self.tga_results['methodology_enhancement']['multi_perspective_analysis']['perspectives']
        
        # Analizar sesgo en definición de perspectivas
        perspective_analysis = {}
        
        for perspective_name, perspective_data in perspectives.items():
            top_3_sources = perspective_data['priority_ranking'][:3]
            interpretation = perspective_data['interpretation']
            
            # Clasificar orientación ideológica
            if perspective_name == 'liberal_constitucional':
                ideological_lean = 'INDIVIDUAL_RIGHTS'
                regulatory_preference = 'MINIMAL_STATE'
            elif perspective_name == 'estatista_regulatorio':
                ideological_lean = 'COLLECTIVE_PROTECTION'  
                regulatory_preference = 'STRONG_STATE'
            elif perspective_name == 'tecno_pragmatico':
                ideological_lean = 'UTILITARIAN_BALANCE'
                regulatory_preference = 'ADAPTIVE_STATE'
            
            perspective_analysis[perspective_name] = {
                'ideological_lean': ideological_lean,
                'regulatory_preference': regulatory_preference,
                'top_sources': top_3_sources,
                'interpretation_bias': self._analyze_interpretation_bias(interpretation)
            }
        
        # Detectar sesgo estructural
        regulatory_perspectives = sum(1 for p in perspective_analysis.values() 
                                    if p['regulatory_preference'] in ['STRONG_STATE', 'ADAPTIVE_STATE'])
        
        individual_perspectives = sum(1 for p in perspective_analysis.values()
                                    if p['regulatory_preference'] == 'MINIMAL_STATE')
        
        structural_bias = {
            'regulatory_leaning_perspectives': regulatory_perspectives,  # 2 (estatista + tecno)
            'individual_leaning_perspectives': individual_perspectives,   # 1 (liberal)
            'bias_ratio': regulatory_perspectives / individual_perspectives,  # 2:1 pro-regulatorio
            'bias_detected': regulatory_perspectives > individual_perspectives
        }
        
        print(f"Perspectivas pro-regulatorias: {regulatory_perspectives}")
        print(f"Perspectivas pro-individuales: {individual_perspectives}")
        print(f"Ratio sesgo: {structural_bias['bias_ratio']:.1f}:1 hacia regulación")
        
        if structural_bias['bias_detected']:
            print("⚠️ SESGO ESTRUCTURAL DETECTADO: Sobrerrepresentación perspectivas pro-regulatorias")
        
        return {
            'perspective_analysis': perspective_analysis,
            'structural_bias': structural_bias
        }
    
    def audit_redundancy_decisions(self) -> Dict:
        """
        AUDITORÍA 2: ¿Las decisiones de redundancia favorecen fuentes colectivistas?
        """
        print("\n=== AUDITORÍA 2: DECISIONES DE REDUNDANCIA ===")
        
        redundancy_data = self.tga_results['methodology_enhancement']['redundancy_analysis']
        key_findings = self.tga_results['key_findings']['redundancy_findings']
        
        # Analizar la decisión AI Act vs GDPR
        redundant_pair = key_findings['redundant_pairs'][0]  # "ai_act_eu <-> gdpr_eu"
        redundancy_score = redundancy_data[redundant_pair]['composite_redundancy']
        
        # Clasificar orientación ideológica de cada fuente
        source_ideologies = {
            'ai_act_eu': {
                'regulatory_intensity': 'HIGH',
                'individual_vs_collective': 'COLLECTIVE_FOCUSED',
                'state_role': 'STRONG_INTERVENTION',
                'innovation_stance': 'PRECAUTIONARY'
            },
            'gdpr_eu': {
                'regulatory_intensity': 'HIGH', 
                'individual_vs_collective': 'INDIVIDUAL_FOCUSED',  # Derechos individuales datos
                'state_role': 'MODERATE_INTERVENTION',
                'innovation_stance': 'RIGHTS_PROTECTIVE'
            }
        }
        
        # ¿La recomendación (mantener AI Act, eliminar GDPR) tiene sesgo?
        recommended_keep = 'ai_act_eu'
        recommended_remove = 'gdpr_eu'
        
        bias_analysis = {
            'redundancy_score': redundancy_score,
            'decision': f"Keep {recommended_keep}, Remove {recommended_remove}",
            'kept_source_ideology': source_ideologies[recommended_keep],
            'removed_source_ideology': source_ideologies[recommended_remove],
            'potential_bias': self._analyze_redundancy_bias(source_ideologies, recommended_keep, recommended_remove)
        }
        
        print(f"Decisión redundancia: Mantener {recommended_keep}, Eliminar {recommended_remove}")
        print(f"Fuente mantenida: {bias_analysis['kept_source_ideology']['individual_vs_collective']}")
        print(f"Fuente eliminada: {bias_analysis['removed_source_ideology']['individual_vs_collective']}")
        
        if bias_analysis['potential_bias']['bias_detected']:
            print("⚠️ POSIBLE SESGO: Decisión favorece orientación colectivista")
        
        return bias_analysis
    
    def audit_bias_documentation(self) -> Dict:
        """
        AUDITORÍA 3: ¿La documentación de sesgos está sesgada hacia colectivismo?
        """
        print("\n=== AUDITORÍA 3: SESGO EN DOCUMENTACIÓN DE SESGOS ===")
        
        bias_doc = self.tga_results['methodology_enhancement']['bias_documentation']['potential_biases']
        
        # Analizar qué tipo de sesgos se documentan como "riesgo"
        overestimation_risks = bias_doc['overestimation_risks']
        underestimation_risks = bias_doc['underestimation_risks']
        
        # Clasificar fuentes por orientación
        source_orientations = {
            'art19_cn': 'INDIVIDUAL_RIGHTS',
            'sabsay_const': 'INDIVIDUAL_RIGHTS', 
            'gdpr_eu': 'INDIVIDUAL_RIGHTS',  # Derechos datos personales
            'ai_act_eu': 'COLLECTIVE_PROTECTION',  # Protección social IA
            'ley_datos_personales': 'MIXED',
            'ley_defensa_consumidor': 'COLLECTIVE_PROTECTION'  # Protección consumidores
        }
        
        # ¿Se considera "riesgo" sobreestimar fuentes individualistas?
        individual_sources_flagged = [
            source for source in overestimation_risks.keys() 
            if source_orientations.get(source) == 'INDIVIDUAL_RIGHTS'
        ]
        
        # ¿Se considera "riesgo" subestimar fuentes colectivistas?
        collective_sources_flagged = [
            source for source in underestimation_risks.keys()
            if source_orientations.get(source) == 'COLLECTIVE_PROTECTION'  
        ]
        
        bias_in_bias_doc = {
            'individual_sources_flagged_overestimation': individual_sources_flagged,  # ['art19_cn', 'sabsay_const']
            'collective_sources_flagged_underestimation': collective_sources_flagged,  # ['ai_act_eu'] 
            'pattern': 'Flags individual rights sources as overestimation risk, collective sources as underestimation risk',
            'meta_bias_detected': len(individual_sources_flagged) > 0 and len(collective_sources_flagged) > 0
        }
        
        print(f"Fuentes individualistas flaggeadas como 'riesgo sobreestimar': {individual_sources_flagged}")
        print(f"Fuentes colectivistas flaggeadas como 'riesgo subestimar': {collective_sources_flagged}")
        
        if bias_in_bias_doc['meta_bias_detected']:
            print("⚠️ META-SESGO DETECTADO: La 'corrección de sesgo' está sesgada hacia colectivismo")
        
        return bias_in_bias_doc
    
    def audit_consensus_interpretation(self) -> Dict:
        """
        AUDITORÍA 4: ¿El "consenso" se interpreta con sesgo colectivista?
        """
        print("\n=== AUDITORÍA 4: INTERPRETACIÓN DE CONSENSO ===")
        
        consensus_data = self.tga_results['methodology_enhancement']['multi_perspective_analysis']['consensus_analysis']
        
        high_consensus_sources = consensus_data['high_consensus']  # ['ley_defensa_consumidor', 'ley_datos_personales']
        divergent_sources = consensus_data['divergent']           # ['sabsay_const', 'gdpr_eu']
        
        # Clasificar por orientación ideológica
        source_orientations = {
            'ley_defensa_consumidor': 'COLLECTIVE_PROTECTION',
            'ley_datos_personales': 'MIXED_INDIVIDUAL_COLLECTIVE',
            'sabsay_const': 'INDIVIDUAL_CONSTITUTIONAL', 
            'gdpr_eu': 'INDIVIDUAL_DATA_RIGHTS'
        }
        
        # ¿Hay patrón ideológico en el "consenso"?
        consensus_orientation_analysis = {
            'high_consensus_orientations': [source_orientations[s] for s in high_consensus_sources],
            'divergent_orientations': [source_orientations[s] for s in divergent_sources],
            'pattern_detected': self._analyze_consensus_pattern(high_consensus_sources, divergent_sources, source_orientations)
        }
        
        print(f"Fuentes 'consensuales': {high_consensus_sources}")
        print(f"Orientaciones consensuales: {consensus_orientation_analysis['high_consensus_orientations']}")
        print(f"Fuentes 'divergentes': {divergent_sources}")
        print(f"Orientaciones divergentes: {consensus_orientation_analysis['divergent_orientations']}")
        
        if consensus_orientation_analysis['pattern_detected']['bias_toward_collective']:
            print("⚠️ SESGO DE CONSENSO: 'Consenso' favorece fuentes colectivistas")
        
        return consensus_orientation_analysis
    
    def _analyze_interpretation_bias(self, interpretation_text: str) -> Dict:
        """Analizar sesgo en texto de interpretación"""
        
        collective_keywords = ['regulación', 'protección', 'supervisión', 'control', 'ciudadanos']
        individual_keywords = ['libertad', 'autonomía', 'privada', 'individual', 'limitación poder']
        
        collective_count = sum(1 for keyword in collective_keywords if keyword in interpretation_text.lower())
        individual_count = sum(1 for keyword in individual_keywords if keyword in interpretation_text.lower())
        
        return {
            'collective_emphasis': collective_count,
            'individual_emphasis': individual_count,
            'net_bias': collective_count - individual_count,
            'bias_direction': 'COLLECTIVE' if collective_count > individual_count else 'INDIVIDUAL' if individual_count > collective_count else 'NEUTRAL'
        }
    
    def _analyze_redundancy_bias(self, source_ideologies: Dict, kept: str, removed: str) -> Dict:
        """Analizar si decisión de redundancia tiene sesgo ideológico"""
        
        kept_collective_focus = source_ideologies[kept]['individual_vs_collective'] == 'COLLECTIVE_FOCUSED'
        removed_individual_focus = source_ideologies[removed]['individual_vs_collective'] == 'INDIVIDUAL_FOCUSED'
        
        bias_detected = kept_collective_focus and removed_individual_focus
        
        return {
            'bias_detected': bias_detected,
            'bias_type': 'PRO_COLLECTIVE' if bias_detected else 'NEUTRAL',
            'explanation': f"Kept collective-focused source ({kept}), removed individual-focused source ({removed})" if bias_detected else "No clear ideological bias"
        }
    
    def _analyze_consensus_pattern(self, consensus_sources: List, divergent_sources: List, orientations: Dict) -> Dict:
        """Analizar patrón ideológico en consenso vs divergencia"""
        
        consensus_collective = sum(1 for s in consensus_sources 
                                 if 'COLLECTIVE' in orientations[s] or 'MIXED' in orientations[s])
        consensus_individual = sum(1 for s in consensus_sources 
                                 if 'INDIVIDUAL' in orientations[s] and 'MIXED' not in orientations[s])
        
        divergent_collective = sum(1 for s in divergent_sources 
                                 if 'COLLECTIVE' in orientations[s])
        divergent_individual = sum(1 for s in divergent_sources 
                                 if 'INDIVIDUAL' in orientations[s])
        
        return {
            'consensus_collective': consensus_collective,
            'consensus_individual': consensus_individual, 
            'divergent_collective': divergent_collective,
            'divergent_individual': divergent_individual,
            'bias_toward_collective': consensus_collective > consensus_individual and divergent_individual > divergent_collective
        }
    
    def generate_comprehensive_bias_audit(self) -> Dict:
        """Generar auditoría comprehensiva de sesgo colectivista"""
        
        print("="*80)
        print("AUDITORÍA CRÍTICA: ¿SESGO PRO-COLECTIVISTA EN TGA?")
        print("="*80)
        
        perspective_audit = self.audit_perspective_balance()
        redundancy_audit = self.audit_redundancy_decisions() 
        bias_doc_audit = self.audit_bias_documentation()
        consensus_audit = self.audit_consensus_interpretation()
        
        # Síntesis de hallazgos
        bias_indicators = {
            'structural_perspective_bias': perspective_audit['structural_bias']['bias_detected'],  # 2:1 pro-regulatorio
            'redundancy_decision_bias': redundancy_audit['potential_bias']['bias_detected'],      # Mantener colectivista
            'meta_bias_in_documentation': bias_doc_audit['meta_bias_detected'],                   # Sesgo en "corrección" 
            'consensus_interpretation_bias': consensus_audit['pattern_detected']['bias_toward_collective']  # Consenso = colectivista
        }
        
        total_bias_indicators = sum(bias_indicators.values())
        
        overall_assessment = {
            'total_bias_indicators': total_bias_indicators,
            'bias_severity': 'HIGH' if total_bias_indicators >= 3 else 'MODERATE' if total_bias_indicators >= 2 else 'LOW',
            'primary_bias_direction': 'PRO_COLLECTIVE_REGULATORY',
            'confidence_in_tga_neutrality': 'LOW' if total_bias_indicators >= 3 else 'MODERATE'
        }
        
        print(f"\n🚨 RESULTADO AUDITORÍA:")
        print(f"   Indicadores de sesgo detectados: {total_bias_indicators}/4")
        print(f"   Severidad del sesgo: {overall_assessment['bias_severity']}")
        print(f"   Dirección del sesgo: {overall_assessment['primary_bias_direction']}")
        print(f"   Confianza en neutralidad TGA: {overall_assessment['confidence_in_tga_neutrality']}")
        
        if total_bias_indicators >= 2:
            print("\n⚠️ CONCLUSIÓN: TGA introduces systematic pro-collectivist bias")
            print("   RECOMENDACIÓN: Aplicar correcciones anti-colectivistas o descartar TGA")
        
        return {
            'perspective_balance': perspective_audit,
            'redundancy_decisions': redundancy_audit,
            'bias_documentation': bias_doc_audit, 
            'consensus_interpretation': consensus_audit,
            'bias_indicators': bias_indicators,
            'overall_assessment': overall_assessment
        }

def main():
    """Ejecutar auditoría completa de sesgo colectivista"""
    
    auditor = CollectivistBiasAudit()
    audit_results = auditor.generate_comprehensive_bias_audit()
    
    # Guardar resultados auditoría
    with open('audit_sesgo_colectivista_tga.json', 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Auditoría completada")
    print(f"📄 Resultados guardados en: audit_sesgo_colectivista_tga.json")
    
    return audit_results

if __name__ == "__main__":
    results = main()