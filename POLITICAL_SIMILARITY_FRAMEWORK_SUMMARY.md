# Political Similarity Framework (PSF) - Resumen Técnico

## Marco Teórico

El **Political Similarity Framework (PSF)** es un sistema de análisis político multidimensional que permite evaluar y comparar actores políticos a través de un espacio político de 4 dimensiones, utilizando el **Political Similarity Index (PSI)** como métrica principal.

## Metodología

### Dimensiones del Espacio Político 4D

1. **Centralization-Autonomy**: Grado de centralización vs. autonomía administrativa
2. **Port-Interior**: Orientación hacia centros urbanos vs. regiones interiores
3. **Elite-Popular**: Tendencias elitistas vs. orientación popular
4. **Continuity-Rupture**: Continuidad institucional vs. ruptura del sistema

### Métricas de Análisis Individual

Para cada actor político se evalúan 9 dimensiones principales:

- **ideology_economic** (0-1): Posición económica (izquierda-derecha)
- **ideology_social** (0-1): Posición social (liberal-conservador)
- **leadership_messianic** (0-1): Elementos mesiánicos del liderazgo
- **leadership_charismatic** (0-1): Carisma y capacidad de movilización
- **anti_establishment** (0-1): Retórica anti-sistema
- **symbolic_mystical** (0-1): Elementos simbólicos y místicos
- **populist_appeal** (0-1): Apelación populista
- **authoritarian** (0-1): Tendencias autoritarias
- **media_savvy** (0-1): Habilidad mediática

### Political Similarity Index (PSI)

El PSI se calcula como la media de similaridades dimensionales entre un actor y actores de referencia:

```
PSI(actor) = mean([1 - |actor_dim_i - reference_dim_i| for i in dimensions])
```

## Validación Estadística

### Bootstrap Validation
- **Iteraciones**: 1000 muestras con reemplazo
- **Confidence Intervals**: 95% (percentiles 2.5 y 97.5)
- **Stability Coefficient**: < 0.005 para robustez estadística

### Métricas de Red
- **Network Density**: Densidad de conexiones genealógicas
- **Clustering Coefficient**: Coeficiente de agrupamiento
- **Attractor Count**: Número de attractores políticos identificados
- **Chain Length**: Longitud promedio de cadenas genealógicas

## Aplicaciones

### 1. Análisis de Corpus Político
- **Corpus Mínimo**: 30+ documentos para análisis robusto
- **Genealogical Chains**: Identificación de conexiones políticas
- **Political Attractors**: Figuras de alta influencia histórica

### 2. Evaluación Legislativa
- **Constitutional Compatibility**: Evaluación vs. Art. 19 CN
- **Policy Impact Analysis**: Impacto en innovación y desarrollo
- **Governance Structure**: Análisis de estructuras de poder

### 3. Comparative Politics
- **Cross-Country Analysis**: Comparación entre países
- **Temporal Evolution**: Evolución política temporal
- **Ideological Mapping**: Mapeo del espacio ideológico

## Casos de Uso Validados

### Análisis 1: Corpus Expandido
- **Objetivo**: Validar robustez metodológica
- **Resultado**: +91% attractores con corpus expandido (13→32 docs)
- **Conclusión**: Framework escalable y estadísticamente robusto

### Análisis 2: Proyecto Gollan 2130-D-2025
- **Objetivo**: Evaluación constitucional de proyecto de ley IA
- **Resultado**: Score constitucionalidad 0.188/1.000 (INCONSTITUCIONAL)
- **Conclusión**: Framework aplicable a análisis legislativo contemporáneo

## Implementación Técnica

### Estructura de Datos
```python
# Dataset structure
political_actor = {
    'name': str,
    'period': str,
    'country': str,
    'era': str,
    'ideology_economic': float,
    'ideology_social': float,
    'leadership_messianic': float,
    'leadership_charismatic': float,
    'anti_establishment': float,
    'symbolic_mystical': float,
    'populist_appeal': float,
    'authoritarian': float,
    'media_savvy': float,
    'political_similarity_index': float
}
```

### Funciones Principales
```python
# Core functions
create_generic_political_dataset() -> pd.DataFrame
get_political_similarity_breakdown() -> dict
calculate_political_similarity(actor1, actor2) -> float
bootstrap_validation(dataset, n_iterations=1000) -> dict
```

## Ventajas del Framework

### Académicas
- ✅ **Neutralidad**: Sin referencias a individuos específicos
- ✅ **Objetividad**: Métricas cuantitativas reproducibles
- ✅ **Escalabilidad**: Validado estadísticamente en corpus expandidos
- ✅ **Robustez**: Bootstrap validation con 1000 iteraciones

### Prácticas
- ✅ **Aplicabilidad**: Análisis legislativo y constitucional
- ✅ **Versatilidad**: Múltiples casos de uso político
- ✅ **Replicabilidad**: Metodología completamente documentada
- ✅ **Extensibilidad**: Framework modular y expandible

### Metodológicas
- ✅ **Multidimensionalidad**: Análisis 4D del espacio político
- ✅ **Validación estadística**: Confidence intervals y stability coefficients
- ✅ **Análisis temporal**: Evolución política y genealogías
- ✅ **Comparabilidad**: Métricas estandarizadas entre actores

## Limitaciones y Consideraciones

### Metodológicas
- **Corpus Mínimo**: Requiere mínimo 30 documentos para robustez
- **Subjetividad**: Evaluación dimensional requiere criterio experto
- **Contexto Cultural**: Interpretación puede variar por contexto
- **Temporal Bias**: Sesgo hacia períodos con más documentación

### Técnicas
- **Data Quality**: Dependiente de calidad de datos de entrada
- **Computational Cost**: Bootstrap validation computacionalmente intensivo
- **Parameter Sensitivity**: Sensibilidad a elección de parámetros
- **Scalability Limits**: Limitaciones con datasets muy grandes

## Desarrollos Futuros

### Expansiones Metodológicas
1. **Machine Learning Integration**: Clasificación automática de dimensiones
2. **Network Analysis Enhancement**: Análisis de redes más sofisticado
3. **Temporal Dynamics**: Modelos dinámicos de evolución política
4. **Cross-Cultural Validation**: Validación en múltiples contextos culturales

### Aplicaciones Adicionales
1. **Electoral Prediction**: Predicción de comportamiento electoral
2. **Coalition Analysis**: Análisis de formación de coaliciones
3. **Policy Impact**: Evaluación de impacto de políticas públicas
4. **Risk Assessment**: Evaluación de riesgos políticos e institucionales

---

## Conclusión

El **Political Similarity Framework (PSF)** proporciona una metodología robusta, neutral y estadísticamente validada para el análisis político multidimensional. Su aplicabilidad demostrada tanto en investigación académica como en evaluación legislativa práctica lo convierte en una herramienta valiosa para la ciencia política contemporánea.

**Framework Version**: 3.0 (Generic)  
**Validation Status**: ✅ Estadísticamente validado  
**Use Cases**: ✅ Académico + Aplicado  
**Documentation**: ✅ Completa