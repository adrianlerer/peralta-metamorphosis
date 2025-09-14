# Universal Analysis Framework

Un marco de análisis universal que implementa 8 meta-principios aplicables a cualquier dominio de análisis, desde texto y finanzas hasta análisis científico y toma de decisiones empresariales.

## 🎯 Meta-Principios Universales

### 1. **Marco de Abstención Matemática**
- Abstención inteligente con garantías estadísticas rigurosas
- Múltiples métodos de cálculo de límites (Hoeffding, Bootstrap, Bayesian, etc.)
- Decisiones de abstención automáticas cuando la confianza es insuficiente

### 2. **Límites de Confianza en Decisiones**
- Intervalos de confianza con múltiples métodos estadísticos
- Bootstrap BCa, Clopper-Pearson, Wilson Score, y más
- Evaluación automática de riesgo y calidad de límites

### 3. **Análisis Genealógico de Influencias**
- Rastreo completo de dependencias y orígenes de decisiones
- Grafo dirigido de influencias con métricas de centralidad
- Identificación de influencias críticas y caminos de ancestría

### 4. **Pipeline Multi-Etapa con Validación**
- Procesamiento estructurado en etapas bien definidas
- Validación en múltiples puntos del pipeline
- Trazabilidad completa del proceso de análisis

### 5. **Evaluación Ensemble Multi-Modelo**
- Combinación inteligente de múltiples modelos/enfoques
- Estrategias adaptativas de combinación (promedio, voto mayoritario, etc.)
- Métricas de consenso y diversidad entre modelos

### 6. **Cuantificación de Incertidumbre**
- Medición rigurosa de incertidumbre epistémica y aleatórica
- Análisis de variabilidad para datos numéricos, categóricos y genéricos
- Métricas de confianza contextualmente apropiadas

### 7. **Salida Estructurada con Metadatos**
- Resultados completamente estructurados y serializables
- Metadatos completos incluidos automáticamente
- Trazabilidad temporal y de versiones

### 8. **Hibridación Adaptativa**
- Combinación dinámica de componentes según el contexto
- Selección automática de mejores enfoques por situación
- Aprendizaje continuo de rendimiento histórico

## 🏗️ Arquitectura

```
universal_analysis_framework/
├── core/                          # Framework principal
│   └── universal_framework.py     # Clases base y registry
├── mathematical/                  # Meta-principio 1: Abstención matemática  
│   └── abstention_framework.py    # Límites de confianza y abstención
├── ensemble/                      # Meta-principio 5: Evaluación ensemble
│   └── multi_model_evaluator.py   # Sistema ensemble multi-modelo
├── genealogical/                  # Meta-principio 3: Análisis genealógico
│   └── influence_tracker.py       # Rastreo de influencias y dependencias
├── hybridization/                 # Meta-principio 8: Hibridación adaptativa
│   └── adaptive_hybridizer.py     # Sistema de hibridación dinámica
├── domains/                       # Ejemplos de implementación
│   ├── text_analysis_example.py   # Análisis de texto
│   └── financial_analysis_example.py # Análisis financiero
├── integration/                   # API IntegridAI Suite
│   └── integrid_api.py            # API REST para integración
└── tests/                         # Tests unitarios
```

## 🚀 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Instalar el framework
pip install -e .
```

## 📋 Dependencias

- `numpy` - Computación numérica
- `scipy` - Métodos estadísticos avanzados
- `networkx` - Análisis de grafos genealógicos
- `fastapi` - API REST IntegridAI
- `uvicorn` - Servidor ASGI
- `pydantic` - Validación de datos

## 🎯 Uso Básico

### Análisis de Texto

```python
from universal_analysis_framework.domains.text_analysis_example import SimpleTextAnalyzer

# Crear analizador
analyzer = SimpleTextAnalyzer()

# Analizar texto
text = "This is an amazing example of universal analysis framework."
result = analyzer.analyze(text)

print(f"Confianza: {result.confidence:.3f}")
print(f"Abstención: {result.abstained}")
print(f"Resultado: {result.result}")
```

### Análisis Financiero

```python
from universal_analysis_framework.domains.financial_analysis_example import FinancialAnalyzer, FinancialData

# Crear analizador financiero
analyzer = FinancialAnalyzer()

# Datos financieros
data = FinancialData(
    symbol="AAPL",
    prices=[150, 152, 148, 155, 160],  # Precios históricos
    volumes=[1000000, 1100000, 900000, 1200000, 1300000],
    market_cap=2.5e12,  # $2.5T
    pe_ratio=25.5,
    roe=0.20
)

# Analizar
result = analyzer.analyze(data)
print(f"Recomendación: {result.result.investment_recommendation}")
print(f"Riesgo: {result.result.risk_score:.3f}")
```

### Uso del Framework Universal Directamente

```python
from universal_analysis_framework.core.universal_framework import UniversalAnalyzer, universal_registry

# Implementar analizador personalizado
class CustomAnalyzer(UniversalAnalyzer[str, dict]):
    def preprocess_input(self, input_data: str) -> dict:
        return {"text": input_data, "length": len(input_data)}
    
    def extract_features(self, preprocessed_data: dict) -> dict:
        return {"word_count": len(preprocessed_data["text"].split())}
    
    def perform_core_analysis(self, features: dict) -> dict:
        return {"analysis_result": "completed", "features": features}
    
    def calculate_confidence_metrics(self, result: dict, features: dict) -> dict:
        return {"base_confidence": 0.85}
    
    def perform_genealogical_analysis(self, input_data: str, result: dict) -> dict:
        return {"genealogy": "tracked"}

# Usar analizador
analyzer = CustomAnalyzer("custom_domain")
result = analyzer.analyze("Sample input")
```

## 🔧 API IntegridAI Suite

### Iniciar Servidor

```bash
cd universal_analysis_framework/integration/
python integrid_api.py
```

La API estará disponible en `http://localhost:8000`

### Endpoints Principales

#### Análisis Universal
```bash
POST /analyze
{
    "domain": "text_analysis",
    "input_data": "Text to analyze",
    "confidence_threshold": 0.80
}
```

#### Límites Matemáticos
```bash
POST /mathematical/bounds
{
    "data": [0.85, 0.90, 0.75, 0.88],
    "methods": ["bootstrap_percentile", "hoeffding"],
    "confidence_level": 0.95
}
```

#### Evaluación Ensemble
```bash
POST /ensemble/evaluate
{
    "input_data": "Data to evaluate",
    "strategy": "confidence_weighted"
}
```

#### Hibridización Adaptativa
```bash
POST /hybridization/analyze
{
    "domain": "financial_analysis",
    "data_characteristics": {"complexity": "high"},
    "performance_requirements": {"accuracy": 0.85}
}
```

### Documentación Interactiva

Accede a `http://localhost:8000/docs` para la documentación interactiva de Swagger/OpenAPI.

## 🧪 Ejemplos Avanzados

### Crear Modelo Ensemble Personalizado

```python
from universal_analysis_framework.ensemble.multi_model_evaluator import UniversalModel, ModelType

class MyCustomModel(UniversalModel):
    def __init__(self):
        super().__init__("my_model", ModelType.MACHINE_LEARNING)
    
    def predict(self, input_data):
        # Tu lógica de predicción
        result = {"prediction": "custom_result"}
        confidence = 0.85
        return result, confidence
    
    def get_metadata(self):
        return {"version": "1.0", "type": "custom"}

# Usar en ensemble
from universal_analysis_framework.ensemble.multi_model_evaluator import universal_ensemble_evaluator

model = MyCustomModel()
universal_ensemble_evaluator.add_model(model)
result = universal_ensemble_evaluator.evaluate("test data")
```

### Configurar Hibridización Personalizada

```python
from universal_analysis_framework.hybridization.adaptive_hybridizer import universal_adaptive_hybridizer, ComponentType

# Añadir componente personalizado
def my_analysis_function(data):
    return {"processed": data}, 0.9  # resultado, confianza

universal_adaptive_hybridizer.add_function_component(
    component_id="my_component",
    component_type=ComponentType.ALGORITHM,
    function=my_analysis_function,
    performance_metrics={"accuracy": 0.85},
    context_suitability={"my_domain": 0.95},
    reliability_score=0.90
)
```

### Análisis Genealógico Detallado

```python
from universal_analysis_framework.genealogical.influence_tracker import UniversalInfluenceTracker, InfluenceType

tracker = UniversalInfluenceTracker("my_analysis")

# Rastrear procesamiento
input_id, process_id, output_id = tracker.track_processing_step(
    "data_processing",
    "raw_data", 
    "processed_data",
    "custom_processing_function"
)

# Analizar genealogía
genealogy = tracker.analyze_genealogy()
critical_influences = tracker.find_critical_influences()
```

## 🧮 Métodos Matemáticos Soportados

### Cálculo de Límites de Confianza

- **Hoeffding**: Para datos acotados con garantías PAC
- **Bootstrap Percentil**: Remuestreo para distribuciones generales  
- **Bootstrap BCa**: Bias-corrected and accelerated bootstrap
- **Clopper-Pearson**: Límites exactos para proporciones binomiales
- **Wilson Score**: Límites para proporciones con mejor cobertura
- **Bayesian Credible**: Intervalos creíbles bayesianos
- **Ensemble Variance**: Basado en varianza de ensemble de modelos

### Estrategias de Ensemble

- **Simple Average**: Promedio simple de resultados
- **Weighted Average**: Promedio ponderado por pesos específicos
- **Majority Vote**: Voto mayoritario para resultados categóricos
- **Confidence Weighted**: Ponderación por confianza de cada modelo
- **Dynamic Selection**: Selección dinámica basada en múltiples criterios

### Estrategias de Hibridización

- **Context Adaptive**: Adaptación basada en contexto del análisis
- **Performance Based**: Selección basada en métricas de rendimiento
- **Confidence Driven**: Priorización por niveles de confianza
- **Data Dependent**: Adaptación según características de los datos

## 📊 Métricas y Monitoreo

El framework incluye métricas completas:

- **Confianza**: Múltiples métricas de confianza por componente
- **Incertidumbre**: Cuantificación rigurosa de incertidumbre
- **Consenso**: Métricas de acuerdo entre modelos
- **Diversidad**: Medidas de diversidad en ensemble
- **Centralidad**: Métricas de importancia en grafo genealógico
- **Rendimiento**: Tiempos de procesamiento y eficiencia

## 🔬 Casos de Uso

### 1. **Análisis de Texto y NLP**
- Análisis de sentimiento con abstención inteligente
- Extracción de temas con ensemble de métodos
- Análisis de calidad textual con límites de confianza

### 2. **Análisis Financiero y Riesgo**
- Recomendaciones de inversión con evaluación de riesgo
- Análisis técnico y fundamental hibridizado
- Evaluación de portafolios con incertidumbre cuantificada

### 3. **Análisis Científico**
- Combinación de múltiples metodologías experimentales
- Evaluación de significancia con múltiples tests
- Trazabilidad completa de decisiones científicas

### 4. **Toma de Decisiones Empresariales**
- Análisis de datos de negocio con múltiples perspectivas
- Evaluación de riesgo empresarial con abstención
- Combinación de análisis cuantitativo y cualitativo

### 5. **Sistemas de Recomendación**
- Hibridización de algoritmos colaborativos y de contenido
- Abstención cuando la confianza es insuficiente
- Análisis genealógico de factores de recomendación

## 🔮 Extensibilidad

El framework está diseñado para máxima extensibilidad:

### Nuevos Dominios
```python
# Implementar UniversalAnalyzer para tu dominio
class MyDomainAnalyzer(UniversalAnalyzer[InputType, OutputType]):
    # Implementar métodos abstractos
    pass

# Registrar en el sistema
universal_registry.register_analyzer("my_domain", MyDomainAnalyzer())
```

### Nuevos Métodos Matemáticos
```python
# Extender framework matemático
class CustomBoundCalculator:
    def calculate_custom_bound(self, data, confidence_level):
        # Tu lógica personalizada
        pass
```

### Nuevos Componentes de Hibridización
```python
# Añadir componentes al hibridizador
universal_adaptive_hybridizer.add_component(
    component_id="new_component",
    component_type=ComponentType.ALGORITHM,
    implementation=my_function,
    # ... parámetros de configuración
)
```

## 🤝 Integración con IntegridAI Suite

Este framework está diseñado para integrarse perfectamente con la suite IntegridAI:

- **API REST** estándar para todas las funciones
- **Formato de datos** compatible con otros componentes
- **Metadatos estructurados** para trazabilidad
- **Escalabilidad horizontal** con múltiples instancias
- **Monitoreo integrado** con métricas estándar

## 📈 Rendimiento

### Optimizaciones Incluidas

- **Ejecución paralela** de modelos ensemble
- **Cache inteligente** de resultados intermedios
- **Lazy loading** de componentes pesados
- **Límites de timeout** configurables
- **Pool de threads** para concurrencia

### Escalabilidad

- Diseño **stateless** para escalabilidad horizontal
- **Separación de concerns** entre componentes
- **APIs asíncronas** para alto throughput
- **Configuración flexible** de recursos

## 🐛 Debugging y Logging

```python
import logging

# Configurar logging detallado
logging.basicConfig(level=logging.DEBUG)

# Los componentes incluyen logging automático:
# - Decisiones de abstención
# - Resultados de ensemble  
# - Influencias genealógicas
# - Selecciones de hibridización
```

## ⚠️ Consideraciones de Producción

### Seguridad
- Validación estricta de entrada con Pydantic
- Limits de rate limiting recomendados  
- Sanitización automática de datos

### Reliability  
- Manejo robusto de errores en todos los componentes
- Timeouts configurables para evitar bloqueos
- Abstención automática ante datos insuficientes

### Monitoring
- Métricas completas de rendimiento incluidas
- Historial de análisis para auditoría
- Endpoints de health check

## 🧪 Testing

```bash
# Ejecutar tests
pytest universal_analysis_framework/tests/

# Tests de integración
pytest universal_analysis_framework/tests/integration/

# Tests de rendimiento  
pytest universal_analysis_framework/tests/performance/
```

## 📝 Licencia

Este framework está diseñado para uso en la suite IntegridAI. Consulta términos de uso específicos para implementación empresarial.

## 🤖 Contribuciones

Para contribuir nuevos dominios, métodos matemáticos o componentes de hibridización:

1. Implementa las interfaces abstractas correspondientes
2. Añade tests unitarios completos  
3. Documenta el nuevo componente
4. Integra con el sistema de registry existente

## 📞 Soporte

- **Documentación API**: `http://localhost:8000/docs` 
- **Health Check**: `http://localhost:8000/health`
- **Estadísticas**: `http://localhost:8000/stats`

---

**Universal Analysis Framework** - Aplicando principios universales de análisis a cualquier dominio con garantías matemáticas, trazabilidad completa y hibridización adaptativa.