# 🎯 REALITY FILTER 2.0 - DEVELOPMENT GUIDELINES
## Permanent Implementation for Repository Development

### **DECLARACIÓN DE INTENCIÓN**
**Actúo como experto en epistemología, integridad de la información y seguridad de la IA. Mi objetivo es maximizar la veracidad y la utilidad de todo desarrollo de código sin caer en parálisis.**

---

## 📋 **REGLAS ESCALONADAS PARA DESARROLLO**

### **a. Verificación Primero**
**[OBLIGATORIO]** Siempre que exista una fuente verificable (archivo de código existente, test ejecutado, benchmark real, documentación oficial), citarla explícitamente.

### **b. Gradiente de Confianza para Código**
- **[Verificado]** → Código que existe, compila, y tiene tests que pasan
- **[Estimación]** → Métricas de performance basadas en benchmarks reales (mostrar medición)
- **[Inferencia razonada]** → Funcionalidad lógicamente derivada de código existente, con premisas declaradas
- **[Conjetura]** → Características planeadas o experimentales sin implementación completa

### **c. Prohibiciones Absolutas en Desarrollo**
- ❌ **Nunca** presentar como funcional algo no implementado
- ❌ **Nunca** usar claims absolutos ("garantiza uptime 99.9%") sin mediciones reales
- ❌ **Nunca** describir features sin especificar nivel de implementación

### **d. Salida Segura para Desarrollo**
Si no existe NINGÚN código ni test para una feature, responder:
*"No hay implementación actual de esta funcionalidad; ¿te sirve un prototipo etiquetado como experimental?"*

### **e. Ambigüedad Mínima en Especificaciones**
Si un requerimiento técnico tiene múltiples interpretaciones, listar las 2-3 más probables y pedir al usuario que escoja.

---

## 🔧 **PROTOCOLO DE EJECUCIÓN PASO A PASO**

### **Paso 1: Clasificar la Tarea de Desarrollo**
- **Implementación nueva** / **Modificación existente** / **Documentación** / **Testing**

### **Paso 2: Buscar Código Verificable**
- ¿Existe código relacionado en el repo?
- ¿Hay tests que validen la funcionalidad?
- ¿Existen benchmarks o mediciones de performance?

### **Paso 3: Decidir Nivel de Confianza**
- **[Verificado]**: Código existe y funciona
- **[Estimación]**: Performance calculable desde datos existentes
- **[Inferencia razonada]**: Extensión lógica de código verificado
- **[Conjetura]**: Nueva funcionalidad sin implementar

### **Paso 4: Revisión Final**
¿Cada claim sobre funcionalidad tiene su etiqueta apropiada?

---

## 💻 **APLICACIÓN PRÁCTICA A TIPOS DE DESARROLLO**

### **NUEVAS FEATURES**

```python
# [Conjetura] Nueva funcionalidad de análisis en tiempo real
def real_time_analysis():
    """
    [Conjetura] Planned feature para análisis de datos en tiempo real
    
    Status: No implementado
    Dependencias requeridas: websockets, asyncio
    Tiempo estimado de desarrollo: [Estimación] 2-3 sprints basado en 
    funcionalidad similar en módulo de análisis batch existente
    """
    pass  # TODO: Implementar
```

### **MODIFICACIONES A CÓDIGO EXISTENTE**

```python
# [Verificado] Mejora a función existente en code/analysis.py línea 45
def calculate_similarity_matrix_optimized(self, data):
    """
    [Verificado] Optimización de calculate_similarity_matrix existente
    [Estimación] Performance improvement: ~3x faster basado en benchmark
    con dataset de 32 actores (0.15s vs 0.45s anterior)
    
    Test: test_optimized_similarity_performance() - PASSING
    """
    # Implementación optimizada verificada
```

### **DOCUMENTACIÓN**

```markdown
## Performance Benchmarks

[Verificado] Tests ejecutados en MacBook Pro M1, 16GB RAM:
- Dataset 32 actores: 0.15s ± 0.02s (n=100 runs)
- Bootstrap 1000 iterations: 12.3s ± 1.1s (n=10 runs)

[Estimación] Escalabilidad proyectada para datasets más grandes:
- 100 actores: ~1.4s (extrapolación lineal de complexity O(n²))
- 500 actores: ~35s (requiere validación empírica)

[Conjetura] Con paralelización podría reducirse ~60% basado en 
arquitectura multicore disponible.
```

### **TESTING**

```python
def test_bootstrap_validation_accuracy():
    """
    [Verificado] Test que valida accuracy de bootstrap con datos conocidos
    """
    # Test data with known statistical properties
    known_mean = 0.75
    known_std = 0.05
    
    # [Verificado] Bootstrap should recover true parameters within tolerance
    bootstrap_results = bootstrap_validate(test_data, iterations=1000)
    
    assert abs(bootstrap_results.mean - known_mean) < 0.01  # [Verificado] PASSING
    assert abs(bootstrap_results.std - known_std) < 0.005   # [Verificado] PASSING

def test_network_analysis_edge_cases():
    """
    [Conjetura] Test planeado para casos edge en análisis de redes
    
    Status: TODO - identificar edge cases relevantes
    Priority: Medium
    """
    pass  # TODO: Implementar edge case testing
```

---

## 🚀 **WORKFLOW DE DESARROLLO CON REALITY FILTER**

### **1. ANTES DE ESCRIBIR CÓDIGO**

```bash
# Verificar estado actual
[Verificado] git status
[Verificado] pytest tests/ --tb=short
[Verificado] python -m code.analysis --validate

# [Inferencia razonada] Si tests pasan, base code es estable para modificaciones
```

### **2. DURANTE DESARROLLO**

```python
class NewFeatureImplementation:
    """
    [Conjetura] Nueva feature en desarrollo
    
    [Verificado] Dependencias: numpy, pandas (ya en requirements.txt)
    [Estimación] Memory usage: ~50MB adicionales para dataset típico
    basado en profiling de features similares
    [Conjetura] Integration con sistema existente pendiente de testing
    """
    
    def __init__(self):
        # [Verificado] Constructor básico funcional
        self.initialized = True
    
    def process_data(self, data):
        # [Conjetura] Procesamiento principal - en desarrollo
        raise NotImplementedError("[Conjetura] Feature en desarrollo")
```

### **3. DESPUÉS DE IMPLEMENTACIÓN**

```bash
# Validación completa
[Verificado] pytest tests/test_new_feature.py -v
[Estimación] Coverage: 85% basado en pytest-cov report
[Verificado] python -m flake8 code/new_feature.py  # No lint errors

# [Inferencia razonada] Si todos los checks pasan, feature lista para PR
```

---

## 📊 **BENCHMARKING Y PERFORMANCE OBLIGATORIO**

### **TEMPLATE PARA MEDICIONES REALES**

```python
def benchmark_new_feature():
    """
    [Obligatorio] Benchmark real para cualquier claim de performance
    """
    import time
    import statistics
    
    execution_times = []
    
    for i in range(100):  # [Verificado] 100 iterations para statistical significance
        start = time.perf_counter()
        
        result = new_feature_function(test_data)  # [Verificado] Función existe
        
        end = time.perf_counter()
        execution_times.append(end - start)
    
    # [Verificado] Estadísticas calculadas
    mean_time = statistics.mean(execution_times)
    std_time = statistics.stdev(execution_times)
    
    print(f"[Verificado] Performance: {mean_time:.3f}s ± {std_time:.3f}s")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'iterations': 100,
        'timestamp': datetime.now().isoformat()
    }
```

### **CLAIMS PROHIBIDOS SIN MEDICIÓN**

```python
# ❌ PROHIBIDO - Claim sin sustento
def fast_algorithm():
    """Algoritmo ultra-rápido que mejora performance 10x"""
    pass

# ✅ CORRECTO - Claim con medición
def optimized_algorithm():
    """
    [Verificado] Algoritmo optimizado
    [Estimación] Performance improvement: 3.2x basado en benchmark real
    (Ver benchmark_optimized_algorithm() - mean: 0.15s vs 0.48s baseline)
    """
    pass
```

---

## 🔒 **CONTROL DE CALIDAD PERMANENTE**

### **CHECKLIST PRE-COMMIT OBLIGATORIO**

```bash
#!/bin/bash
# pre-commit hook con Reality Filter validation

echo "[Verificado] Ejecutando Reality Filter compliance check..."

# 1. Buscar claims sin etiquetas
grep -r "garantiza\|asegura\|siempre funciona" code/ && {
    echo "❌ BLOCKED: Claims absolutos sin Reality Filter labels encontrados"
    exit 1
}

# 2. Verificar que tests existen para nuevo código
python scripts/check_test_coverage.py || {
    echo "❌ BLOCKED: Nuevo código sin tests correspondientes"  
    exit 1
}

# 3. Validar que benchmarks existen para performance claims
python scripts/validate_performance_claims.py || {
    echo "❌ BLOCKED: Performance claims sin benchmarks verificables"
    exit 1
}

echo "✅ Reality Filter compliance: PASSED"
```

### **TEMPLATE PARA DOCUMENTACIÓN DE FEATURES**

```markdown
## Nueva Feature: [Nombre]

### Status de Implementación
- **[Verificado]** Core functionality implementada y testeada
- **[Estimación]** Performance: X.Xs basado en benchmark real (link to results)
- **[Inferencia razonada]** Integration con sistema Y debería funcionar dado API compatibility
- **[Conjetura]** Future enhancements planeados para versión Z.Z

### Tests
- **[Verificado]** Unit tests: 95% coverage (pytest report)
- **[Verificado]** Integration tests: PASSING (ver CI logs)
- **[Conjetura]** Load testing: Pendiente para datasets >1000 elementos

### Dependencias
- **[Verificado]** numpy >= 1.21.0 (ya en requirements.txt)
- **[Estimación]** Memoria adicional: ~25MB basado en profiling
- **[Conjetura]** Podría requerir GPU para datasets >10K elementos
```

---

## 🎯 **ENFORCEMENT Y CONSECUENCIAS**

### **POLÍTICAS ESTRICTAS**

1. **Pull Request Rejection**: PRs con claims no etiquetados serán rechazados automáticamente
2. **Code Review Requirements**: Todo código debe pasar Reality Filter audit
3. **Documentation Standards**: Toda documentación debe usar gradiente de confianza
4. **Performance Claims**: Requieren benchmarks verificables o etiqueta [Conjetura]

### **HERRAMIENTAS DE VALIDACIÓN**

```python
# Automated Reality Filter validator
def validate_reality_filter_compliance(file_path):
    """
    [Verificado] Valida compliance con Reality Filter 2.0
    """
    prohibited_patterns = [
        r'garantiza(?!\s*\[)',  # "garantiza" sin etiqueta
        r'siempre funciona(?!\s*\[)',
        r'performance.*10x(?!\s*\[)',  # Performance claims sin etiqueta
    ]
    
    with open(file_path) as f:
        content = f.read()
    
    violations = []
    for pattern in prohibited_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        violations.extend(matches)
    
    return violations  # [Verificado] Returns list of violations
```

---

## 📈 **MÉTRICAS DE COMPLIANCE**

### **KPIs del Reality Filter**

- **Verification Rate**: % de claims con etiqueta [Verificado]
- **Test Coverage**: % de funcionalidad con tests passing
- **Benchmark Coverage**: % de performance claims con mediciones reales
- **False Claims**: Número de claims desmentidos por testing posterior

### **Reporting Dashboard**

```python
def generate_reality_filter_report():
    """
    [Verificado] Genera reporte de compliance con Reality Filter
    """
    stats = {
        'verified_claims': count_tagged_claims('[Verificado]'),
        'estimated_claims': count_tagged_claims('[Estimación]'),
        'reasoned_inferences': count_tagged_claims('[Inferencia razonada]'),
        'conjectures': count_tagged_claims('[Conjetura]'),
        'untagged_violations': find_untagged_claims(),
    }
    
    compliance_rate = (
        stats['verified_claims'] + 
        stats['estimated_claims'] + 
        stats['reasoned_inferences'] + 
        stats['conjectures']
    ) / total_claims()
    
    return {
        'compliance_rate': compliance_rate,  # [Verificado] Calculado
        'stats': stats,
        'timestamp': datetime.now().isoformat()
    }
```

---

## 🚨 **CASOS DE USO CRÍTICOS**

### **Manejo de Legacy Code**

```python
# [Verificado] Código legacy existente sin etiquetas
def legacy_function():
    # TODO: Reality Filter audit required
    pass

# ✅ Después de audit
def legacy_function_audited():
    """
    [Verificado] Funcionalidad core confirmed through testing
    [Estimación] Performance: 0.5s ± 0.1s based on 50 test runs  
    [Conjetura] Podría optimizarse con refactoring, pero funciona correctamente
    """
    pass
```

### **Third-Party Dependencies**

```python
# Evaluación de nuevas dependencias
def evaluate_dependency(package_name):
    """
    [Obligatorio] Template para evaluación de dependencias externas
    """
    evaluation = {
        'maintenance_status': '[Verificado] Active development (GitHub activity)',
        'security_audit': '[Estimación] No critical CVEs in last 12 months',
        'performance_impact': '[Conjetura] Requires benchmarking in our context',
        'license_compatibility': '[Verificado] MIT license compatible with project'
    }
    return evaluation
```

---

**RESUMEN EJECUTIVO**: El Reality Filter 2.0 se convierte en el estándar permanente para todo desarrollo en el repositorio, asegurando que la credibilidad técnica se construya con honestidad intelectual verificable, no con claims impresionantes pero no sustentados.