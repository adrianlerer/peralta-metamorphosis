# Methodology Documentation

## The Peralta Metamorphosis: Computational Methodology

This document provides detailed technical documentation for the computational methods used in "The Peralta Metamorphosis: Quantifying the Evolution of Legal Parasitism Through Computational Analysis of Argentine Constitutional Degradation (1922-2025)".

## Table of Contents

1. [JurisRank Algorithm](#jurisrank-algorithm)
2. [ABAN Genealogical Tracing](#aban-genealogical-tracing)
3. [Legal-Memespace Modeling](#legal-memespace-modeling)
4. [Data Sources and Preparation](#data-sources-and-preparation)
5. [Statistical Validation](#statistical-validation)

## JurisRank Algorithm

### Overview

JurisRank adapts the PageRank algorithm for legal citation networks, incorporating domain-specific factors that reflect the unique characteristics of legal precedent and doctrinal evolution.

### Mathematical Foundation

The core JurisRank equation is:

```
JR(i) = (1-d)/N + d * Σ[j∈M(i)] W(j,i) * JR(j) / C(j)
```

Where:
- `JR(i)` = JurisRank score of case i
- `d` = damping factor (default: 0.85)
- `N` = total number of cases
- `M(i)` = set of cases citing case i
- `W(j,i)` = weighted citation strength from case j to case i
- `C(j)` = total citation count from case j

### Weighting Components

#### 1. Temporal Weighting

Citations experience temporal decay to reflect the diminishing precedential value of older citations:

```
W_temporal(j,i) = exp(-λ * (date_j - date_i))
```

Where:
- `λ` = temporal decay parameter (default: 0.05 per year)
- `date_j - date_i` = time difference in years

#### 2. Hierarchical Weighting

Citations from higher courts receive greater weight:

```
W_hierarchical = {
    'Supreme Court': 1.0,
    'Appeals Court': 0.7,
    'Federal Court': 0.6,
    'Provincial Supreme': 0.5,
    'Lower Court': 0.4
}
```

#### 3. Doctrinal Clustering

Cases with similar doctrinal elements amplify each other's citations:

```
W_doctrinal(j,i) = 1 + α * similarity(doctrine_j, doctrine_i)
```

Where:
- `α` = clustering boost factor (default: 0.5)
- `similarity()` = Jaccard similarity between doctrinal element sets

### Implementation Notes

- **Convergence**: Algorithm iterates until change in scores < 0.0001
- **Dangling nodes**: Cases with no outgoing citations distribute probability uniformly
- **Normalization**: Final scores sum to 1 across all cases

## ABAN Genealogical Tracing

### Overview

The Ancestral Backward Analysis of Networks (ABAN) algorithm traces genealogical lineages through legal citation networks to identify precedential relationships and doctrinal inheritance patterns.

### Algorithm Steps

1. **Ancestor Identification**: For each case, identify all directly cited cases
2. **Primary Ancestor Selection**: Select the most precedentially important ancestor
3. **Inheritance Analysis**: Compare doctrinal elements between ancestor and descendant
4. **Mutation Classification**: Categorize the type of doctrinal change

### Primary Ancestor Selection

The primary ancestor is selected using a composite scoring function:

```
Score(ancestor) = 0.4 * citation_strength + 
                  0.3 * precedential_weight + 
                  0.2 * doctrinal_similarity + 
                  0.1 * temporal_proximity
```

### Inheritance Fidelity Calculation

```
Fidelity = |inherited_elements| / (|inherited_elements| + |mutated_elements|)
```

### Mutation Type Classification

- **Faithful** (fidelity ≥ 0.7, mutations = 0): Perfect doctrinal preservation
- **Conservative** (fidelity ≥ 0.7, mutations ≤ 20%): Minor refinements
- **Incremental** (fidelity ≥ 0.3, mutations ≤ 40%): Moderate evolution
- **Expansive** (fidelity ≥ 0.3, mutations > 40%): Significant additions
- **Transformative** (fidelity < 0.3, mutations ≤ 70%): Substantial change
- **Revolutionary** (fidelity < 0.3, mutations > 70%): Major doctrinal shift

## Legal-Memespace Modeling

### Overview

Legal-Memespace maps legal doctrines in multi-dimensional space and models their competitive dynamics using adapted Lotka-Volterra equations.

### Dimensional Mapping

Four primary dimensions capture the essential axes of legal doctrine evolution:

1. **State vs Individual** (Dimension 0): Balance between state authority and individual rights
2. **Emergency vs Normal** (Dimension 1): Degree of emergency powers and exceptional measures
3. **Formal vs Pragmatic** (Dimension 2): Approach to legal interpretation and constitutional construction
4. **Temporary vs Permanent** (Dimension 3): Temporal character of legal measures and institutions

### PCA Transformation

Case features are mapped to doctrinal space using Principal Component Analysis:

1. **Feature Extraction**: Convert case attributes to numerical features
2. **Standardization**: Scale features to zero mean, unit variance
3. **PCA Application**: Reduce dimensionality to 4 principal components
4. **Normalization**: Scale coordinates to [0,1] range for interpretability

### Competitive Dynamics Simulation

Doctrinal competition follows generalized Lotka-Volterra equations:

```
dN_i/dt = r_i * N_i * (1 - Σ(α_ij * N_j) / K_i)
```

Where:
- `N_i` = prevalence of doctrine i
- `r_i` = intrinsic growth rate of doctrine i
- `α_ij` = competition coefficient (effect of doctrine j on doctrine i)
- `K_i` = carrying capacity of doctrine i

### Phase Transition Detection

Phase transitions are identified using statistical change-point detection:

1. **Moving Window Analysis**: Calculate local doctrinal coordinates using sliding windows
2. **Distance Calculation**: Measure Euclidean distance between consecutive window means
3. **Peak Detection**: Identify local maxima in distance profile
4. **Significance Testing**: Apply Hotelling's T² test for statistical validation

## Data Sources and Preparation

### Case Selection Criteria

- **Temporal Scope**: 1922-2025 (Ercolano to present)
- **Court Level**: Focus on Supreme Court decisions with significant constitutional implications
- **Doctrinal Relevance**: Cases involving emergency powers, property rights, due process, state intervention
- **Citation Impact**: Cases with substantial citation networks and precedential influence

### Citation Network Construction

- **Directed Graph**: Citations flow from citing case to cited case
- **Edge Weights**: Based on citation frequency and strength in judicial opinions
- **Network Validation**: Ensure temporal consistency (no forward citations)
- **Completeness**: Include all significant precedential relationships

### Feature Engineering

Case attributes are converted to numerical features:

- **Doctrinal Elements**: Binary indicators for presence of specific legal principles
- **Court Metadata**: Hierarchical encoding of court levels
- **Temporal Factors**: Year normalization and economic crisis indicators
- **Outcome Variables**: Measures of formalist vs. emergency doctrine adoption

## Statistical Validation

### Significance Testing

- **Phase Transitions**: Hotelling's T² test for multivariate mean differences
- **Correlation Analysis**: Pearson correlation between fiscal impact and congressional rejection
- **Genealogical Dominance**: Binomial test for Peralta lineage prevalence

### Robustness Checks

- **Parameter Sensitivity**: Test algorithm stability across parameter ranges
- **Bootstrap Sampling**: Resample citation networks to assess result stability
- **Cross-Validation**: Temporal holdout validation for predictive accuracy

### Confidence Intervals

- **JurisRank Scores**: Bootstrap confidence intervals for fitness estimates
- **Phase Transition Timing**: Uncertainty bounds on transition date detection
- **Dominance Rates**: Binomial confidence intervals for genealogical percentages

## Implementation Details

### Computational Complexity

- **JurisRank**: O(k * N²) where k = iterations, N = number of cases
- **ABAN Tracing**: O(N * d * log N) where d = maximum genealogical depth
- **Memespace PCA**: O(N * p²) where p = number of features

### Performance Optimizations

- **Sparse Matrices**: Use sparse representation for citation networks
- **Caching**: Cache genealogical computations for repeated queries  
- **Vectorization**: Utilize NumPy vectorized operations for matrix computations
- **Parallel Processing**: Optional multi-core processing for large datasets

### Numerical Stability

- **Matrix Regularization**: Add small diagonal terms to prevent singularities
- **Convergence Monitoring**: Track iteration progress and detect numerical issues
- **Overflow Protection**: Use log-space computations for very small probabilities

## References

1. Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual Web search engine.
2. Fowler, J. H., et al. (2007). Network analysis and the law: Measuring the legal importance of precedents.
3. Lotka, A. J. (1925). Elements of physical biology.
4. Volterra, V. (1926). Fluctuations in the abundance of a species considered mathematically.
5. Hotelling, H. (1931). The generalization of Student's ratio.

---

*This methodology was developed for "The Peralta Metamorphosis" by Ignacio Adrián Lerer. For technical questions, contact: adrian@lerer.com.ar*