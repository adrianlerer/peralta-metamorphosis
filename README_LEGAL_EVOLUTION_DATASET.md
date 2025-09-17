# Argentine Legal Evolution Dataset
## Empirical Foundation for "The Extended Phenotype of Law" Study

### Project Overview

This repository contains a comprehensive dataset documenting legal evolution in Argentina (1950-2024), designed to test theories of cumulative and artificial selection in jurisprudential systems. The dataset provides empirical evidence for understanding how legal norms evolve, mutate, and adapt in high-volatility institutional environments.

### Theoretical Framework

Based on the Extended Phenotype theory applied to legal systems, this dataset captures:

- **Cumulative Selection**: Gradual legal adaptations building on previous norms
- **Artificial Selection**: Deliberate legislative interventions and reforms  
- **Environmental Pressures**: Economic crises, technological changes, social transformations
- **Legal Transplants**: Success/failure patterns of foreign norm adoption
- **Mutation Rates**: Speed of legal change compared to institutional stability

### Repository Structure

```
/data
├── evolution_cases.csv           # Primary legal evolution instances
├── velocity_metrics.csv          # Speed of legal change measurements
├── transplants_tracking.csv      # Foreign legal instrument adoption
├── crisis_periods.csv           # Crisis-accelerated legal changes
├── innovations_exported.csv      # Argentine legal innovations adopted elsewhere
└── metadata/
    ├── normative_sources.csv    # Legal source tracking
    ├── actors_mapping.csv       # Key institutional actors
    └── classification_schema.csv # Evolution type taxonomy

/documentation
├── codebook.md                  # Variable definitions and coding rules
├── methodology.md               # Data collection and validation methods
├── sources.md                   # Bibliography and primary sources
└── validation_notes.md         # Quality control documentation

/analysis
├── descriptive_statistics.R     # Basic dataset statistics
├── evolutionary_metrics.py      # Custom velocity and selection metrics
├── visualizations.py           # Timeline and network visualizations
└── comparative_analysis.R      # Cross-jurisdictional comparisons

/case_studies
├── fideicomiso_evolution/      # Trust law development (1995-2024)
├── convertibility_mutations/   # Currency regime adaptations
├── amparo_constitutionalization/ # Constitutional remedy evolution
└── fintech_adaptation/         # Technology-driven legal changes
```

### Key Variables and Measurements

#### Evolution Velocity Metrics
- **Days to First Mutation**: Time between norm introduction and first evasion/adaptation
- **Regulatory Response Time**: Average time between market innovation and formal regulation
- **Crisis Acceleration Factor**: Speed multiplier during institutional crises
- **Survival Rate**: Percentage of norms surviving 5/10/15 year periods

#### Selection Pressure Indicators  
- **Economic Pressure Index**: Inflation, devaluation, fiscal crisis severity
- **Technological Disruption Level**: Digital transformation impact on legal frameworks
- **Social Demand Intensity**: Public pressure for legal reform (media mentions, protests)
- **International Compliance Pressure**: OECD, IMF, World Bank reform requirements

#### Success/Failure Classification
- **Transplant Success Rate**: Effectiveness of foreign legal instrument adoption
- **Endogenous Innovation Index**: Development of Argentina-specific legal solutions
- **Institutional Persistence Score**: Long-term survival of legal innovations
- **Cross-Border Diffusion**: Adoption of Argentine legal innovations by other countries

### Major Case Studies Included

#### 1. Financial Instruments Evolution
- **Fideicomiso Trajectory** (1995-2024): From Ley 24.441 to tokenization
- **Contractual Adjustment Clauses**: Prohibition (1991) → CER (2002) → UVA (2016) → Crypto (2023)
- **Payment Instruments**: Traditional → eCheq → Digital wallets → Crypto payments
- **Foreign Exchange Controls**: BCRA Communications "A" evolution and market responses

#### 2. Constitutional Rights Expansion
- **Amparo Development**: Siri case (1957) → Constitutional incorporation (1994) → Modern applications
- **Consumer Protection**: From Civil Code to specialized legislation
- **Environmental Rights**: Constitutional incorporation and implementation

#### 3. Crisis-Driven Innovations
- **Hyperinflation Period** (1989-1991): Quasi-currency emergence and regulation
- **2001-2002 Collapse**: Corralito, pesification, emergency legislation
- **2008 Crisis Response**: Counter-cyclical measures and institutional adaptations
- **Pandemic Adaptations** (2020-2024): Digital procedures, emergency powers, remote work regulation

### Methodology

#### Data Collection
1. **Primary Sources**: InfoLeg, SAIJ, CSJN decisions, BCRA communications, Boletín Oficial
2. **Secondary Analysis**: Academic literature, institutional reports, comparative studies
3. **Quantitative Tracking**: Regulatory frequency analysis, temporal pattern identification
4. **Qualitative Coding**: Institutional actor behavior, success/failure pattern analysis

#### Quality Control
- **Source Triangulation**: Multiple source verification for each data point
- **Temporal Validation**: Cross-referencing dates across different databases
- **Expert Review**: Academic and practitioner validation of classifications
- **Replication Package**: Full documentation for independent verification

### Comparative Benchmarks

#### Regional Comparisons
- **Chile**: Successful legal modernization trajectory (1990-2024)
- **Brazil**: Large economy legal evolution patterns
- **Mexico**: NAFTA/USMCA-driven legal harmonization
- **Colombia**: Post-conflict institutional transformation

#### OECD Baselines
- **Regulatory Quality Index**: World Bank governance indicators
- **Legal System Efficiency**: World Justice Project Rule of Law Index
- **Innovation Adoption Speed**: Technology penetration rates in legal practice

### Key Findings Preview

#### Evolutionary Patterns Identified
1. **Crisis Acceleration**: 3-5x increase in legal change velocity during economic crises
2. **Transplant Adaptation**: 70% modification rate for foreign legal instruments within 5 years
3. **Innovation Export Success**: 40% of crisis-period innovations adopted by regional peers
4. **Institutional Memory Loss**: 15-year cycle for forgetting previous crisis adaptations

#### Unique Argentine Contributions
- **Pesification Asymmetrica**: Differential currency conversion methodology
- **Corralito Framework**: Bank deposit freeze legal architecture
- **UVA Indexation**: Inflation-adjusted financial instrument innovation
- **Consumer Collective Actions**: Class action adaptation to civil law system

### Academic Applications

This dataset supports research in:
- **Legal Evolution Theory**: Empirical testing of selection mechanisms in law
- **Institutional Economics**: Crisis response and adaptation patterns
- **Comparative Law**: Transplant success/failure prediction models
- **Financial Regulation**: Innovation-regulation co-evolution dynamics
- **Constitutional Development**: Rights expansion in developing democracies

### Citation

```bibtex
@dataset{lerer2025_argentine_legal_evolution,
  title={Argentine Legal Evolution Dataset: Empirical Evidence for Extended Phenotype Theory of Law},
  author={Lerer, Adrian and Contributors},
  year={2025},
  publisher={GitHub},
  url={https://github.com/adrianlerer/legal-evolution-theory},
  note={Comprehensive dataset of legal evolution patterns in Argentina, 1950-2024}
}
```

### Contributing

This is an open academic project. Contributions welcome for:
- Additional case documentation
- Methodology refinements  
- Cross-validation of classifications
- Comparative data from other jurisdictions

See `CONTRIBUTING.md` for detailed guidelines.

### License

Creative Commons Attribution 4.0 International (CC BY 4.0)

### Contact

Adrian Lerer - Repository Maintainer  
Academic Institution Affiliation  
Email: [contact information]

---

**Repository Status**: Active Development  
**Last Updated**: September 2025  
**Version**: 1.0.0-beta  
**Expected Completion**: December 2025