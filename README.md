# The Peralta Metamorphosis - Computational Legal Analysis Tools

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Computational tools for quantifying the evolution of legal parasitism in Argentine constitutional law (1922-2025).

## 📖 Paper

These tools support the paper: **"The Peralta Metamorphosis: Quantifying the Evolution of Legal Parasitism Through Computational Analysis of Argentine Constitutional Degradation (1922-2025)"** by Ignacio Adrián Lerer

Available at: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=XXXXX)

## 🛠️ Tools Included

### JurisRank
Measures memetic fitness of legal doctrines through citation network analysis, adapting PageRank with temporal and hierarchical constraints.

**Key Features:**
- Temporal decay modeling for citation networks
- Court hierarchy weighting system
- Memetic fitness scoring for legal doctrines

### RootFinder
Traces genealogical paths of legal evolution using ABAN (Ancestral Backward Analysis of Networks) algorithm.

**Key Features:**
- Genealogical lineage tracing
- Inheritance fidelity calculation
- Mutation type classification
- Peralta dominance analysis

### Legal-Memespace
Maps competitive dynamics between legal doctrines in 4-dimensional space using Lotka-Volterra equations.

**Key Features:**
- Multi-dimensional doctrinal space mapping
- Competitive dynamics simulation
- Phase transition detection
- Evolutionary trajectory modeling

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/adrianlerer/peralta-metamorphosis.git
cd peralta-metamorphosis

# Install dependencies
pip install -r requirements.txt

# Run main analysis
python analysis/reproduce_paper.py
```

## 📊 Key Findings

- **Formalist doctrine fitness**: 0.89 (1922) → 0.03 (2025)
- **Emergency doctrine fitness**: 0.11 (1922) → 0.94 (2025)
- **Peralta dominance**: 89% of post-1990 cases trace genealogically to Peralta v. Estado Nacional
- **Phase transition coordinates**: [0.31, 0.89, 0.45, 0.67] detected around 1989-1991
- **Congressional selectivity**: 78% rejection rate for spending-related DNUs vs. 23% for others (2024-2025)

## 📁 Repository Structure

```
peralta-metamorphosis/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── jurisrank/
│   ├── __init__.py
│   └── jurisrank.py          # PageRank-based fitness calculation
├── rootfinder/
│   ├── __init__.py
│   └── rootfinder.py         # ABAN genealogical tracing
├── legal_memespace/
│   ├── __init__.py
│   └── memespace.py          # Lotka-Volterra modeling
├── analysis/
│   ├── __init__.py
│   ├── reproduce_paper.py    # Main reproduction script
│   ├── visualizations.py     # Plotting utilities
│   └── statistical_tests.py  # Hypothesis testing
├── data/
│   ├── cases/
│   │   ├── argentine_cases.csv
│   │   └── case_features.csv
│   ├── citations/
│   │   ├── citation_matrix.csv
│   │   └── citation_network.json
│   └── congressional/
│       ├── dnu_analysis_2024.csv
│       └── legislative_responses.csv
├── tests/
│   ├── test_jurisrank.py
│   ├── test_rootfinder.py
│   └── test_memespace.py
└── docs/
    ├── methodology.md
    ├── data_sources.md
    └── api_reference.md
```

## 🔬 Methodology

### 1. JurisRank Algorithm
Based on PageRank with legal-specific modifications:
- **Temporal weighting**: Recent citations weighted higher
- **Hierarchical weighting**: Supreme Court cases weighted more heavily
- **Doctrinal clustering**: Similar doctrines amplify each other's scores

### 2. ABAN Genealogical Tracing
- **Backward traversal**: Follow citation chains to identify precedential lineages
- **Inheritance analysis**: Quantify doctrinal element preservation vs. mutation
- **Fidelity scoring**: Measure genealogical integrity across generations

### 3. Memespace Modeling
- **4D mapping**: State/Individual, Emergency/Normal, Formal/Pragmatic, Temporary/Permanent axes
- **Competition dynamics**: Lotka-Volterra equations model doctrinal competition
- **Phase transitions**: Statistical change-point detection in doctrinal evolution

## 📈 Usage Examples

### Calculate Doctrine Fitness
```python
from jurisrank.jurisrank import JurisRank
import pandas as pd
import numpy as np

# Load case data
cases_df = pd.read_csv('data/cases/argentine_cases.csv')
citations_df = pd.read_csv('data/citations/citation_matrix.csv')

# Initialize JurisRank
jr = JurisRank(damping_factor=0.85, max_iterations=100)

# Calculate fitness scores
fitness_scores = jr.calculate_jurisrank(citation_matrix, cases_df)

# Display results
for case_id, fitness in fitness_scores.items():
    print(f"{case_id}: {fitness:.3f}")
```

### Trace Genealogical Lineage
```python
from rootfinder.rootfinder import RootFinder
import networkx as nx

# Create citation network
G = nx.from_pandas_edgelist(citations_df, source='citing_case', 
                           target='cited_case', create_using=nx.DiGraph())

# Initialize RootFinder
rf = RootFinder()

# Trace lineage of a modern case
genealogy = rf.trace_genealogy('Massa_2006', G, max_depth=10)

# Display genealogical path
for node in genealogy:
    print(f"Generation {node.generation}: {node.case_id}")
    print(f"  Fidelity: {node.inheritance_fidelity:.2f}")
    print(f"  Mutations: {node.mutation_type}")
```

### Map Doctrinal Space
```python
from legal_memespace.memespace import LegalMemespace
import matplotlib.pyplot as plt

# Initialize memespace
lm = LegalMemespace(n_dimensions=4)

# Map doctrines to 4D space
coordinates = lm.map_doctrinal_space(cases_df)

# Detect phase transitions
phase_transition = lm.calculate_phase_transition(coordinates, cases_df['date'])

print(f"Phase transition at: {phase_transition['date']}")
print(f"New coordinates: {phase_transition['coordinates_after']}")
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/test_jurisrank.py -v

# Run with coverage
python -m pytest tests/ --cov=jurisrank --cov=rootfinder --cov=legal_memespace
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use these tools in your research, please cite:

```bibtex
@article{lerer2025peralta,
  title={The Peralta Metamorphosis: Quantifying the Evolution of Legal Parasitism Through Computational Analysis of Argentine Constitutional Degradation (1922-2025)},
  author={Lerer, Ignacio Adrián},
  journal={SSRN Electronic Journal},
  year={2025},
  url={https://github.com/adrianlerer/peralta-metamorphosis}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Ignacio Adrián Lerer**  
Buenos Aires, Argentina  
Email: [your-email@example.com]  
ORCID: [0000-0000-0000-0000]

## 🙏 Acknowledgments

- Thanks to the open-source Python community for foundational libraries
- NetworkX team for graph analysis tools
- SciPy contributors for numerical computing capabilities
- Argentine Supreme Court for case law accessibility
- Legal scholars who provided feedback on methodological approaches

## 📚 References

1. Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual Web search engine.
2. Fowler, J. H., et al. (2007). Network analysis and the law: Measuring the legal importance of precedents.
3. Lupu, Y., & Voeten, E. (2012). Precedent in international courts: A network analysis.
4. Cross, F. B., et al. (2010). Citations in the U.S. Supreme Court: An empirical study of their use and significance.

---

**Note**: This repository contains computational tools for academic research. Results should be interpreted within their proper legal and historical context.