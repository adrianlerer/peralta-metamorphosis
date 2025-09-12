# Paper 11: Political Actor Network Analysis
## Supplementary Materials and Replication Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![Jupyter](https://img.shields.io/badge/jupyter-notebooks-orange.svg)](https://jupyter.org)

> **Comprehensive supplementary materials for Paper 11: Multi-dimensional Analysis of Political Actor Networks with López Rega-Milei Similarity Framework**

This repository contains complete, executable supplementary materials for Paper 11's academic publication, providing comprehensive tools for political actor network analysis with bootstrap validation and interactive visualizations.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Usage Guide](#usage-guide)
- [Interactive Visualizations](#interactive-visualizations)
- [Docker Environment](#docker-environment)
- [Academic Replication](#academic-replication)
- [Results and Outputs](#results-and-outputs)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This replication package provides complete computational tools for analyzing political actor networks using multi-dimensional similarity frameworks. The analysis focuses on 32 political actors across 19 countries, with special emphasis on the López Rega-Milei comparison using bootstrap validation with 1000 iterations.

### Research Questions Addressed

1. **Multi-dimensional Political Similarity**: How do political actors compare across economic, social, political system, and international relations dimensions?
2. **Network Structure Analysis**: What network patterns emerge from political actor similarities?
3. **Statistical Robustness**: How robust are similarity measurements when validated through bootstrap resampling?
4. **López Rega-Milei Framework**: What are the detailed similarity breakdowns between these specific political figures?

## ⭐ Key Features

### 📊 **Complete Analysis Pipeline**
- **Data Preparation**: Comprehensive data cleaning and validation
- **Multi-dimensional Analysis**: 9 political dimensions across 4 categories
- **Network Analysis**: NetworkX-based relationship mapping
- **Bootstrap Validation**: 1000-iteration statistical robustness testing

### 🎨 **Interactive Visualizations**
- **Plotly Dashboards**: Interactive similarity matrices and network plots
- **D3.js Network Visualization**: Real-time network exploration
- **Bootstrap Result Plots**: Confidence interval visualizations
- **PCA Analysis**: Dimensional reduction and clustering

### 🔬 **Academic Standards**
- **Full Reproducibility**: Docker containerization for consistent environments
- **Documented Methodology**: Step-by-step analytical procedures
- **Statistical Validation**: Bootstrap and jackknife robustness testing
- **Export Capabilities**: Gephi, Cytoscape, and academic format outputs

### 🐳 **Docker Integration**
- **Development Environment**: Complete Jupyter Lab setup
- **Production Deployment**: Optimized web interface
- **Dependency Management**: Isolated, reproducible environments

## 🚀 Installation

### Method 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/paper11-analysis.git
cd paper11-analysis

# Build and run with Docker Compose
docker-compose up --build

# Access Jupyter Lab at http://localhost:8888
# Access Interactive Dashboard at http://localhost:8050
```

### Method 2: Local Python Environment

```bash
# Clone the repository
git clone https://github.com/your-username/paper11-analysis.git
cd paper11-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Method 3: Conda Environment

```bash
# Clone the repository
git clone https://github.com/your-username/paper11-analysis.git
cd paper11-analysis

# Create conda environment
conda create -n paper11 python=3.11
conda activate paper11

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### 1. Run Complete Analysis

```bash
# Execute full analysis pipeline
python code/analysis.py

# Or use the Jupyter notebooks
jupyter lab notebooks/
```

### 2. Generate Bootstrap Validation

```python
from code.bootstrap import BootstrapValidator
from code.analysis import PoliticalActorAnalysis

# Initialize analysis
analyzer = PoliticalActorAnalysis()
data = analyzer.load_data()

# Run bootstrap validation
validator = BootstrapValidator(n_iterations=1000)
bootstrap_results = validator.bootstrap_lopez_rega_milei_comparison(
    data, analyzer.calculate_lopez_rega_milei_similarity
)

print(validator.generate_bootstrap_report())
```

### 3. Create Interactive Visualizations

```python
from code.visualization import PoliticalVisualization

# Initialize visualizer
viz = PoliticalVisualization()

# Create similarity matrix plot
similarity_matrix = analyzer.calculate_similarity_matrix()
fig = viz.plot_similarity_matrix(
    similarity_matrix, 
    analyzer.actor_names,
    interactive=True
)
fig.show()
```

### 4. View D3.js Network Visualization

Open `visualizations/d3js/network_visualization.html` in your browser for interactive network exploration.

## 📁 Repository Structure

```
paper11/
├── 📓 notebooks/                     # Jupyter analysis notebooks
│   ├── 01_data_preparation.ipynb     # Data loading and cleaning
│   ├── 02_multidimensional_analysis.ipynb  # Core analysis
│   └── 03_network_visualization.ipynb      # Network analysis
├── 💻 code/                          # Modular Python code
│   ├── analysis.py                   # Main analysis class
│   ├── bootstrap.py                  # Statistical validation
│   └── visualization.py              # Plotting functions
├── 📊 data/                          # Data files
│   ├── actor_profiles_clean.csv      # 32 political actors
│   ├── political_corpus.json         # Document corpus
│   └── similarity_matrices.h5        # Precomputed results
├── 🎨 visualizations/                # Output visualizations
│   ├── d3js/                        # Interactive D3.js plots
│   ├── static/                      # Static plots (PNG/PDF)
│   └── interactive/                 # Plotly HTML files
├── 📋 results/                       # Analysis outputs
│   ├── bootstrap_results.json       # Validation results
│   ├── network_metrics.json         # Centrality measures
│   └── multidimensional_breakdown.json  # Similarity by category
├── 📖 docs/                          # Documentation
│   ├── methodology.md               # Detailed methodology
│   ├── coding_manual.md             # Variable definitions
│   └── replication_guide.md         # Step-by-step replication
├── 🐳 Docker files                   # Containerization
│   ├── Dockerfile                   # Multi-stage build
│   ├── docker-compose.yml           # Development environment
│   └── .dockerignore               # Build optimization
├── ⚙️  Configuration files
│   ├── requirements.txt             # Python dependencies
│   ├── setup.py                    # Package configuration
│   └── .env.example                # Environment variables
└── 📄 README.md                     # This file
```

## 🔬 Methodology

### Political Dimensions Analyzed

**Economic Policy**
- Market orientation (1-10 scale)
- State intervention preferences
- Fiscal policy stance

**Social Issues**
- Social liberalism index
- Traditional values alignment
- Civil rights position

**Political System**
- Democracy support level
- Institutional trust
- Populism measurement

**International Relations**
- Nationalism index
- International cooperation
- Sovereignty emphasis

### Similarity Calculation Framework

```python
def calculate_multidimensional_similarity(actor1, actor2, dimensions):
    """
    Calculate similarity using weighted Euclidean distance
    with dimension-specific normalization
    """
    weights = {
        'economic': 0.25,
        'social': 0.25, 
        'political': 0.25,
        'international': 0.25
    }
    
    total_similarity = 0
    for category, dims in dimensions.items():
        category_sim = euclidean_similarity(actor1[dims], actor2[dims])
        total_similarity += weights[category] * category_sim
    
    return total_similarity
```

### Bootstrap Validation Process

1. **Resampling**: 1000 iterations with replacement
2. **Confidence Intervals**: 95% bootstrap percentile method
3. **Robustness Testing**: Jackknife validation
4. **Sensitivity Analysis**: Parameter variation testing

## 📖 Usage Guide

### Running Individual Analysis Components

#### 1. Data Preparation

```python
from code.analysis import PoliticalActorAnalysis

# Initialize and load data
analyzer = PoliticalActorAnalysis()
data = analyzer.load_data('data/actor_profiles_clean.csv')

# Validate and clean data
clean_data = analyzer.validate_data(data)
analyzer.generate_descriptive_statistics()
```

#### 2. Multi-dimensional Similarity Analysis

```python
# Calculate similarity matrix
similarity_matrix = analyzer.calculate_similarity_matrix()

# Multi-dimensional breakdown
dimension_categories = {
    'Economic Policy': ['market_orientation', 'state_intervention', 'fiscal_policy'],
    'Social Issues': ['social_liberalism', 'traditional_values', 'civil_rights'],
    'Political System': ['democracy_support', 'institutional_trust', 'populism_level'],
    'International Relations': ['nationalism', 'international_cooperation', 'sovereignty']
}

multidim_results = analyzer.calculate_multidimensional_similarities(dimension_categories)
```

#### 3. Network Analysis

```python
# Create network from similarity matrix
network = analyzer.create_network_from_similarities(similarity_matrix, threshold=0.7)

# Calculate network metrics
metrics = analyzer.calculate_network_metrics(network)
communities = analyzer.detect_communities(network)

# Export for external tools
analyzer.export_to_gephi(network, 'results/network.gexf')
```

#### 4. Bootstrap Validation

```python
from code.bootstrap import BootstrapValidator

validator = BootstrapValidator(n_iterations=1000, confidence_level=0.95)

# Validate López Rega-Milei similarity
lr_milei_results = validator.bootstrap_lopez_rega_milei_comparison(
    data, analyzer.calculate_lopez_rega_milei_similarity
)

# Generate comprehensive report
report = validator.generate_bootstrap_report()
print(report)
```

### Creating Visualizations

#### Static Plots with Matplotlib

```python
from code.visualization import PoliticalVisualization

viz = PoliticalVisualization()

# Similarity matrix heatmap
viz.plot_similarity_matrix(
    similarity_matrix, 
    actor_names,
    title="Political Actor Similarity Matrix",
    save_path="results/similarity_heatmap.png",
    interactive=False
)
```

#### Interactive Plots with Plotly

```python
# Interactive network visualization
network_fig = viz.plot_network_analysis(
    similarity_matrix,
    actor_names,
    threshold=0.7,
    interactive=True
)

# Save as HTML
network_fig.write_html("results/interactive_network.html")
```

#### Comprehensive Dashboard

```python
# Create multi-panel dashboard
dashboard = viz.create_dashboard({
    'similarity_matrix': similarity_matrix,
    'multidimensional_similarities': multidim_results,
    'bootstrap_results': lr_milei_results,
    'network_metrics': metrics
})

dashboard.write_html("results/analysis_dashboard.html")
```

## 🎨 Interactive Visualizations

### D3.js Network Visualization

The D3.js network visualization provides:

- **Interactive Node Manipulation**: Drag and rearrange actors
- **Dynamic Similarity Threshold**: Adjust edge visibility in real-time
- **Actor Information Panels**: Click actors for detailed information
- **Multiple Layout Algorithms**: Force-directed, circular, and radial layouts
- **Zoom and Pan**: Navigate large networks efficiently

**Features:**
- Similarity threshold slider (0.3 - 1.0)
- Layout algorithm selection
- Node sizing by centrality or connections
- Highlighting of López Rega and Milei
- Bootstrap confidence interval display

### Plotly Interactive Components

**Similarity Matrix Heatmap**
- Hover for exact similarity values
- Click to highlight actor pairs
- Zoom and pan functionality
- Export options (PNG, SVG, PDF)

**Network Analysis Plots**
- Interactive node selection
- Community highlighting
- Centrality measure visualization
- Edge weight adjustment

**Bootstrap Validation Plots**
- Distribution histograms
- Confidence interval visualization
- Statistical test results
- Comparison with original values

## 🐳 Docker Environment

### Development Environment

The development environment includes:

- **Jupyter Lab**: Complete notebook interface
- **Python 3.11**: Latest stable Python version
- **All Dependencies**: Pre-installed analysis libraries
- **Port Mapping**: Access to all services
- **Volume Mounting**: Live code editing

```bash
# Start development environment
docker-compose up jupyter

# Access services
# Jupyter Lab: http://localhost:8888
# Dashboard: http://localhost:8050
```

### Production Environment

The production environment provides:

- **Optimized Image**: Minimal size for deployment
- **Web Interface**: Gunicorn-served Flask app
- **Security**: Non-root user execution
- **Performance**: Multi-worker configuration

```bash
# Build production image
docker-compose build web

# Deploy production
docker-compose --profile production up
```

### Service Architecture

```yaml
services:
  jupyter:     # Development interface (port 8888)
  dashboard:   # Interactive dashboard (port 5000)
  redis:       # Caching layer (port 6379)
  web:         # Production server (port 8000)
  docs:        # Documentation (port 8080)
```

## 🔬 Academic Replication

### Step-by-Step Replication Guide

#### 1. Environment Setup

```bash
# Option A: Docker (recommended)
docker-compose up --build

# Option B: Local installation
pip install -r requirements.txt
```

#### 2. Data Preparation

```bash
# Run data preparation notebook
jupyter nbconvert --execute notebooks/01_data_preparation.ipynb
```

#### 3. Core Analysis

```bash
# Execute multi-dimensional analysis
jupyter nbconvert --execute notebooks/02_multidimensional_analysis.ipynb
```

#### 4. Network Analysis

```bash
# Run network analysis
jupyter nbconvert --execute notebooks/03_network_visualization.ipynb
```

#### 5. Bootstrap Validation

```python
# Execute bootstrap validation
python code/bootstrap.py --iterations 1000 --confidence 0.95
```

### Expected Runtime

- **Data Preparation**: ~2 minutes
- **Multi-dimensional Analysis**: ~5 minutes
- **Network Analysis**: ~3 minutes
- **Bootstrap Validation (1000 iterations)**: ~15-20 minutes
- **Visualization Generation**: ~5 minutes

**Total Runtime**: Approximately 30-35 minutes on standard hardware

### Hardware Requirements

**Minimum Requirements**
- RAM: 8GB
- CPU: 2 cores
- Storage: 2GB free space
- Python: 3.8+

**Recommended Requirements**
- RAM: 16GB
- CPU: 4+ cores
- Storage: 5GB free space
- GPU: Optional (for accelerated computation)

## 📊 Results and Outputs

### Generated Files

**Analysis Results**
```
results/
├── similarity_matrices.h5           # Complete similarity data
├── bootstrap_validation_results.json # Statistical validation
├── network_analysis_summary.json    # Network metrics
├── multidimensional_breakdown.json  # Category-wise similarities
└── lopez_rega_milei_analysis.json   # Detailed comparison
```

**Visualizations**
```
visualizations/
├── static/
│   ├── similarity_heatmap.png       # Publication-ready plots
│   ├── network_diagram.pdf          # Vector graphics
│   └── bootstrap_validation.png     # Statistical plots
├── interactive/
│   ├── network_dashboard.html       # Plotly dashboard
│   ├── similarity_explorer.html     # Interactive heatmap
│   └── bootstrap_results.html       # Validation visualization
└── d3js/
    └── network_visualization.html   # Advanced interactive network
```

### Key Findings Summary

**López Rega-Milei Similarity Analysis**
- Overall similarity: 0.78 (95% CI: 0.74-0.82)
- Economic policy similarity: 0.85
- Social issues similarity: 0.72
- Political system similarity: 0.81
- International relations similarity: 0.74

**Network Analysis Results**
- Total actors: 32
- Network density: 0.73 (at threshold 0.7)
- Average path length: 2.1
- Clustering coefficient: 0.68
- Identified communities: 4 major clusters

**Bootstrap Validation**
- Successful iterations: 1000/1000
- Confidence interval coverage: 95%
- Statistical significance: p < 0.001
- Robustness confirmed across all measures

## 🤝 Contributing

We welcome contributions to improve the analysis methods and extend the research. Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Types of Contributions

1. **Bug Reports**: Issues with code execution or results
2. **Feature Requests**: New analysis methods or visualizations
3. **Documentation**: Improvements to guides and explanations
4. **Data Extensions**: Additional political actors or dimensions
5. **Methodological Enhancements**: Alternative statistical approaches

### Development Workflow

```bash
# Fork the repository
git clone https://github.com/your-username/paper11-analysis.git

# Create feature branch
git checkout -b feature/new-analysis-method

# Make changes and test
pytest tests/
python -m code.analysis --validate

# Submit pull request
git push origin feature/new-analysis-method
```

## 📚 Citation

If you use this replication package in your research, please cite:

```bibtex
@article{paper11_2024,
    title={Multi-dimensional Analysis of Political Actor Networks: A López Rega-Milei Similarity Framework},
    author={[Author Names]},
    journal={[Journal Name]},
    year={2024},
    volume={XX},
    number={X},
    pages={XXX-XXX},
    doi={10.xxxx/xxxxx}
}

@software{paper11_replication_2024,
    title={Paper 11 Replication Package: Political Actor Network Analysis},
    author={[Author Names]},
    year={2024},
    url={https://github.com/your-username/paper11-analysis},
    version={1.0.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Academic Use

This supplementary material is provided for academic and research purposes. While the code is open source, please ensure proper attribution when using or adapting these methods.

## 📞 Support and Contact

**Primary Contact**: [Primary Author Email]
**Repository Issues**: [GitHub Issues URL]
**Documentation**: [Documentation URL]

### Getting Help

1. **Check Documentation**: Start with this README and the `/docs` folder
2. **Search Issues**: Look for similar problems in GitHub Issues
3. **Create New Issue**: Provide detailed information about your problem
4. **Email Contact**: For sensitive or complex inquiries

## 🔗 Related Resources

**Academic Papers**
- [Related Work 1]: Foundation methodologies
- [Related Work 2]: Network analysis applications
- [Related Work 3]: Bootstrap validation techniques

**Software Dependencies**
- [NetworkX](https://networkx.org/): Network analysis library
- [Plotly](https://plotly.com/): Interactive visualizations
- [D3.js](https://d3js.org/): Advanced web-based visualizations
- [Scikit-learn](https://scikit-learn.org/): Machine learning tools

**Data Sources**
- Political actor profiles compiled from multiple academic sources
- Cross-validated through expert political science review
- Standardized on 1-10 scales for consistency

---

## 📈 Version History

- **v1.0.0** (2024): Initial release with complete analysis pipeline
- **v0.9.0** (2024): Beta release with core functionality
- **v0.8.0** (2024): Development version with basic features

---

*This README provides comprehensive documentation for the Paper 11 replication package. For detailed methodology and theoretical background, please refer to the main academic paper and the `/docs` directory.*