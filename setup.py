"""
Setup script for The Peralta Metamorphosis computational tools
Author: Ignacio Adrián Lerer
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="peralta-metamorphosis",
    version="1.0.0",
    author="Ignacio Adrián Lerer",
    author_email="adrian@lerer.com.ar",
    description="Computational tools for quantifying legal evolution in Argentina",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/adrianlerer/peralta-metamorphosis",
    
    packages=find_packages(),
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Sociology :: History",
    ],
    
    python_requires=">=3.8",
    
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "joblib>=1.1.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "reproduce-peralta=analysis.reproduce_paper:main",
        ],
    },
    
    include_package_data=True,
    
    package_data={
        "": ["data/cases/*.csv", "data/citations/*.csv", "data/congressional/*.csv"],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/adrianlerer/peralta-metamorphosis/issues",
        "Source": "https://github.com/adrianlerer/peralta-metamorphosis",
        "Documentation": "https://github.com/adrianlerer/peralta-metamorphosis/docs",
        "Paper": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=XXXXX",
    },
    
    keywords=[
        "legal-analysis", 
        "constitutional-law", 
        "argentina", 
        "computational-law",
        "network-analysis",
        "jurisprudence",
        "pagerank",
        "genealogical-analysis",
        "lotka-volterra",
        "phase-transitions"
    ],
    
    zip_safe=False,
)