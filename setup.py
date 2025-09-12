"""
Setup configuration for Paper 11: Political Actor Network Analysis
Complete academic replication package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="paper11-analysis",
    version="1.0.0",
    
    # Package information
    description="Multi-dimensional Analysis of Political Actor Networks with López Rega-Milei Similarity Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # Author and contact
    author="Paper 11 Research Team",
    author_email="contact@paper11research.org",
    url="https://github.com/your-username/paper11-analysis",
    
    # Package configuration
    packages=find_packages(exclude=['tests*', 'docs*']),
    package_dir={'': '.'},
    
    # Include data files
    package_data={
        'paper11': [
            'data/*.csv',
            'data/*.json',
            'data/*.h5',
            'visualizations/d3js/*.html',
            'visualizations/d3js/*.js',
            'visualizations/d3js/*.css',
            'docs/*.md',
            'notebooks/*.ipynb',
        ],
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies for specific features
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
            'pre-commit>=3.3.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=2.0.0',
        ],
        'gpu': [
            'cupy-cuda11x>=12.0.0',  # For CUDA 11.x
        ],
        'web': [
            'flask>=2.3.0',
            'dash>=2.10.0',
            'gunicorn>=20.1.0',
        ],
    },
    
    # Command line interfaces
    entry_points={
        'console_scripts': [
            'paper11-analyze=code.analysis:main',
            'paper11-bootstrap=code.bootstrap:main',
            'paper11-visualize=code.visualization:main',
            'paper11-dashboard=code.dashboard:main',
        ],
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Classification
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Sociology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords
    keywords=[
        'political science',
        'network analysis', 
        'bootstrap validation',
        'similarity analysis',
        'academic research',
        'replication package',
        'data visualization',
        'political actors',
        'multidimensional analysis',
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://paper11-analysis.readthedocs.io/',
        'Bug Reports': 'https://github.com/your-username/paper11-analysis/issues',
        'Source': 'https://github.com/your-username/paper11-analysis',
        'Academic Paper': 'https://doi.org/10.xxxx/paper11',
    },
    
    # License
    license='MIT',
    
    # Zip safe
    zip_safe=False,
)