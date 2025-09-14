"""
Political Analysis Package: Adapting RootFinder and Legal-Memespace for Political Antagonisms

This package adapts the existing computational tools from the peralta-metamorphosis
repository to analyze political evolution and antagonisms in Argentine history.

Tools included:
- PoliticalRootFinder: Traces political genealogies through semantic networks
- PoliticalMemespace: Maps political positions in 4D space  
- IntegratedPoliticalAnalysis: Combines both tools for comprehensive analysis

Author: Ignacio Adrián Lerer
"""

from .political_rootfinder import PoliticalRootFinder, PoliticalGenealogyNode
from .political_memespace import PoliticalMemespace, PoliticalPhaseTransition
from .integrate_political_analysis import IntegratedPoliticalAnalysis

__version__ = "1.0.0"
__author__ = "Ignacio Adrián Lerer"

__all__ = [
    'PoliticalRootFinder',
    'PoliticalGenealogyNode', 
    'PoliticalMemespace',
    'PoliticalPhaseTransition',
    'IntegratedPoliticalAnalysis'
]