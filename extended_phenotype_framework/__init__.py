"""
Extended Phenotype Legal Theory Framework

Implementation of Dawkins' extended phenotype theory applied to legal systems.
Based on the comprehensive analysis by Ignacio Adrián Lerer applying 
Richard Dawkins' 2024 concepts (palimpsest, verticovirus/horizontovirus, 
genetic book of the dead, coalescence theory) to legal analysis.

This framework models law as CONSTRUCTED by entities (constructors) rather 
than evolved, implementing the revolutionary concept that legal structures 
are extended phenotypes of power entities.
"""

__version__ = "0.1.0"
__author__ = "Ignacio Adrián Lerer"

from .core.constructors.constructor_base import Constructor, ConstructorType, ConstructionStrategy
from .core.phenotypes.legal_phenotype import LegalPhenotype, PhenotypeType
from .core.environment.legal_landscape import LegalLandscape

__all__ = [
    'Constructor',
    'ConstructorType', 
    'ConstructionStrategy',
    'LegalPhenotype',
    'PhenotypeType',
    'LegalLandscape'
]