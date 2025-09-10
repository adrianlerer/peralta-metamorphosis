"""
JurisRank: Legal Doctrine Fitness Through Citation Network Analysis

This module implements the JurisRank algorithm for measuring memetic fitness
of legal doctrines through modified PageRank analysis of citation networks.

Author: Ignacio Adrián Lerer
License: MIT
"""

from .jurisrank import JurisRank

__version__ = "1.0.0"
__author__ = "Ignacio Adrián Lerer"
__email__ = "adrian@lerer.com.ar"

__all__ = ['JurisRank']