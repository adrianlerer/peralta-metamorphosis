"""
RootFinder: Genealogical Tracing Through Legal Evolution

This module implements the ABAN (Ancestral Backward Analysis of Networks) 
algorithm for tracing genealogical lineages in legal doctrine evolution.

Author: Ignacio Adrián Lerer
License: MIT
"""

from .rootfinder import RootFinder, GenealogyNode

__version__ = "1.0.0"
__author__ = "Ignacio Adrián Lerer"
__email__ = "adrian@lerer.com.ar"

__all__ = ['RootFinder', 'GenealogyNode']