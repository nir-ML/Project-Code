"""
DDI Analysis Modules - Modular Architecture for Polypharmacy Risk Analysis

This package provides a modular architecture for drug-drug interaction
analysis and polypharmacy risk assessment.

Architecture Overview:

    Orchestrator (Central Pipeline Controller)
        |
        v
    InteractionDetector -> SeverityClassifier
                                |
                                v
    ReportGenerator     <-  AlternativeFinder

Modules:
--------
- BaseModule: Abstract base class for all modules
- InteractionDetector: Detects drug-drug interactions from a medication list
- SeverityClassifier: Classifies and scores interaction severity
- AlternativeFinder: Finds safer therapeutic alternatives via ATC classification
- ReportGenerator: Generates human-readable clinical reports
- Orchestrator: Coordinates the full analysis pipeline

Usage:
------
>>> from modules import Orchestrator
>>> import pandas as pd
>>> 
>>> # Load DDI data
>>> df = pd.read_csv('ddi_data.csv')
>>> 
>>> # Initialize orchestrator
>>> orchestrator = Orchestrator()
>>> orchestrator.initialize(df)
>>> 
>>> # Analyze medications
>>> result = orchestrator.analyze_drugs(['Warfarin', 'Aspirin', 'Metoprolol'])
>>> print(result['reports'])
"""

from .base_module import BaseModule, Result, PipelineStatus, Message
from .interaction_detector import InteractionDetector
from .severity_classifier import SeverityClassifier
from .alternative_finder import AlternativeFinder
from .report_generator import ReportGenerator
from .orchestrator import Orchestrator
from .llm_client import BioMistralClient, OllamaClient, get_llm_client
from .drug_risk_network import DrugRiskNetwork, DrugNode, DDIEdge
from .recommender import AlternativeRecommender, AlternativeCandidate

# FAERS External Validation
from .faers_integration import FAERSClient, FAERSValidator, FAERSDrugProfile

# GNN Risk Assessment (optional - requires torch_geometric)
try:
    from .gnn_risk_assessment import (
        GNNSeverityPredictor, 
        GNNEmbeddingPredictor, 
        DrugEmbedder,
        run_gnn_comparison
    )
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# Comprehensive Comparison
from .comprehensive_comparison import ComprehensiveComparison, AlgorithmicRiskAssessor

__all__ = [
    # Base classes
    'BaseModule',
    'Result', 
    'PipelineStatus',
    'Message',
    # Core modules
    'InteractionDetector',
    'SeverityClassifier',
    'AlternativeFinder',
    'ReportGenerator',
    # Main orchestrator
    'Orchestrator',
    # LLM clients
    'BioMistralClient',
    'OllamaClient',
    'get_llm_client',
    # Drug Risk Network
    'DrugRiskNetwork',
    'DrugNode',
    'DDIEdge',
    # Recommender
    'AlternativeRecommender',
    'AlternativeCandidate',
    # FAERS External Validation
    'FAERSClient',
    'FAERSValidator',
    'FAERSDrugProfile',
    # GNN Risk Assessment
    'GNN_AVAILABLE',
    'GNNSeverityPredictor',
    'GNNEmbeddingPredictor',
    'DrugEmbedder',
    'run_gnn_comparison',
    # Comprehensive Comparison
    'ComprehensiveComparison',
    'AlgorithmicRiskAssessor'
]

__version__ = '1.0.0'
__author__ = 'Polypharmacy Risk-aware Recommender System'
