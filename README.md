# Polypharmacy Risk-aware Drug Recommender System

A modular system for drug-drug interaction (DDI) analysis and polypharmacy risk assessment.

## Overview

This system analyzes drug-drug interactions and provides:
- Drug Risk Network with centrality metrics
- Polypharmacy Risk Index (PRI) for risk quantification
- Alternative drug recommendations based on ATC classification
- Severity classification validated against DDInter (66.4% accuracy)
- Web interface with LLM-powered explanations

## Installation

```bash
cd DDI-riskAnalysis-Recommendation
./setup.sh
```

Or manually:
```bash
pip install -r requirements.txt
# Optional: Install Ollama for LLM features
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

## Quick Start

### Command Line
```bash
python main.py --drugs "Warfarin,Aspirin,Metoprolol,Lisinopril"
python main.py --interactive
python main.py --sample cardiovascular_basic
```

### Web Application
```bash
python ddi_app.py
# Open http://localhost:7860
```

### Python API
```python
from modules import Orchestrator
import pandas as pd

df = pd.read_csv('data/ddi_cardio_or_antithrombotic_labeled (1).csv')
orchestrator = Orchestrator(verbose=True)
orchestrator.initialize(df)

result = orchestrator.analyze_drugs(['Warfarin', 'Aspirin', 'Metoprolol'])
print(result['reports'])
```

## Architecture

The system uses a modular pipeline:

```
Orchestrator (Pipeline Controller)
    |
    v
InteractionDetector --> SeverityClassifier --> AlternativeFinder --> ReportGenerator
    |                         |                      |                     |
    v                         v                      v                     v
DDI Detection           Risk Scoring          ATC Matching         Report Generation
```

### Modules

| Module | Purpose |
|--------|---------|
| `InteractionDetector` | Detects DDIs from medication lists |
| `SeverityClassifier` | Classifies interaction severity |
| `DrugRiskNetwork` | Builds graph-based risk network |
| `AlternativeFinder` | Finds safer therapeutic alternatives |
| `ReportGenerator` | Generates clinical reports |

## Methodology

### Severity Classification

Rule-based classifier with empirically-derived keyword weights, validated against DDInter:

| Metric | Value |
|--------|-------|
| Exact Accuracy | 66.4% |
| Adjacent Accuracy | 99.3% |
| Cohen's Kappa | +0.096 |
| Validation Set | n=11,150 pairs |

Classification rules based on:
- FDA Black Box Warnings (Contraindicated)
- CHEST Guidelines, bleeding risk (Major)
- CYP interactions, concentration changes (Moderate)
- Sedation, GI effects (Minor)

### Polypharmacy Risk Index (PRI)

```
PRI = 0.25*(Degree Centrality) + 0.30*(Weighted Degree) + 0.20*(Betweenness) + 0.25*(Severity Profile)
```

Severity weights: Contraindicated=10, Major=7, Moderate=4, Minor=1

### Alternative Drug Ranking

Alternatives are scored based on:
- Risk Reduction: 35%
- Centrality Reduction: 20%
- Phenotype Avoidance: 25%
- New Interaction Penalty: 20%

## Data Format

Input CSV columns:

| Column | Description |
|--------|-------------|
| `drugbank_id_1/2` | DrugBank identifiers |
| `drug_name_1/2` | Drug names |
| `atc_1/2` | ATC classification codes |
| `is_cardiovascular_1/2` | Cardiovascular drug flag |
| `is_antithrombotic_1/2` | Antithrombotic drug flag |
| `interaction_description` | Interaction mechanism text |
| `severity_label` | Severity classification |
| `severity_confidence` | Classification confidence score |
| `severity_numeric` | Numeric severity (1-4) |

## Project Structure

```
├── modules/                    # Core modules
│   ├── orchestrator.py         # Pipeline controller
│   ├── interaction_detector.py # DDI detection
│   ├── severity_classifier.py  # Severity classification
│   ├── alternative_finder.py   # Alternative recommendations
│   ├── report_generator.py     # Report generation
│   ├── drug_risk_network.py    # Network analysis
│   ├── recommender.py          # Drug ranking
│   ├── llm_client.py           # LLM integration
│   └── faers_integration.py    # FAERS API client
├── data/                       # DDI datasets
│   └── ddi_cardio_or_antithrombotic_labeled (1).csv
├── external_data/              # Validation data
│   ├── ddinter/                # DDInter validation set
│   ├── sider/                  # Side effect data
│   └── ctd/                    # Chemical-disease data
├── knowledge_graph_fact_based/ # Built knowledge graph
│   ├── knowledge_graph.pkl
│   └── neo4j_export/           # Neo4j CSV exports
├── main.py                     # CLI entry point
├── ddi_app.py                  # Web application
├── build_fact_based_kg.py      # KG construction script
├── recalibrate_severity.py     # Severity classification
└── validate_against_ddinter.py # Validation script
```

## External Validation

### FAERS Integration
```python
from modules import FAERSClient
client = FAERSClient()
profile = client.get_drug_profile("Warfarin")
```

### DDInter Validation
```bash
python validate_against_ddinter.py
```

## Disclaimer

This analysis is for informational purposes only and should not replace professional clinical judgment. Consult qualified healthcare providers before making medication changes.

## License

For research and educational purposes.
