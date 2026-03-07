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

## Data Setup

**Important:** This project requires external datasets that cannot be redistributed due to licensing restrictions.

### Quick Setup
```bash
python scripts/download_data.py
```

This script will:
1. Create required directories
2. Check which data files are present
3. Provide download instructions for missing files

### Required Data Sources
| Source | License | Required |
|--------|---------|----------|
| [DrugBank](https://go.drugbank.com/) | Academic License | Yes |
| [DDInter](http://ddinter.scbdd.com/) | CC BY-NC-SA 4.0 | Yes |
| [SIDER](http://sideeffects.embl.de/) | CC BY-NC-SA 4.0 | Yes |
| [CTD](https://ctdbase.org/) | Free for academic | Optional |

See **[DATA_SOURCES.md](DATA_SOURCES.md)** for detailed instructions on obtaining each dataset.

### Alternative: API-Based Setup (No SIDER/CTD Required)

If you only have DrugBank, you can build the knowledge graph using public APIs:

```bash
# Build KG using DrugBank + OpenFDA API for side effects
python build_kg_api_based.py --output knowledge_graph_api_based

# Skip API calls (DrugBank only)
python build_kg_api_based.py --skip-api

# Limit API calls for testing
python build_kg_api_based.py --max-drugs 50
```

This approach uses:
- **DrugBank XML** (local, requires license)
- **OpenFDA API** (public, no license required) for side effects
- DrugBank indications for disease associations

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
├── scripts/                    # Utility scripts
│   └── download_data.py        # Data setup helper
├── data/                       # DDI datasets (see DATA_SOURCES.md)
├── external_data/              # External validation data
│   ├── ddinter/                # DDInter validation set
│   ├── sider/                  # Side effect data
│   └── ctd/                    # Chemical-disease data
├── knowledge_graph_fact_based/ # Built knowledge graph (generated)
│   ├── knowledge_graph.pkl
│   └── neo4j_export/           # Neo4j CSV exports
├── main.py                     # CLI entry point
├── ddi_app.py                  # Web application
├── build_fact_based_kg.py      # KG construction script
├── recalibrate_severity.py     # Severity classification
├── validate_against_ddinter.py # Validation script
├── DATA_SOURCES.md             # Data acquisition guide
└── README.md
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

This code is provided for research and educational purposes.

**Note:** The datasets used by this project (DrugBank, DDInter, SIDER, CTD) have their own licensing terms. See [DATA_SOURCES.md](DATA_SOURCES.md) for details. Users must obtain datasets directly from the original sources and comply with their respective licenses.
