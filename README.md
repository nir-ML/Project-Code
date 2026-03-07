# Polypharmacy Risk-aware Drug Recommender System

A modular system for drug-drug interaction (DDI) analysis and polypharmacy risk assessment, focused on **cardiovascular and antithrombotic drugs**.

## Overview

This system analyzes drug-drug interactions and provides:
- Drug Risk Network with centrality metrics
- Polypharmacy Risk Index (PRI) for risk quantification
- Alternative drug recommendations based on ATC classification
- Severity classification validated against DDInter (66.4% accuracy)
- Web interface with LLM-powered explanations

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get DrugBank Data
You need to obtain the DrugBank XML database (requires free academic license):
- Register at [DrugBank](https://go.drugbank.com/)
- Download "Full Database" in XML format
- Note the path to `full database.xml`

> **Why?** DrugBank's license prohibits redistribution. Each user must obtain their own copy. The knowledge graph is built at runtime from your local DrugBank file.

### 3. Run the Application
```bash
python run_app.py
```

The application will:
1. **Prompt for DrugBank XML path** (or pass as argument)
2. **Filter to cardiovascular (ATC C*) and antithrombotic (ATC B01*) drugs**
3. **Use included FAERS data** (public domain, no download needed)
4. **Build the knowledge graph at runtime**
5. **Launch the web interface** at http://localhost:7860

You can also specify the DrugBank path directly:
```bash
python run_app.py /path/to/full\ database.xml
# or
DRUGBANK_XML=/path/to/drugbank.xml python run_app.py
```

## Data Licensing

This repo is designed to avoid license conflicts:

| Data | License | In Repo? | Notes |
|------|---------|----------|-------|
| **DrugBank XML** | Academic License | ❌ No | User provides at runtime |
| **Knowledge Graph** | Derived from DrugBank | ❌ No | Built at runtime |
| **FAERS Data** | Public Domain (FDA) | ✅ Yes | Freely shareable |
| **Drug Class Refs** | Wikipedia/Public | ✅ Yes | Freely shareable |

### What's Included (Safe to Share)
- `external_data/faers_comprehensive_reports.json` - FDA adverse event reports
- `external_data/high_risk_drug_classes_reference.json` - QT-prolonging drugs, MAOIs, etc.

### What Users Must Provide
- DrugBank XML file (`full database.xml`) - get from [drugbank.com](https://go.drugbank.com/)

## Drug Filter

The application filters DrugBank to drugs relevant for cardiovascular care:
- **Cardiovascular drugs**: ATC codes starting with "C"
- **Antithrombotic drugs**: ATC codes starting with "B01"

This reduces ~20,000 drugs to ~4,300 focused on cardiac/blood medications and their ~490,000 interactions.

## Installation

```bash
git clone <repo-url>
cd DDI-riskAnalysis-Recommendation
pip install -r requirements.txt
```

Or use the setup script:
```bash
./setup.sh
```

Optional LLM support:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
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
├── run_app.py                  # 🚀 Main entry point - run this!
├── ddi_app.py                  # Web application (Gradio)
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
├── data/                       # User provides DrugBank XML here
│   └── .gitkeep                # (empty - user adds full database.xml)
├── external_data/              # Reference data
│   ├── faers_comprehensive_reports.json  # ✅ Included (Public Domain)
│   ├── high_risk_drug_classes_reference.json  # ✅ Included
│   ├── ddinter/                # Empty - optional validation data
│   ├── sider/                  # Empty - optional side effect data
│   └── ctd/                    # Empty - optional disease data
├── knowledge_graph_fact_based/ # Generated at runtime
│   └── .gitkeep                # (empty until run_app.py builds it)
├── main.py                     # CLI entry point
├── build_kg_api_based.py       # Alternative KG builder
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
