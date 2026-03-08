# Polypharmacy Risk-aware Drug Recommender System

A modular system for drug-drug interaction (DDI) analysis and polypharmacy risk assessment, focused on cardiovascular and antithrombotic drugs.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-Research-green.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)

## Features

- **Drug-Drug Interaction Detection** - Analyze interactions between multiple medications
- **Risk Network Analysis** - Graph-based centrality metrics for polypharmacy risk
- **Polypharmacy Risk Index (PRI)** - Quantified risk scoring for drug regimens
- **Alternative Drug Recommendations** - ATC-based safer alternatives
- **Severity Classification** - Rule-based classification validated against DDInter (66.4% accuracy)
- **LLM-Powered Explanations** - Clinical summaries using local LLM (Ollama)
- **Web Interface** - Interactive Gradio application

---

## Quick Start

### 1. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Obtain DrugBank Data
You need the DrugBank XML database (requires free academic license):
1. Register at DrugBank (https://go.drugbank.com/)
2. Download "Full Database" in XML format
3. Note the path to \`full database.xml\`

**Note:** DrugBank's license prohibits redistribution. Each user must obtain their own copy. The knowledge graph is built at runtime from your local file.

### 3. Run the Application
\`\`\`bash
python run_app.py
\`\`\`

The application will:
1. Prompt for DrugBank XML path (or pass as argument/env var)
2. Filter to cardiovascular and antithrombotic drugs
3. Build knowledge graph at runtime (~759K DDIs)
4. Launch web interface at http://localhost:7860

**Alternative ways to specify DrugBank path:**
\`\`\`bash
# Command line argument
python run_app.py /path/to/full\ database.xml

# Environment variable
DRUGBANK_XML=/path/to/drugbank.xml python run_app.py
\`\`\`

---

## Data Licensing

This repository is designed to avoid license conflicts:

| Data Source | License | Included | Notes |
|-------------|---------|----------|-------|
| DrugBank XML | Academic License | No | User provides at runtime |
| Knowledge Graph | Derived from DrugBank | No | Built at runtime |
| CTD (drug-disease) | Public API | Auto | Fetched via API, cached locally |
| SIDER (side effects) | CC BY-NC-SA | Auto | Auto-downloaded from public URLs |
| FAERS Data | Public Domain (FDA) | Yes | 116K adverse event records |
| High-Risk Drug Classes | Public Domain | Yes | QT-prolonging, MAOIs, etc. |

### Included Data (Safe to Share)
\`\`\`
external_data/
├── faers_comprehensive_reports.json    # FDA adverse events (Public Domain)
├── high_risk_drug_classes_reference.json  # Drug class references
├── ctd_cache.json                      # CTD API responses (auto-generated)
└── sider/                              # Auto-downloaded at runtime
    ├── drug_names.tsv
    └── meddra_all_se.tsv.gz
\`\`\`

### Data Users Must Provide
- DrugBank XML file (\`full database.xml\`) - obtain from https://go.drugbank.com/

---

## Drug Filter

The application filters DrugBank to cardiovascular-relevant drugs using ATC classification:

| Category | ATC Code | Example Drugs |
|----------|----------|---------------|
| Cardiovascular | C* | Warfarin, Metoprolol, Atorvastatin |
| Antithrombotic | B01* | Aspirin, Clopidogrel, Rivaroxaban |

**Result:** Approximately 4,300 drugs with 759,000 interactions (filtered from over 20,000 total drugs)

---

## Architecture

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator (Pipeline)                     │
└─────────────────────────────────────────────────────────────────┘
        │              │                │                │
        ▼              ▼                ▼                ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Interaction  │ │   Severity    │ │  Alternative  │ │    Report     │
│   Detector    │ │  Classifier   │ │    Finder     │ │   Generator   │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │              │                │                │
        ▼              ▼                ▼                ▼
   DDI Detection   Risk Scoring    ATC Matching    LLM Synthesis
\`\`\`

### Core Modules

| Module | Purpose |
|--------|---------|
| \`interaction_detector.py\` | Detects DDIs from medication lists |
| \`severity_classifier.py\` | Classifies interaction severity (4 levels) |
| \`drug_risk_network.py\` | Builds graph-based risk network |
| \`alternative_finder.py\` | Finds safer therapeutic alternatives |
| \`report_generator.py\` | Generates clinical reports |
| \`llm_client.py\` | LLM integration (Ollama/Llama3) |
| \`faers_integration.py\` | FDA adverse event data |

---

## Methodology

### Severity Classification

Rule-based classifier with empirically-derived keyword weights:

| Severity Level | Keywords/Triggers |
|----------------|-------------------|
| Contraindicated | FDA Black Box warnings, "do not use", fatal risk |
| Major | Bleeding risk, clinical guideline warnings, life-threatening |
| Moderate | CYP interactions, concentration changes, monitoring required |
| Minor | Sedation, GI effects, additive effects |

**Validation Results (DDInter):**

| Metric | Value |
|--------|-------|
| Exact Accuracy | 66.4% |
| Adjacent Accuracy | 99.3% |
| Cohen's Kappa | +0.096 |
| Validation Set | n=11,150 pairs |

### Polypharmacy Risk Index (PRI)

\`\`\`
PRI = 0.25 x (Degree) + 0.30 x (WeightedDegree) + 0.20 x (Betweenness) + 0.25 x (SeverityProfile)
\`\`\`

| Risk Level | PRI Score |
|------------|-----------|
| High Risk | > 0.5 |
| Medium Risk | 0.3 - 0.5 |
| Lower Risk | < 0.3 |

Severity weights: Contraindicated=10, Major=7, Moderate=4, Minor=1

### Alternative Drug Ranking Score (ARS)

\`\`\`
ARS = 0.70 x (Severity Reduction) + 0.30 x (PRI Improvement)
\`\`\`

---

## Project Structure

\`\`\`
├── run_app.py                  # Main entry point
├── ddi_app.py                  # Web application (Gradio)
├── main.py                     # Command-line interface
├── modules/
│   ├── orchestrator.py         # Pipeline controller
│   ├── interaction_detector.py # DDI detection
│   ├── severity_classifier.py  # Severity classification
│   ├── alternative_finder.py   # Alternative recommendations
│   ├── report_generator.py     # Report generation
│   ├── drug_risk_network.py    # Network analysis
│   ├── recommender.py          # Drug ranking
│   ├── llm_client.py           # LLM integration (Ollama)
│   └── faers_integration.py    # FAERS data integration
├── external_data/
│   ├── faers_comprehensive_reports.json  # FDA data (included)
│   └── high_risk_drug_classes_reference.json  # Reference data
├── data/
│   └── .gitkeep                # User adds DrugBank XML here
├── knowledge_graph_fact_based/ # Generated at runtime
├── requirements.txt
├── DATA_SOURCES.md             # Data acquisition guide
└── README.md
\`\`\`

---

## Installation

### Basic Setup
\`\`\`bash
git clone https://github.com/nir-ML/Project-Code.git
cd Project-Code
pip install -r requirements.txt
\`\`\`

### With Virtual Environment (Recommended)
\`\`\`bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
\`\`\`

### Optional: LLM Support
For AI-powered clinical summaries:
\`\`\`bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama3 model
ollama pull llama3
\`\`\`

---

## Usage

### Web Interface
\`\`\`bash
python run_app.py
# Opens at http://localhost:7860
\`\`\`

### Command Line
\`\`\`bash
python main.py --drugs "warfarin,aspirin,metoprolol"
\`\`\`

### Programmatic
\`\`\`python
from modules import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.analyze(['warfarin', 'aspirin', 'metoprolol'])
print(result['interactions'])
print(result['risk_assessment'])
\`\`\`

---

## Validation

### DDInter Validation
\`\`\`bash
python validate_against_ddinter.py
\`\`\`

### FAERS Integration
\`\`\`python
from modules import FAERSClient
client = FAERSClient()
profile = client.get_drug_profile("Warfarin")
\`\`\`

---

## Disclaimer

**This analysis is for informational and research purposes only.**

- Not a substitute for professional clinical judgment
- Consult qualified healthcare providers before making medication changes
- Drug interaction data may not be complete or up-to-date

---

## License

This code is provided for research and educational purposes.

**Important:** The datasets used by this project have their own licensing terms:
- DrugBank - Academic license required (free for academic use)
- DDInter - Academic use (CC BY-NC-SA 4.0)
- SIDER/CTD - Creative Commons

See [DATA_SOURCES.md](DATA_SOURCES.md) for details on obtaining data.

---

## References

1. Wishart DS, Feunang YD, Guo AC, et al. DrugBank 5.0: a major update to the DrugBank database for 2018. *Nucleic Acids Res*. 2018;46(D1):D1074-D1082. doi:10.1093/nar/gkx1037

2. Xiong G, Wu Z, Yi J, et al. DDInter: an online drug-drug interaction database towards improving clinical decision-making and patient safety. *Nucleic Acids Res*. 2022;50(D1):D1200-D1207. doi:10.1093/nar/gkab880

3. Kuhn M, Letunic I, Jensen LJ, Bork P. The SIDER database of drugs and side effects. *Nucleic Acids Res*. 2016;44(D1):D1075-D1079. doi:10.1093/nar/gkv1075

4. Davis AP, Grondin CJ, Johnson RJ, et al. Comparative Toxicogenomics Database (CTD): update 2023. *Nucleic Acids Res*. 2023;51(D1):D1257-D1262. doi:10.1093/nar/gkac833

5. US Food and Drug Administration. FDA Adverse Event Reporting System (FAERS) Public Dashboard. https://open.fda.gov/apis/drug/event/. Accessed 2024.

6. World Health Organization Collaborating Centre for Drug Statistics Methodology. ATC/DDD Index 2024. https://www.whocc.no/atc_ddd_index/. Accessed 2024.
