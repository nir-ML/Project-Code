# Polypharmacy Risk-aware Drug Recommender System

A modular system for drug-drug interaction (DDI) analysis and polypharmacy risk assessment, focused on **cardiovascular and antithrombotic drugs**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-Research-green.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)

## ✨ Features

- **Drug-Drug Interaction Detection** - Analyze interactions between multiple medications
- **Risk Network Analysis** - Graph-based centrality metrics for polypharmacy risk
- **Polypharmacy Risk Index (PRI)** - Quantified risk scoring for drug regimens
- **Alternative Drug Recommendations** - ATC-based safer alternatives
- **Severity Classification** - Rule-based classification validated against DDInter (66.4% accuracy)
- **LLM-Powered Explanations** - Clinical summaries using local LLM (Ollama)
- **Web Interface** - Interactive Gradio application

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get DrugBank Data
You need the DrugBank XML database (requires free academic license):
1. Register at [DrugBank](https://go.drugbank.com/)
2. Download "Full Database" in XML format
3. Note the path to \`full database.xml\`

> **Why not included?** DrugBank's license prohibits redistribution. Each user must obtain their own copy. The knowledge graph is built at runtime from your local file.

### 3. Run the Application
```bash
python run_app.py
```

The application will:
1. **Prompt for DrugBank XML path** (or pass as argument/env var)
2. **Filter to cardiovascular and antithrombotic drugs**
3. **Build knowledge graph at runtime** (~759K DDIs)
4. **Launch web interface** at http://localhost:7860

**Alternative ways to specify DrugBank path:**
```bash
# Command line argument
python run_app.py /path/to/full\ database.xml

# Environment variable
DRUGBANK_XML=/path/to/drugbank.xml python run_app.py
```

---

## 📊 Data Licensing

This repo is designed to avoid license conflicts:

| Data Source | License | Included? | Notes |
|-------------|---------|-----------|-------|
| **DrugBank XML** | Academic License | ❌ No | User provides at runtime |
| **Knowledge Graph** | Derived from DrugBank | ❌ No | Built at runtime |
| **FAERS Data** | Public Domain (FDA) | ✅ Yes | 116K adverse event records |
| **High-Risk Drug Classes** | Wikipedia/Public | ✅ Yes | QT-prolonging, MAOIs, etc. |

### ✅ What's Included (Safe to Share)
```
external_data/
├── faers_comprehensive_reports.json    # FDA adverse events (Public Domain)
└── high_risk_drug_classes_reference.json  # Drug class references
```

### ❌ What Users Must Provide
- DrugBank XML file (\`full database.xml\`) - obtain from [drugbank.com](https://go.drugbank.com/)

---

## 🎯 Drug Filter

The application filters DrugBank to cardiovascular-relevant drugs:

| Category | ATC Code | Example Drugs |
|----------|----------|---------------|
| **Cardiovascular** | C* | Warfarin, Metoprolol, Atorvastatin |
| **Antithrombotic** | B01* | Aspirin, Clopidogrel, Rivaroxaban |

**Result:** ~4,300 drugs with ~759,000 interactions (from ~20,000 total drugs)

---

## 🏗️ Architecture

```
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
```

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

## 📈 Methodology

### Severity Classification

Rule-based classifier with empirically-derived keyword weights:

| Severity Level | Keywords/Triggers |
|----------------|-------------------|
| **Contraindicated** | FDA Black Box warnings, "do not use", fatal risk |
| **Major** | Bleeding risk, CHEST guidelines, life-threatening |
| **Moderate** | CYP interactions, concentration changes, monitoring |
| **Minor** | Sedation, GI effects, additive effects |

**Validation against DDInter:**
| Metric | Value |
|--------|-------|
| Exact Accuracy | 66.4% |
| Adjacent Accuracy | 99.3% |
| Cohen's Kappa | +0.096 |
| Validation Set | n=11,150 pairs |

### Polypharmacy Risk Index (PRI)

```
PRI = 0.25×(Degree) + 0.30×(WeightedDegree) + 0.20×(Betweenness) + 0.25×(SeverityProfile)
```

| Risk Level | PRI Score |
|------------|-----------|
| High Risk | > 0.5 |
| Medium Risk | 0.3 - 0.5 |
| Lower Risk | < 0.3 |

Severity weights: Contraindicated=10, Major=7, Moderate=4, Minor=1

### Alternative Drug Ranking Score (ARS)

```
ARS = 0.70×(Severity Reduction) + 0.30×(PRI Improvement)
```

---

## 📁 Project Structure

```
├── run_app.py                  # 🚀 Main entry point
├── ddi_app.py                  # Web application (Gradio)
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
│   ├── faers_comprehensive_reports.json  # ✅ FDA data (included)
│   └── high_risk_drug_classes_reference.json  # ✅ Reference data
├── data/
│   └── .gitkeep                # User adds DrugBank XML here
├── knowledge_graph_fact_based/ # Generated at runtime
├── requirements.txt
├── DATA_SOURCES.md             # Data acquisition guide
└── README.md
```

---

## 🛠️ Installation

### Basic Setup
```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

### With Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Optional: LLM Support
For AI-powered clinical summaries:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama3 model
ollama pull llama3
```

---

## 💻 Usage

### Web Interface
```bash
python run_app.py
# Opens at http://localhost:7860
```

### Command Line
```bash
python main.py --drugs "warfarin,aspirin,metoprolol"
```

### Programmatic
```python
from modules import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.analyze(['warfarin', 'aspirin', 'metoprolol'])
print(result['interactions'])
print(result['risk_assessment'])
```

---

## 🔬 Validation

### DDInter Validation
```bash
python validate_against_ddinter.py
```

### FAERS Integration
```python
from modules import FAERSClient
client = FAERSClient()
profile = client.get_drug_profile("Warfarin")
```

---

## ⚠️ Disclaimer

**This analysis is for informational and research purposes only.**

- Not a substitute for professional clinical judgment
- Consult qualified healthcare providers before making medication changes
- Drug interaction data may not be complete or up-to-date

---

## 📄 License

This code is provided for **research and educational purposes**.

**Important:** The datasets used by this project have their own licensing terms:
- **DrugBank** - Academic license required (free for academic use)
- **DDInter** - Academic use
- **SIDER/CTD** - Creative Commons

See [DATA_SOURCES.md](DATA_SOURCES.md) for details on obtaining data.

---

## 🙏 Acknowledgments

- [DrugBank](https://go.drugbank.com/) - Drug interaction database
- [FDA FAERS](https://open.fda.gov/apis/drug/event/) - Adverse event reports
- [DDInter](http://ddinter.scbdd.com/) - Validation dataset
- [Ollama](https://ollama.com/) - Local LLM inference
