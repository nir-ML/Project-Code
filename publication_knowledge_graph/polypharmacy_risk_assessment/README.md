# Polypharmacy Risk Assessment Publication Materials

This directory contains publication materials related to the Polypharmacy Risk Index (PRI) and drug interaction risk assessment.

## Directory Structure

```
polypharmacy_risk_assessment/
├── method_brief/                    # Methods documentation
│   └── polypharmacy_risk_method.tex
├── supplementary/                   # Supplementary materials
│   └── supplementary_polypharmacy.tex
└── plots_and_data/                  # Figures and data files
    ├── polypharmacy_risk_escalation.*
    ├── drug_risk_matrix.*
    ├── severity_heatmap.*
    ├── severity_distribution.csv
    └── generate_polypharmacy_figures.py
```

## Polypharmacy Risk Index (PRI)

The PRI quantifies individual drug risk within multi-drug regimens using four network-based metrics:

```
PRI(d) = 0.25 × C_degree(d) + 0.30 × C_weighted(d) + 0.20 × C_betweenness(d) + 0.25 × S(d)
```

### Component Metrics
- **Degree Centrality (0.25)**: Number of interacting drugs
- **Weighted Degree (0.30)**: Severity-weighted interaction sum
- **Betweenness Centrality (0.20)**: Role in risk propagation pathways
- **Severity Profile Score (0.25)**: Proportion of severe interactions

### Risk Classification
| PRI Score | Risk Level | Clinical Action |
|-----------|------------|-----------------|
| > 0.5 | High Risk | Immediate clinical review |
| 0.3 - 0.5 | Medium Risk | Close monitoring |
| < 0.3 | Lower Risk | Standard monitoring |

## Figures Included

1. **polypharmacy_risk_escalation** - Risk escalation with increasing drug count
2. **drug_risk_matrix** - High-risk drug combination matrix
3. **severity_heatmap** - Severity distribution across drug classes

## Usage

Generate figures:
```bash
python plots_and_data/generate_polypharmacy_figures.py
```

Compile LaTeX documents:
```bash
cd method_brief && pdflatex polypharmacy_risk_method.tex
cd supplementary && pdflatex supplementary_polypharmacy.tex
```
