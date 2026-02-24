# Drug Recommendation System Publication Materials

This directory contains publication materials related to the Multi-Objective Drug Recommendation System for therapeutic substitution.

## Directory Structure

```
recommendation/
├── method_brief/                      # Methods documentation
│   └── recommendation_method.tex
├── supplementary/                     # Supplementary materials
│   └── supplementary_recommendation.tex
└── plots_and_data/                    # Figures and data files
    ├── drug_alternatives_heatmap.*
    ├── drug_substitution_network.*
    ├── network_safe_alternatives.*
    ├── class_alternatives_summary.*
    ├── generate_drug_alternatives.py
    └── generate_network_alternatives.py
```

## Multi-Objective Recommendation System

The recommendation system identifies therapeutic alternatives by optimizing three objectives:

```
RecScore(d_alt) = 0.40 × T(d_alt) + 0.35 × SAF(d_alt) + 0.25 × R(d_alt)
```

### Score Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Therapeutic Similarity (T)** | 0.40 | ATC code matching, shared targets, disease overlap, pathway similarity |
| **Safety Improvement (SAF)** | 0.35 | Reduction in DDI risk with current regimen |
| **Risk Reduction (R)** | 0.25 | Net decrease in severe interactions and PRI |

## Case Study Results

**Original Regimen**: Warfarin, Amiodarone, Digoxin, Quinidine, Propranolol
- 10 severe interactions (2 contraindicated, 8 major)

**After Warfarin → Dabigatran Substitution**:
- Contraindicated: 2 → 1 (50% reduction)
- Major: 8 → 6 (25% reduction)
- Total severe: 10 → 7 (30% reduction)
- Average PRI: 0.569 → 0.504 (11.5% improvement)

### DOAC Alternatives Ranking
| Drug | RecScore | Therapeutic | Safety | Risk↓ |
|------|----------|-------------|--------|-------|
| **Dabigatran** | **0.801** | 0.850 | 0.782 | 0.750 |
| Fondaparinux | 0.524 | 0.850 | 0.347 | 0.250 |
| Apixaban | 0.385 | 0.850 | 0.129 | 0.000 |
| Rivaroxaban | 0.385 | 0.850 | 0.129 | 0.000 |
| Edoxaban | 0.385 | 0.850 | 0.129 | 0.000 |

## Figures Included

1. **drug_alternatives_heatmap** - Recommendation scores for therapeutic substitutions
2. **drug_substitution_network** - Network visualization of drug substitution options
3. **network_safe_alternatives** - Safe alternative drug network
4. **class_alternatives_summary** - Class-level substitution patterns

## Usage

Generate figures:
```bash
python plots_and_data/generate_drug_alternatives.py
python plots_and_data/generate_network_alternatives.py
```

Compile LaTeX documents:
```bash
cd method_brief && pdflatex recommendation_method.tex
cd supplementary && pdflatex supplementary_recommendation.tex
```
