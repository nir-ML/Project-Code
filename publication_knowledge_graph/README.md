# Publication Materials: DDI Knowledge Graph Construction

## Fact-Based Knowledge Graph for Drug-Drug Interaction Analysis

This folder contains publication-quality materials for the DDI Knowledge Graph construction methodology.

## Folder Structure

```
publication_knowledge_graph/
├── README.md                              # This file
├── KNOWLEDGE_GRAPH_CONSTRUCTION.md        # Complete methodology documentation
├── methods.tex                            # Publication methods (LaTeX)
├── data/
│   ├── kg_statistics.json                 # Complete statistics
│   ├── node_counts.csv                    # Node type counts
│   ├── edge_counts.csv                    # Edge type counts
│   └── data_sources.csv                   # Data source provenance
├── figures/
│   └── [knowledge graph visualizations]
└── tables/
    ├── table1_node_statistics.tex         # Node statistics table
    ├── table2_edge_statistics.tex         # Edge statistics table
    └── table3_data_sources.tex            # Data sources table
```

## Key Results

### Knowledge Graph Statistics

| Metric | Count |
|--------|-------|
| **Total Nodes** | 45,655 |
| **Total Edges** | 1,211,674 |
| **Unique Drugs** | 4,313 |
| **DDI Pairs** | 759,774 |

### Node Types

| Node Type | Count | Percentage |
|-----------|-------|------------|
| Pathways | 25,958 | 56.9% |
| Side Effects | 5,548 | 12.2% |
| Drugs | 4,313 | 9.4% |
| Categories | 3,619 | 7.9% |
| Proteins | 3,176 | 7.0% |
| Diseases | 3,041 | 6.7% |

### Edge Types

| Edge Type | Count | Percentage |
|-----------|-------|------------|
| Drug-Drug Interactions | 759,774 | 62.7% |
| Drug-Side Effect | 265,238 | 21.9% |
| Drug-Category | 70,618 | 5.8% |
| Drug-Disease | 63,278 | 5.2% |
| Drug-Pathway | 31,207 | 2.6% |
| Drug-Protein | 21,559 | 1.8% |

### DDI Severity Distribution (Recalibrated)

| Severity | Count | Percentage |
|----------|-------|------------|
| Moderate | 608,742 | 80.1% |
| Minor | 70,682 | 9.3% |
| Contraindicated | 44,306 | 5.8% |
| Major | 36,044 | 4.7% |

## Data Sources

| Source | Entities | Matching Method |
|--------|----------|-----------------|
| **DrugBank** | Drugs, Proteins, Pathways, Categories, SNPs | DrugBank ID (exact) |
| **SIDER** | Side Effects | Drug name (exact) |
| **CTD** | Disease Associations | CAS number, Drug name |

### Data Provenance

All relationships in the knowledge graph are fact-based with exact identifier matching:
- **DrugBank ID matching**: 4,313 drugs
- **SIDER drug name matching**: 1,110 drugs
- **CTD CAS number matching**: 52,753 associations
- **CTD drug name matching**: 10,525 associations

## Methodology Highlights

### 1. Fact-Based Construction
- **No similarity-based matching** - only exact identifier/name matches
- **Full provenance tracking** - every edge records match type and value
- **Multi-source integration** - DrugBank, SIDER, CTD

### 2. Severity Recalibration
- **Rule-Based Classification** using empirically-derived keyword weights
- **DDInter Validation**: 66.4% exact accuracy, Cohen's κ = +0.096
- **Percentile Thresholds**: Optimized via grid search (P96/P92/P2)

### 3. Rich Biological Context
- **Protein targets**: Enzymes, carriers, transporters
- **Pharmacogenomics**: SNP effects and adverse reactions
- **Clinical**: Side effects, disease associations, drug categories

## Citation

If using this knowledge graph, please cite:
- DrugBank (Wishart et al., 2018, NAR)
- SIDER (Kuhn et al., 2016, NAR)
- CTD (Davis et al., 2021, NAR)
- DDInter (Xiong et al., 2022, NAR)
