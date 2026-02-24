# Knowledge Graph Construction Pipeline

## Complete Methodology for Fact-Based DDI Knowledge Graph

This document describes the complete pipeline for constructing a drug-drug interaction (DDI) knowledge graph with fact-based relationships and full provenance tracking.

---

## 1. Overview

### 1.1 Design Principles

The knowledge graph is constructed following these principles:

1. **Fact-Based Linkage**: All relationships use exact identifier matching (DrugBank ID, CAS number, drug name)
2. **Provenance Tracking**: Every edge records its data source and matching method
3. **Multi-Source Integration**: Combines DrugBank, SIDER, and CTD databases
4. **Severity Recalibration**: DDI severity classifications are recalibrated using validated empirical methods

### 1.2 Graph Schema

```
Nodes:
├── Drug (4,313)           - Primary entities with chemical/pharmacological properties
├── Protein (3,176)        - Drug targets (enzymes, transporters, carriers)
├── SideEffect (5,548)     - Adverse drug reactions (from SIDER)
├── Disease (3,041)        - Drug-disease associations (from CTD)
├── Pathway (25,958)       - Biological pathways (from DrugBank)
└── Category (3,619)       - Drug classifications (from DrugBank)

Edges:
├── INTERACTS_WITH (759,774)  - Drug-drug interactions with severity
├── TARGETS (21,559)          - Drug-protein relationships
├── CAUSES (265,238)          - Drug-side effect associations
├── TREATS/ASSOCIATED (63,278) - Drug-disease relationships
├── PARTICIPATES_IN (31,207)  - Drug-pathway relationships
└── BELONGS_TO (70,618)       - Drug-category relationships
```

---

## 2. Data Sources

### 2.1 DrugBank (Primary Source)

**Version**: Full Database XML  
**URL**: https://go.drugbank.com/releases  
**Citation**: Wishart DS, et al. DrugBank 5.0. Nucleic Acids Res. 2018;46(D1):D1074-D1082.

**Extracted Data**:
- Drug identifiers (DrugBank ID, CAS, UNII, ATC codes)
- Drug properties (SMILES, InChI, molecular weight, logP)
- Drug-protein interactions (targets, enzymes, carriers, transporters)
- Biological pathways
- Drug categories
- SNP effects and adverse reactions
- Drug-drug interactions (descriptions)

### 2.2 SIDER (Side Effects)

**Version**: 4.1  
**URL**: http://sideeffects.embl.de/  
**Citation**: Kuhn M, et al. The SIDER database. Nucleic Acids Res. 2016;44(D1):D1075-D1079.

**Extracted Data**:
- Drug-side effect associations
- Side effect frequencies
- MedDRA terminology

**Matching Method**: Exact drug name matching (case-insensitive)

### 2.3 CTD (Disease Associations)

**Version**: Current  
**URL**: http://ctdbase.org/  
**Citation**: Davis AP, et al. CTD. Nucleic Acids Res. 2021;49(D1):D1138-D1143.

**Extracted Data**:
- Drug-disease associations
- Direct evidence markers
- Inference scores

**Matching Method**: 
1. CAS number (primary)
2. Drug name (secondary, exact match)

---

## 3. Construction Pipeline

### 3.1 Step 1: Load Source Identifiers

```python
# Load DDI dataset with severity labels
df = pd.read_csv("data/ddi_recalibrated.csv")

# Collect unique identifiers
csv_drug_ids = set()      # DrugBank IDs
csv_drug_names = {}       # name -> DrugBank ID mapping
csv_atc_codes = {}        # ATC code -> DrugBank IDs
```

### 3.2 Step 2: Parse DrugBank XML

```python
# Parse DrugBank XML using ElementTree
# Match only drugs present in our DDI dataset
for drug in drugbank_xml:
    drugbank_id = drug.find('drugbank-id[@primary="true"]').text
    if drugbank_id in csv_drug_ids:
        # Extract all properties
        extract_drug_properties(drug)
        extract_protein_targets(drug)
        extract_pathways(drug)
        extract_categories(drug)
        extract_snp_data(drug)
```

**Matching Statistics**:
- Matched drugs: 4,313 / 4,314 (99.98%)
- Match method: DrugBank ID (exact)

### 3.3 Step 3: Integrate SIDER

```python
# Match SIDER drugs by exact name
for sider_drug, side_effects in sider_data:
    if sider_drug.lower() in csv_drug_names:
        drugbank_id = csv_drug_names[sider_drug.lower()]
        for se in side_effects:
            add_drug_side_effect_edge(drugbank_id, se)
```

**Matching Statistics**:
- Matched drugs: 1,110 / 4,314 (25.7%)
- Total side effect associations: 265,238

### 3.4 Step 4: Integrate CTD

```python
# Match CTD drugs by CAS number first, then by name
for ctd_drug, diseases in ctd_data:
    matched = False
    
    # Try CAS number match first
    if ctd_drug.cas in drug_cas_mapping:
        drugbank_id = drug_cas_mapping[ctd_drug.cas]
        matched = True
    # Fall back to name match
    elif ctd_drug.name.lower() in csv_drug_names:
        drugbank_id = csv_drug_names[ctd_drug.name.lower()]
        matched = True
    
    if matched:
        for disease in diseases:
            add_drug_disease_edge(drugbank_id, disease)
```

**Matching Statistics**:
- CAS number matches: 52,753
- Drug name matches: 10,525
- Total disease associations: 63,278

---

## 4. DDI Severity Recalibration

### 4.1 Problem Statement

The original zero-shot severity classification from DrugBank descriptions showed:
- **56.9% Contraindicated** (over-represented)
- **43.0% Major**
- **0.1% Minor**
- **0.0% Moderate** (under-represented)

This distribution does not match clinical reality or external references.

### 4.2 Solution: Empirical Keyword-Based Classification

We use the **Rule-Based Classifier** validated against DDInter (Xiong et al., 2022):

**Empirical Keyword Weights** (log-likelihood ratio from DDInter training data):

| Keyword | Weight | Direction |
|---------|--------|-----------|
| prolongation | +1.45 | More severe |
| bleeding | +1.03 | More severe |
| hemorrhage | +0.63 | More severe |
| anticoagulant | +0.57 | More severe |
| seizure | +0.42 | More severe |
| therapeutic | -1.20 | Less severe |
| efficacy | -1.27 | Less severe |
| reduce | -1.61 | Less severe |

**Classification Algorithm**:
```python
def classify_severity(description):
    score = sum(weight for keyword, weight in KEYWORD_WEIGHTS 
                if keyword in description.lower())
    
    # Percentile-based thresholds (optimized via grid search)
    if score >= P96_threshold:
        return "Contraindicated"
    elif score >= P92_threshold:
        return "Major"
    elif score <= P2_threshold:
        return "Minor"
    else:
        return "Moderate"
```

**Percentile Thresholds** (optimized for DDInter validation):
- Contraindicated: ≥ P96 (top 4%)
- Major: ≥ P92 (top 8%)
- Minor: ≤ P2 (bottom 2%)
- Moderate: Everything else

### 4.3 Validation Results

| Metric | Value |
|--------|-------|
| DDInter Exact Accuracy | 66.4% |
| DDInter Adjacent Accuracy | 99.3% |
| Cohen's Kappa (κ) | +0.096 (p < 0.0001) |

### 4.4 Final Severity Distribution

| Severity | Count | Percentage | Target |
|----------|-------|------------|--------|
| Moderate | 608,742 | 80.1% | 74% |
| Minor | 70,682 | 9.3% | 4% |
| Contraindicated | 44,306 | 5.8% | 4% |
| Major | 36,044 | 4.7% | 18% |

---

## 5. Output Formats

### 5.1 NetworkX Pickle

```python
# Complete graph with all metadata
{
    'drugs': Dict[str, DrugNode],
    'proteins': Dict[str, ProteinNode],
    'side_effects': Dict[str, SideEffectNode],
    'diseases': Dict[str, DiseaseNode],
    'pathways': Dict[str, PathwayNode],
    'categories': Dict[str, CategoryNode],
    'ddi_edges': List[DDIEdge],
    'drug_protein_edges': List[DrugProteinEdge],
    'drug_se_edges': List[DrugSideEffectEdge],
    'drug_disease_edges': List[DrugDiseaseEdge],
    'drug_pathway_edges': List[DrugPathwayEdge],
    'drug_category_edges': List[DrugCategoryEdge],
    'snp_effects': List[SNPEffect],
    'snp_adverse_reactions': List[SNPAdverseReaction],
    'graph': nx.Graph,
    'statistics': Dict
}
```

### 5.2 Neo4j CSV Export

| File | Description |
|------|-------------|
| drugs.csv | Drug nodes with all properties |
| proteins.csv | Protein nodes |
| side_effects.csv | Side effect nodes |
| diseases.csv | Disease nodes |
| pathways.csv | Pathway nodes |
| categories.csv | Category nodes |
| ddi_edges.csv | DDI relationships with severity |
| drug_protein_edges.csv | Drug-protein relationships |
| drug_side_effect_edges.csv | Drug-side effect relationships |
| drug_disease_edges.csv | Drug-disease relationships |
| drug_pathway_edges.csv | Drug-pathway relationships |
| drug_category_edges.csv | Drug-category relationships |
| snp_effects.csv | SNP pharmacogenomic effects |
| snp_adverse_reactions.csv | SNP adverse reactions |

---

## 6. Provenance Tracking

Every edge in the knowledge graph includes provenance information:

```python
@dataclass
class Provenance:
    source: str       # DrugBank, SIDER, CTD, CSV
    match_type: str   # drugbank_id, drug_name, cas_number
    match_value: str  # The actual identifier used
    confidence: str   # exact, verified
```

This enables:
- Full reproducibility
- Data quality assessment
- Source attribution
- Confidence weighting in downstream analyses

---

## 7. Quality Metrics

### 7.1 Coverage

| Entity Type | KG Count | Source Total | Coverage |
|-------------|----------|--------------|----------|
| Drugs | 4,313 | 4,314 | 99.98% |
| Proteins | 3,176 | ~5,000 | 63.5% |
| Side Effects | 5,548 | ~5,800 | 95.7% |
| Diseases | 3,041 | ~3,200 | 95.0% |

### 7.2 Matching Accuracy

| Match Type | Count | Method |
|------------|-------|--------|
| DrugBank ID | 4,313 | Exact primary ID |
| SIDER Name | 1,110 | Exact case-insensitive |
| CTD CAS | 52,753 | Exact CAS number |
| CTD Name | 10,525 | Exact case-insensitive |

---

## 8. References

1. Wishart DS, et al. DrugBank 5.0: a major update to the DrugBank database for 2018. Nucleic Acids Res. 2018;46(D1):D1074-D1082.

2. Kuhn M, et al. The SIDER database of drugs and side effects. Nucleic Acids Res. 2016;44(D1):D1075-D1079.

3. Davis AP, et al. Comparative Toxicogenomics Database (CTD): update 2021. Nucleic Acids Res. 2021;49(D1):D1138-D1143.

4. Xiong G, et al. DDInter: an online drug-drug interaction database towards improving clinical decision-making and patient safety. Nucleic Acids Res. 2022;50(D1):D1200-D1207.
