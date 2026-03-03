#!/usr/bin/env python3
"""
Fact-Based DDI Knowledge Graph Builder.

Builds knowledge graph with ONLY fact-based links using exact identifier matching:
1. DrugBank ID (primary)
2. Drug Name (secondary - exact match)
3. CAS Number
4. ATC Code

All relationships have provenance tracking showing the source and matching method.

Sources:
- DrugBank XML: Targets, Enzymes, Carriers, Transporters, Pathways, Categories, DDIs
- SIDER: Side effects (matched by drug name)
- CTD: Drug-disease associations (matched by CAS or drug name)

NO similarity matching - only exact identifier/name matches.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
import pandas as pd
import networkx as nx
import pickle
import json
import gzip
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DrugBank XML namespace
NS = {'db': 'http://www.drugbank.ca'}


# ---
# DATA CLASSES WITH PROVENANCE
# ---

@dataclass
class Provenance:
    """Tracks the source and method of data linkage."""
    source: str  # DrugBank, SIDER, CTD, etc.
    match_type: str  # drugbank_id, drug_name, cas_number, atc_code
    match_value: str  # The actual value used for matching
    confidence: str = "exact"  # exact, verified


@dataclass
class DrugNode:
    """Drug with all identifiers for matching."""
    drugbank_id: str
    name: str
    name_lower: str = ""  # For case-insensitive matching
    synonyms: List[str] = field(default_factory=list)
    cas_number: str = ""
    unii: str = ""
    atc_codes: List[str] = field(default_factory=list)
    
    # From DrugBank
    type: str = ""  # small molecule, biotech
    description: str = ""
    indication: str = ""
    mechanism_of_action: str = ""
    pharmacodynamics: str = ""
    metabolism: str = ""
    toxicity: str = ""
    half_life: str = ""
    protein_binding: str = ""
    
    # External IDs (for cross-referencing)
    pubchem_cid: str = ""
    chembl_id: str = ""
    kegg_id: str = ""
    
    # Chemical properties
    smiles: str = ""
    inchi: str = ""
    inchi_key: str = ""
    molecular_weight: str = ""
    molecular_formula: str = ""
    logp: str = ""
    
    # Classification
    kingdom: str = ""
    superclass: str = ""
    drug_class: str = ""
    direct_parent: str = ""
    
    # Groups
    groups: List[str] = field(default_factory=list)
    
    # Food interactions
    food_interactions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.name_lower = self.name.lower().strip()


@dataclass
class ProteinNode:
    """Protein target/enzyme/carrier/transporter from DrugBank."""
    id: str
    name: str
    type: str  # target, enzyme, carrier, transporter
    uniprot_id: str = ""
    gene_name: str = ""
    organism: str = ""
    general_function: str = ""
    specific_function: str = ""
    cellular_location: str = ""
    actions: List[str] = field(default_factory=list)
    known_action: str = ""
    
    # For enzymes
    inhibition_strength: str = ""
    induction_strength: str = ""
    
    provenance: Optional[Provenance] = None


@dataclass
class SideEffectNode:
    """Side effect from SIDER with UMLS CUI."""
    umls_cui: str
    name: str
    meddra_type: str = ""  # PT, LLT
    provenance: Optional[Provenance] = None


@dataclass 
class DiseaseNode:
    """Disease from CTD with MeSH ID."""
    mesh_id: str
    name: str
    provenance: Optional[Provenance] = None


@dataclass
class PathwayNode:
    """SMPDB pathway from DrugBank."""
    smpdb_id: str
    name: str
    category: str = ""
    enzymes: List[str] = field(default_factory=list)
    provenance: Optional[Provenance] = None


@dataclass
class CategoryNode:
    """MeSH category from DrugBank."""
    name: str
    mesh_id: str = ""
    provenance: Optional[Provenance] = None


# ---
# EDGE TYPES WITH PROVENANCE
# ---

@dataclass
class DDIEdge:
    """Drug-drug interaction from DrugBank or CSV."""
    drug1_id: str
    drug2_id: str
    description: str = ""
    severity: str = ""
    provenance: Optional[Provenance] = None


@dataclass
class DrugProteinEdge:
    """Drug-protein relationship from DrugBank."""
    drug_id: str
    protein_id: str
    type: str  # target, enzyme, carrier, transporter
    actions: List[str] = field(default_factory=list)
    known_action: str = ""
    inhibition_strength: str = ""
    induction_strength: str = ""
    provenance: Optional[Provenance] = None


@dataclass
class DrugSideEffectEdge:
    """Drug-side effect from SIDER - matched by drug name."""
    drug_id: str
    side_effect_id: str
    frequency: str = ""
    provenance: Optional[Provenance] = None


@dataclass
class DrugDiseaseEdge:
    """Drug-disease from CTD - matched by CAS or drug name."""
    drug_id: str
    disease_id: str
    relationship_type: str = ""  # therapeutic, marker/mechanism
    provenance: Optional[Provenance] = None


@dataclass
class DrugPathwayEdge:
    """Drug-pathway from DrugBank."""
    drug_id: str
    pathway_id: str
    provenance: Optional[Provenance] = None


@dataclass
class DrugCategoryEdge:
    """Drug-category from DrugBank."""
    drug_id: str
    category_name: str
    provenance: Optional[Provenance] = None


@dataclass
class SNPEffectData:
    """SNP effect on drug response from DrugBank."""
    drug_id: str
    protein_name: str
    gene_symbol: str
    uniprot_id: str
    rs_id: str
    allele: str
    defining_change: str
    description: str
    pubmed_id: str = ""
    provenance: Optional[Provenance] = None


# ---
# FACT-BASED KNOWLEDGE GRAPH BUILDER
# ---

class FactBasedKGBuilder:
    """
    Builds knowledge graph using ONLY fact-based exact matches.
    No similarity matching - all links are verified by identifier.
    """
    
    def __init__(self, csv_path: str, xml_path: str):
        self.csv_path = csv_path
        self.xml_path = xml_path
        
        # Drug lookups (built from CSV and XML)
        self.csv_drug_ids: Set[str] = set()
        self.csv_drug_names: Dict[str, str] = {}  # name_lower -> drugbank_id
        self.csv_atc_codes: Dict[str, Set[str]] = defaultdict(set)  # atc -> drug_ids
        
        # Nodes
        self.drugs: Dict[str, DrugNode] = {}
        self.proteins: Dict[str, ProteinNode] = {}
        self.side_effects: Dict[str, SideEffectNode] = {}
        self.diseases: Dict[str, DiseaseNode] = {}
        self.pathways: Dict[str, PathwayNode] = {}
        self.categories: Dict[str, CategoryNode] = {}
        
        # Edges
        self.ddi_edges: List[DDIEdge] = []
        self.drug_protein_edges: List[DrugProteinEdge] = []
        self.drug_se_edges: List[DrugSideEffectEdge] = []
        self.drug_disease_edges: List[DrugDiseaseEdge] = []
        self.drug_pathway_edges: List[DrugPathwayEdge] = []
        self.drug_category_edges: List[DrugCategoryEdge] = []
        self.snp_effects: List[SNPEffectData] = []
        self.snp_adverse_reactions: List[SNPEffectData] = []
        
        # Additional lookups built from XML parsing
        self.drug_by_name: Dict[str, str] = {}  # name_lower -> drugbank_id
        self.drug_by_cas: Dict[str, str] = {}  # cas_number -> drugbank_id
        self.drug_by_synonym: Dict[str, str] = {}  # synonym_lower -> drugbank_id
        
        # Statistics
        self.stats = {
            'matching': defaultdict(int),
            'sources': defaultdict(int),
        }
    
    # ---
    # STEP 1: Load CSV to get source drug identifiers
    # ---
    
    def load_csv_identifiers(self) -> None:
        """Load drug identifiers from the source CSV file."""
        logger.info(f"Loading identifiers from {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Collect all unique drug IDs and names
        for col_suffix in ['1', '2']:
            id_col = f'drugbank_id_{col_suffix}'
            name_col = f'drug_name_{col_suffix}'
            atc_col = f'atc_{col_suffix}'
            
            for _, row in df.iterrows():
                drug_id = row[id_col]
                drug_name = str(row[name_col]).strip()
                
                self.csv_drug_ids.add(drug_id)
                self.csv_drug_names[drug_name.lower()] = drug_id
                
                # Parse ATC codes
                atc_str = str(row[atc_col])
                if atc_str and atc_str != 'nan':
                    # Parse ATC list like "['B01AE02']"
                    import ast
                    try:
                        atc_list = ast.literal_eval(atc_str)
                        for atc in atc_list:
                            self.csv_atc_codes[atc].add(drug_id)
                    except:
                        pass
        
        # Determine which severity column to use (prefer calibrated > recalibrated > label)
        severity_col = 'severity_label'
        if 'severity_calibrated' in df.columns:
            severity_col = 'severity_calibrated'
            logger.info(f"  Using recalibrated severity column: {severity_col}")
        elif 'severity_recalibrated' in df.columns:
            severity_col = 'severity_recalibrated'
            logger.info(f"  Using recalibrated severity column: {severity_col}")
        
        # Store DDI edges from CSV (these are ground truth)
        for _, row in df.iterrows():
            # Get severity - use recalibrated if available
            severity = row.get(severity_col, row['severity_label'])
            
            self.ddi_edges.append(DDIEdge(
                drug1_id=row['drugbank_id_1'],
                drug2_id=row['drugbank_id_2'],
                description=row['interaction_description'],
                severity=severity,
                provenance=Provenance(
                    source="CSV",
                    match_type="drugbank_id",
                    match_value=f"{row['drugbank_id_1']}-{row['drugbank_id_2']}",
                ),
            ))
        
        logger.info(f"  Found {len(self.csv_drug_ids)} unique DrugBank IDs")
        logger.info(f"  Found {len(self.csv_drug_names)} unique drug names")
        logger.info(f"  Found {len(self.ddi_edges)} DDI edges from CSV")
    
    # ---
    # STEP 2: Parse DrugBank XML for exact ID matches
    # ---
    
    def _get_text(self, elem, tag: str) -> str:
        """Safely get text from XML element."""
        child = elem.find(f'db:{tag}', NS)
        if child is not None and child.text:
            return child.text.strip()
        return ""
    
    def _get_all_text(self, elem, tag: str) -> List[str]:
        """Get text from all matching child elements."""
        return [c.text.strip() for c in elem.findall(f'db:{tag}', NS) if c.text]
    
    def parse_drugbank_xml(self) -> None:
        """Parse DrugBank XML and extract data for CSV drugs only."""
        logger.info(f"Parsing DrugBank XML: {self.xml_path}")
        
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        drug_count = 0
        matched_count = 0
        
        for drug_elem in root.findall('db:drug', NS):
            # Get primary DrugBank ID
            drugbank_id = ""
            for id_elem in drug_elem.findall('db:drugbank-id', NS):
                if id_elem.get('primary') == 'true':
                    drugbank_id = id_elem.text.strip() if id_elem.text else ""
                    break
            
            if not drugbank_id:
                continue
            
            drug_count += 1
            
            # Check if this drug is in our CSV
            drug_name = self._get_text(drug_elem, 'name')
            
            # Match by DrugBank ID first, then by name
            is_match = False
            match_type = ""
            match_value = ""
            
            if drugbank_id in self.csv_drug_ids:
                is_match = True
                match_type = "drugbank_id"
                match_value = drugbank_id
            elif drug_name.lower() in self.csv_drug_names:
                is_match = True
                match_type = "drug_name"
                match_value = drug_name
                # Update drug ID mapping if matched by name
                drugbank_id = self.csv_drug_names[drug_name.lower()]
            
            if not is_match:
                continue
            
            matched_count += 1
            self.stats['matching'][match_type] += 1
            
            # Create drug node with full data
            drug = DrugNode(
                drugbank_id=drugbank_id,
                name=drug_name,
                type=drug_elem.get('type', ''),
                description=self._get_text(drug_elem, 'description'),
                cas_number=self._get_text(drug_elem, 'cas-number'),
                unii=self._get_text(drug_elem, 'unii'),
            )
            
            # Synonyms
            synonyms_elem = drug_elem.find('db:synonyms', NS)
            if synonyms_elem is not None:
                for syn in synonyms_elem.findall('db:synonym', NS):
                    if syn.text:
                        drug.synonyms.append(syn.text.strip())
                        self.drug_by_synonym[syn.text.strip().lower()] = drugbank_id
            
            # Build lookup by name and CAS
            self.drug_by_name[drug.name_lower] = drugbank_id
            if drug.cas_number:
                self.drug_by_cas[drug.cas_number] = drugbank_id
            
            # Groups
            groups_elem = drug_elem.find('db:groups', NS)
            if groups_elem is not None:
                drug.groups = self._get_all_text(groups_elem, 'group')
            
            # Pharmacology
            drug.indication = self._get_text(drug_elem, 'indication')
            drug.pharmacodynamics = self._get_text(drug_elem, 'pharmacodynamics')
            drug.mechanism_of_action = self._get_text(drug_elem, 'mechanism-of-action')
            drug.toxicity = self._get_text(drug_elem, 'toxicity')
            drug.metabolism = self._get_text(drug_elem, 'metabolism')
            drug.half_life = self._get_text(drug_elem, 'half-life')
            drug.protein_binding = self._get_text(drug_elem, 'protein-binding')
            
            # Classification
            class_elem = drug_elem.find('db:classification', NS)
            if class_elem is not None:
                drug.kingdom = self._get_text(class_elem, 'kingdom')
                drug.superclass = self._get_text(class_elem, 'superclass')
                drug.drug_class = self._get_text(class_elem, 'class')
                drug.direct_parent = self._get_text(class_elem, 'direct-parent')
            
            # ATC codes
            atc_elem = drug_elem.find('db:atc-codes', NS)
            if atc_elem is not None:
                for code in atc_elem.findall('db:atc-code', NS):
                    atc_code = code.get('code', '')
                    if atc_code:
                        drug.atc_codes.append(atc_code)
            
            # External identifiers
            ext_ids_elem = drug_elem.find('db:external-identifiers', NS)
            if ext_ids_elem is not None:
                for ext_id in ext_ids_elem.findall('db:external-identifier', NS):
                    resource = self._get_text(ext_id, 'resource')
                    identifier = self._get_text(ext_id, 'identifier')
                    if resource == 'PubChem Compound':
                        drug.pubchem_cid = identifier
                    elif resource == 'ChEMBL':
                        drug.chembl_id = identifier
                    elif resource in ['KEGG Drug', 'KEGG Compound']:
                        drug.kegg_id = identifier
            
            # Calculated properties
            calc_props = drug_elem.find('db:calculated-properties', NS)
            if calc_props is not None:
                for prop in calc_props.findall('db:property', NS):
                    kind = self._get_text(prop, 'kind')
                    value = self._get_text(prop, 'value')
                    if kind == 'SMILES':
                        drug.smiles = value
                    elif kind == 'InChI':
                        drug.inchi = value
                    elif kind == 'InChIKey':
                        drug.inchi_key = value
                    elif kind == 'Molecular Weight':
                        drug.molecular_weight = value
                    elif kind == 'Molecular Formula':
                        drug.molecular_formula = value
                    elif kind == 'logP':
                        drug.logp = value
            
            # Food interactions
            food_elem = drug_elem.find('db:food-interactions', NS)
            if food_elem is not None:
                drug.food_interactions = self._get_all_text(food_elem, 'food-interaction')
            
            self.drugs[drugbank_id] = drug
            
            # Parse protein relationships (targets, enzymes, carriers, transporters)
            prov = Provenance(source="DrugBank", match_type=match_type, match_value=match_value)
            
            for protein_type in ['targets', 'enzymes', 'carriers', 'transporters']:
                proteins_elem = drug_elem.find(f'db:{protein_type}', NS)
                if proteins_elem is not None:
                    singular = protein_type[:-1]  # targets -> target
                    for protein_elem in proteins_elem.findall(f'db:{singular}', NS):
                        self._parse_protein(protein_elem, singular, drugbank_id, prov)
            
            # Parse pathways
            pathways_elem = drug_elem.find('db:pathways', NS)
            if pathways_elem is not None:
                for pathway_elem in pathways_elem.findall('db:pathway', NS):
                    self._parse_pathway(pathway_elem, drugbank_id, prov)
            
            # Parse categories
            categories_elem = drug_elem.find('db:categories', NS)
            if categories_elem is not None:
                for cat_elem in categories_elem.findall('db:category', NS):
                    self._parse_category(cat_elem, drugbank_id, prov)
            
            # Parse SNP effects
            snp_effects_elem = drug_elem.find('db:snp-effects', NS)
            if snp_effects_elem is not None:
                for effect in snp_effects_elem.findall('db:effect', NS):
                    self._parse_snp_effect(effect, drugbank_id, prov, is_adverse=False)
            
            # Parse SNP adverse reactions
            snp_adr_elem = drug_elem.find('db:snp-adverse-drug-reactions', NS)
            if snp_adr_elem is not None:
                for reaction in snp_adr_elem.findall('db:reaction', NS):
                    self._parse_snp_effect(reaction, drugbank_id, prov, is_adverse=True)
            
            if matched_count % 500 == 0:
                logger.info(f"  Processed {matched_count} matched drugs...")
        
        logger.info(f"  Total drugs in XML: {drug_count}")
        logger.info(f"  Matched drugs: {matched_count}")
        logger.info(f"  Matching breakdown: {dict(self.stats['matching'])}")
    
    def _parse_protein(self, elem, protein_type: str, drug_id: str, prov: Provenance) -> None:
        """Parse protein target/enzyme/carrier/transporter."""
        protein_id = self._get_text(elem, 'id')
        if not protein_id:
            return
        
        if protein_id not in self.proteins:
            protein = ProteinNode(
                id=protein_id,
                name=self._get_text(elem, 'name'),
                type=protein_type,
                organism=self._get_text(elem, 'organism'),
                known_action=self._get_text(elem, 'known-action'),
                provenance=prov,
            )
            
            # Actions
            actions_elem = elem.find('db:actions', NS)
            if actions_elem is not None:
                protein.actions = self._get_all_text(actions_elem, 'action')
            
            # For enzymes
            if protein_type == 'enzyme':
                protein.inhibition_strength = self._get_text(elem, 'inhibition-strength')
                protein.induction_strength = self._get_text(elem, 'induction-strength')
            
            # Polypeptide info
            polypeptide = elem.find('db:polypeptide', NS)
            if polypeptide is not None:
                protein.gene_name = self._get_text(polypeptide, 'gene-name')
                protein.general_function = self._get_text(polypeptide, 'general-function')
                protein.specific_function = self._get_text(polypeptide, 'specific-function')
                protein.cellular_location = self._get_text(polypeptide, 'cellular-location')
                
                # UniProt ID
                ext_ids = polypeptide.find('db:external-identifiers', NS)
                if ext_ids is not None:
                    for ext_id in ext_ids.findall('db:external-identifier', NS):
                        if self._get_text(ext_id, 'resource') in ['UniProtKB', 'UniProt Accession']:
                            protein.uniprot_id = self._get_text(ext_id, 'identifier')
                            break
            
            self.proteins[protein_id] = protein
        
        # Create drug-protein edge
        edge = DrugProteinEdge(
            drug_id=drug_id,
            protein_id=protein_id,
            type=protein_type,
            actions=self.proteins[protein_id].actions.copy(),
            known_action=self.proteins[protein_id].known_action,
            provenance=prov,
        )
        if protein_type == 'enzyme':
            edge.inhibition_strength = self.proteins[protein_id].inhibition_strength
            edge.induction_strength = self.proteins[protein_id].induction_strength
        
        self.drug_protein_edges.append(edge)
        self.stats['sources']['DrugBank-protein'] += 1
    
    def _parse_pathway(self, elem, drug_id: str, prov: Provenance) -> None:
        """Parse pathway."""
        smpdb_id = self._get_text(elem, 'smpdb-id')
        if not smpdb_id:
            return
        
        if smpdb_id not in self.pathways:
            pathway = PathwayNode(
                smpdb_id=smpdb_id,
                name=self._get_text(elem, 'name'),
                category=self._get_text(elem, 'category'),
                provenance=prov,
            )
            
            enzymes_elem = elem.find('db:enzymes', NS)
            if enzymes_elem is not None:
                pathway.enzymes = self._get_all_text(enzymes_elem, 'uniprot-id')
            
            self.pathways[smpdb_id] = pathway
        
        self.drug_pathway_edges.append(DrugPathwayEdge(
            drug_id=drug_id,
            pathway_id=smpdb_id,
            provenance=prov,
        ))
        self.stats['sources']['DrugBank-pathway'] += 1
    
    def _parse_category(self, elem, drug_id: str, prov: Provenance) -> None:
        """Parse category."""
        cat_name = self._get_text(elem, 'category')
        if not cat_name:
            return
        
        if cat_name not in self.categories:
            self.categories[cat_name] = CategoryNode(
                name=cat_name,
                mesh_id=self._get_text(elem, 'mesh-id'),
                provenance=prov,
            )
        
        self.drug_category_edges.append(DrugCategoryEdge(
            drug_id=drug_id,
            category_name=cat_name,
            provenance=prov,
        ))
        self.stats['sources']['DrugBank-category'] += 1
    
    def _parse_snp_effect(self, elem, drug_id: str, prov: Provenance, is_adverse: bool) -> None:
        """Parse SNP effect or adverse reaction."""
        snp = SNPEffectData(
            drug_id=drug_id,
            protein_name=self._get_text(elem, 'protein-name'),
            gene_symbol=self._get_text(elem, 'gene-symbol'),
            uniprot_id=self._get_text(elem, 'uniprot-id'),
            rs_id=self._get_text(elem, 'rs-id'),
            allele=self._get_text(elem, 'allele'),
            defining_change=self._get_text(elem, 'defining-change') if not is_adverse else "",
            description=self._get_text(elem, 'adverse-reaction') if is_adverse else self._get_text(elem, 'description'),
            pubmed_id=self._get_text(elem, 'pubmed-id'),
            provenance=prov,
        )
        
        if is_adverse:
            self.snp_adverse_reactions.append(snp)
            self.stats['sources']['DrugBank-snp-adr'] += 1
        else:
            self.snp_effects.append(snp)
            self.stats['sources']['DrugBank-snp'] += 1
    
    # ---
    # STEP 3: Integrate SIDER using exact drug name matching
    # ---
    
    def integrate_sider(self, sider_dir: str = "external_data/sider") -> None:
        """Load SIDER side effects using exact drug name matching."""
        logger.info("\n--- Integrating SIDER (exact drug name match) ---")
        
        se_file = Path(sider_dir) / "meddra_all_se.tsv.gz"
        names_file = Path(sider_dir) / "drug_names.tsv"
        
        if not se_file.exists():
            logger.warning(f"SIDER file not found: {se_file}")
            return
        
        # Load drug name mappings from SIDER
        sider_drug_names: Dict[str, str] = {}  # stitch_id -> drug_name
        if names_file.exists():
            with open(names_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        stitch_id = parts[0]
                        name = parts[1].strip()
                        sider_drug_names[stitch_id] = name
        
        # Build reverse mapping: drug_name_lower -> stitch_ids
        name_to_stitch: Dict[str, List[str]] = defaultdict(list)
        for stitch_id, name in sider_drug_names.items():
            name_to_stitch[name.lower()].append(stitch_id)
        
        # Build our drug name to drugbank_id mapping
        our_drug_names: Dict[str, str] = {}  # name_lower -> drugbank_id
        for drug_id, drug in self.drugs.items():
            our_drug_names[drug.name_lower] = drug_id
            for syn in drug.synonyms:
                our_drug_names[syn.lower()] = drug_id
        
        # Find matching drug names
        matched_stitch_to_drugbank: Dict[str, str] = {}
        for name_lower, drug_id in our_drug_names.items():
            if name_lower in name_to_stitch:
                for stitch_id in name_to_stitch[name_lower]:
                    matched_stitch_to_drugbank[stitch_id] = drug_id
                    self.stats['matching']['sider_drug_name'] += 1
        
        logger.info(f"  Matched {len(matched_stitch_to_drugbank)} SIDER drugs by name")
        
        # Load side effects for matched drugs
        try:
            with gzip.open(se_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        stitch_flat = parts[0]
                        umls_cui = parts[2]
                        meddra_type = parts[3]
                        se_name = parts[5]
                        
                        # Check if we have this drug
                        drugbank_id = matched_stitch_to_drugbank.get(stitch_flat)
                        
                        if drugbank_id:
                            # Add side effect node
                            if umls_cui not in self.side_effects:
                                self.side_effects[umls_cui] = SideEffectNode(
                                    umls_cui=umls_cui,
                                    name=se_name,
                                    meddra_type=meddra_type,
                                    provenance=Provenance(
                                        source="SIDER",
                                        match_type="umls_cui",
                                        match_value=umls_cui,
                                    ),
                                )
                            
                            # Add drug-SE edge with provenance
                            drug_name = self.drugs[drugbank_id].name
                            self.drug_se_edges.append(DrugSideEffectEdge(
                                drug_id=drugbank_id,
                                side_effect_id=umls_cui,
                                provenance=Provenance(
                                    source="SIDER",
                                    match_type="drug_name",
                                    match_value=drug_name,
                                ),
                            ))
                            self.stats['sources']['SIDER-side_effect'] += 1
            
            logger.info(f"  Loaded {len(self.side_effects)} unique side effects")
            logger.info(f"  Loaded {len(self.drug_se_edges)} drug-side effect associations")
            
        except Exception as e:
            logger.error(f"Error loading SIDER: {e}")
    
    # ---
    # STEP 4: Integrate CTD using CAS number or drug name
    # ---
    
    def integrate_ctd(self, ctd_dir: str = "external_data/ctd") -> None:
        """Load CTD drug-disease associations using CAS or drug name matching."""
        logger.info("\n--- Integrating CTD (CAS/drug name match) ---")
        
        filepath = Path(ctd_dir) / "CTD_chemicals_diseases.tsv.gz"
        if not filepath.exists():
            logger.warning(f"CTD file not found: {filepath}")
            return
        
        # Build our lookup maps
        our_cas_to_drug: Dict[str, str] = {}
        our_name_to_drug: Dict[str, str] = {}
        
        for drug_id, drug in self.drugs.items():
            if drug.cas_number:
                our_cas_to_drug[drug.cas_number] = drug_id
            our_name_to_drug[drug.name_lower] = drug_id
            for syn in drug.synonyms:
                our_name_to_drug[syn.lower()] = drug_id
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        chemical_name = parts[0].strip()
                        cas_rn = parts[2].strip() if len(parts) > 2 else ""
                        disease_name = parts[3] if len(parts) > 3 else ""
                        disease_id = parts[4] if len(parts) > 4 else ""
                        direct_evidence = parts[5] if len(parts) > 5 else ""
                        
                        # Only include direct evidence
                        if not direct_evidence:
                            continue
                        
                        # Try to match by CAS first, then by name
                        drugbank_id = None
                        match_type = ""
                        match_value = ""
                        
                        if cas_rn and cas_rn in our_cas_to_drug:
                            drugbank_id = our_cas_to_drug[cas_rn]
                            match_type = "cas_number"
                            match_value = cas_rn
                        elif chemical_name.lower() in our_name_to_drug:
                            drugbank_id = our_name_to_drug[chemical_name.lower()]
                            match_type = "drug_name"
                            match_value = chemical_name
                        
                        if drugbank_id and disease_id:
                            # Add disease node
                            if disease_id not in self.diseases:
                                self.diseases[disease_id] = DiseaseNode(
                                    mesh_id=disease_id,
                                    name=disease_name,
                                    provenance=Provenance(
                                        source="CTD",
                                        match_type="mesh_id",
                                        match_value=disease_id,
                                    ),
                                )
                            
                            # Add drug-disease edge with provenance
                            self.drug_disease_edges.append(DrugDiseaseEdge(
                                drug_id=drugbank_id,
                                disease_id=disease_id,
                                relationship_type=direct_evidence,
                                provenance=Provenance(
                                    source="CTD",
                                    match_type=match_type,
                                    match_value=match_value,
                                ),
                            ))
                            self.stats['sources']['CTD-disease'] += 1
                            self.stats['matching'][f'ctd_{match_type}'] += 1
            
            logger.info(f"  Loaded {len(self.diseases)} unique diseases")
            logger.info(f"  Loaded {len(self.drug_disease_edges)} drug-disease associations")
            
        except Exception as e:
            logger.error(f"Error loading CTD: {e}")
    
    # ---
    # STEP 5: Build NetworkX Graph
    # ---
    
    def build_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph with all data."""
        logger.info("\n--- Building NetworkX Graph ---")
        
        G = nx.MultiDiGraph()
        
        # Add drug nodes
        for drug_id, drug in self.drugs.items():
            G.add_node(
                drug_id,
                node_type='drug',
                name=drug.name,
                drug_type=drug.type,
                cas_number=drug.cas_number,
                atc_codes=drug.atc_codes,
                indication=drug.indication[:500] if drug.indication else "",
                mechanism=drug.mechanism_of_action[:500] if drug.mechanism_of_action else "",
                smiles=drug.smiles,
                molecular_weight=drug.molecular_weight,
            )
        
        # Add protein nodes
        for protein_id, protein in self.proteins.items():
            G.add_node(
                protein_id,
                node_type='protein',
                subtype=protein.type,
                name=protein.name,
                uniprot_id=protein.uniprot_id,
                gene_name=protein.gene_name,
                source=protein.provenance.source if protein.provenance else "",
            )
        
        # Add side effect nodes
        for se_id, se in self.side_effects.items():
            G.add_node(
                f"SE:{se_id}",
                node_type='side_effect',
                name=se.name,
                umls_cui=se.umls_cui,
                meddra_type=se.meddra_type,
            )
        
        # Add disease nodes
        for disease_id, disease in self.diseases.items():
            G.add_node(
                f"DIS:{disease_id}",
                node_type='disease',
                name=disease.name,
                mesh_id=disease.mesh_id,
            )
        
        # Add pathway nodes
        for pathway_id, pathway in self.pathways.items():
            G.add_node(
                pathway_id,
                node_type='pathway',
                name=pathway.name,
                category=pathway.category,
            )
        
        # Add category nodes
        for cat_name, category in self.categories.items():
            G.add_node(
                f"CAT:{cat_name}",
                node_type='category',
                name=cat_name,
                mesh_id=category.mesh_id,
            )
        
        # Add DDI edges
        for edge in self.ddi_edges:
            if edge.drug1_id in self.drugs and edge.drug2_id in self.drugs:
                G.add_edge(
                    edge.drug1_id,
                    edge.drug2_id,
                    edge_type='ddi',
                    description=edge.description[:300] if edge.description else "",
                    severity=edge.severity,
                    source=edge.provenance.source if edge.provenance else "",
                )
        
        # Add drug-protein edges
        for edge in self.drug_protein_edges:
            if edge.drug_id in self.drugs and edge.protein_id in self.proteins:
                G.add_edge(
                    edge.drug_id,
                    edge.protein_id,
                    edge_type=f'drug_{edge.type}',
                    actions=edge.actions,
                    source='DrugBank',
                    match_type=edge.provenance.match_type if edge.provenance else "",
                )
        
        # Add drug-side effect edges
        for edge in self.drug_se_edges:
            if edge.drug_id in self.drugs:
                G.add_edge(
                    edge.drug_id,
                    f"SE:{edge.side_effect_id}",
                    edge_type='causes_side_effect',
                    source='SIDER',
                    match_type=edge.provenance.match_type if edge.provenance else "",
                    match_value=edge.provenance.match_value if edge.provenance else "",
                )
        
        # Add drug-disease edges
        for edge in self.drug_disease_edges:
            if edge.drug_id in self.drugs:
                G.add_edge(
                    edge.drug_id,
                    f"DIS:{edge.disease_id}",
                    edge_type='drug_disease',
                    relationship_type=edge.relationship_type,
                    source='CTD',
                    match_type=edge.provenance.match_type if edge.provenance else "",
                    match_value=edge.provenance.match_value if edge.provenance else "",
                )
        
        # Add drug-pathway edges
        for edge in self.drug_pathway_edges:
            if edge.drug_id in self.drugs and edge.pathway_id in self.pathways:
                G.add_edge(
                    edge.drug_id,
                    edge.pathway_id,
                    edge_type='in_pathway',
                    source='DrugBank',
                )
        
        # Add drug-category edges
        for edge in self.drug_category_edges:
            if edge.drug_id in self.drugs:
                G.add_edge(
                    edge.drug_id,
                    f"CAT:{edge.category_name}",
                    edge_type='has_category',
                    source='DrugBank',
                )
        
        logger.info(f"  Total nodes: {G.number_of_nodes()}")
        logger.info(f"  Total edges: {G.number_of_edges()}")
        
        return G
    
    # ---
    # STEP 6: Export with full provenance
    # ---
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            'nodes': {
                'drugs': len(self.drugs),
                'proteins': len(self.proteins),
                'side_effects': len(self.side_effects),
                'diseases': len(self.diseases),
                'pathways': len(self.pathways),
                'categories': len(self.categories),
                'total': (len(self.drugs) + len(self.proteins) + len(self.side_effects) +
                         len(self.diseases) + len(self.pathways) + len(self.categories)),
            },
            'edges': {
                'ddi': len(self.ddi_edges),
                'drug_protein': len(self.drug_protein_edges),
                'drug_side_effect': len(self.drug_se_edges),
                'drug_disease': len(self.drug_disease_edges),
                'drug_pathway': len(self.drug_pathway_edges),
                'drug_category': len(self.drug_category_edges),
                'total': (len(self.ddi_edges) + len(self.drug_protein_edges) + 
                         len(self.drug_se_edges) + len(self.drug_disease_edges) +
                         len(self.drug_pathway_edges) + len(self.drug_category_edges)),
            },
            'snp_data': {
                'snp_effects': len(self.snp_effects),
                'snp_adverse_reactions': len(self.snp_adverse_reactions),
            },
            'matching': dict(self.stats['matching']),
            'sources': dict(self.stats['sources']),
        }
    
    def export(self, output_dir: str) -> None:
        """Export knowledge graph with full provenance."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        neo4j_path = output_path / "neo4j_export"
        neo4j_path.mkdir(exist_ok=True)
        
        logger.info(f"\nExporting to {output_dir}")
        
        # Export drugs
        drugs_data = []
        for drug_id, drug in self.drugs.items():
            drugs_data.append({
                'drugbank_id': drug_id,
                'name': drug.name,
                'type': drug.type,
                'cas_number': drug.cas_number,
                'unii': drug.unii,
                'atc_codes': '|'.join(drug.atc_codes),
                'groups': '|'.join(drug.groups),
                'indication': drug.indication[:1000] if drug.indication else "",
                'mechanism_of_action': drug.mechanism_of_action[:1000] if drug.mechanism_of_action else "",
                'smiles': drug.smiles,
                'inchi_key': drug.inchi_key,
                'molecular_weight': drug.molecular_weight,
                'logp': drug.logp,
                'pubchem_cid': drug.pubchem_cid,
                'chembl_id': drug.chembl_id,
                'kegg_id': drug.kegg_id,
            })
        pd.DataFrame(drugs_data).to_csv(neo4j_path / 'drugs.csv', index=False)
        
        # Export proteins
        proteins_data = []
        for protein_id, protein in self.proteins.items():
            proteins_data.append({
                'protein_id': protein_id,
                'name': protein.name,
                'type': protein.type,
                'uniprot_id': protein.uniprot_id,
                'gene_name': protein.gene_name,
                'organism': protein.organism,
                'general_function': protein.general_function[:500] if protein.general_function else "",
                'specific_function': protein.specific_function[:500] if protein.specific_function else "",
                'cellular_location': protein.cellular_location,
                'actions': '|'.join(protein.actions),
                'source': protein.provenance.source if protein.provenance else "",
                'match_type': protein.provenance.match_type if protein.provenance else "",
            })
        pd.DataFrame(proteins_data).to_csv(neo4j_path / 'proteins.csv', index=False)
        
        # Export side effects with provenance
        if self.side_effects:
            se_data = []
            for se_id, se in self.side_effects.items():
                se_data.append({
                    'umls_cui': se.umls_cui,
                    'name': se.name,
                    'meddra_type': se.meddra_type,
                    'source': 'SIDER',
                })
            pd.DataFrame(se_data).to_csv(neo4j_path / 'side_effects.csv', index=False)
        
        # Export diseases with provenance
        if self.diseases:
            disease_data = []
            for disease_id, disease in self.diseases.items():
                disease_data.append({
                    'mesh_id': disease.mesh_id,
                    'name': disease.name,
                    'source': 'CTD',
                })
            pd.DataFrame(disease_data).to_csv(neo4j_path / 'diseases.csv', index=False)
        
        # Export pathways
        pathway_data = []
        for pathway_id, pathway in self.pathways.items():
            pathway_data.append({
                'smpdb_id': pathway_id,
                'name': pathway.name,
                'category': pathway.category,
                'enzymes': '|'.join(pathway.enzymes),
                'source': 'DrugBank',
            })
        pd.DataFrame(pathway_data).to_csv(neo4j_path / 'pathways.csv', index=False)
        
        # Export categories
        category_data = []
        for cat_name, category in self.categories.items():
            category_data.append({
                'name': cat_name,
                'mesh_id': category.mesh_id,
                'source': 'DrugBank',
            })
        pd.DataFrame(category_data).to_csv(neo4j_path / 'categories.csv', index=False)
        
        # Export DDI edges with provenance
        ddi_data = []
        for edge in self.ddi_edges:
            ddi_data.append({
                'drug1_id': edge.drug1_id,
                'drug2_id': edge.drug2_id,
                'description': edge.description[:500] if edge.description else "",
                'severity': edge.severity,
                'source': edge.provenance.source if edge.provenance else "",
                'match_type': edge.provenance.match_type if edge.provenance else "",
            })
        pd.DataFrame(ddi_data).to_csv(neo4j_path / 'ddi_edges.csv', index=False)
        
        # Export drug-protein edges with provenance
        dp_data = []
        for edge in self.drug_protein_edges:
            dp_data.append({
                'drug_id': edge.drug_id,
                'protein_id': edge.protein_id,
                'relationship_type': edge.type,
                'actions': '|'.join(edge.actions),
                'known_action': edge.known_action,
                'inhibition_strength': edge.inhibition_strength,
                'induction_strength': edge.induction_strength,
                'source': 'DrugBank',
                'match_type': edge.provenance.match_type if edge.provenance else "",
                'match_value': edge.provenance.match_value if edge.provenance else "",
            })
        pd.DataFrame(dp_data).to_csv(neo4j_path / 'drug_protein_edges.csv', index=False)
        
        # Export drug-side effect edges with provenance
        if self.drug_se_edges:
            dse_data = []
            for edge in self.drug_se_edges:
                dse_data.append({
                    'drug_id': edge.drug_id,
                    'side_effect_id': edge.side_effect_id,
                    'frequency': edge.frequency,
                    'source': 'SIDER',
                    'match_type': edge.provenance.match_type if edge.provenance else "",
                    'match_value': edge.provenance.match_value if edge.provenance else "",
                })
            pd.DataFrame(dse_data).to_csv(neo4j_path / 'drug_side_effect_edges.csv', index=False)
        
        # Export drug-disease edges with provenance
        if self.drug_disease_edges:
            dd_data = []
            for edge in self.drug_disease_edges:
                dd_data.append({
                    'drug_id': edge.drug_id,
                    'disease_id': edge.disease_id,
                    'relationship_type': edge.relationship_type,
                    'source': 'CTD',
                    'match_type': edge.provenance.match_type if edge.provenance else "",
                    'match_value': edge.provenance.match_value if edge.provenance else "",
                })
            pd.DataFrame(dd_data).to_csv(neo4j_path / 'drug_disease_edges.csv', index=False)
        
        # Export drug-pathway edges
        dp_data = []
        for edge in self.drug_pathway_edges:
            dp_data.append({
                'drug_id': edge.drug_id,
                'pathway_id': edge.pathway_id,
                'source': 'DrugBank',
            })
        pd.DataFrame(dp_data).to_csv(neo4j_path / 'drug_pathway_edges.csv', index=False)
        
        # Export drug-category edges
        dc_data = []
        for edge in self.drug_category_edges:
            dc_data.append({
                'drug_id': edge.drug_id,
                'category_name': edge.category_name,
                'source': 'DrugBank',
            })
        pd.DataFrame(dc_data).to_csv(neo4j_path / 'drug_category_edges.csv', index=False)
        
        # Export SNP effects
        if self.snp_effects:
            snp_data = []
            for snp in self.snp_effects:
                snp_data.append({
                    'drug_id': snp.drug_id,
                    'protein_name': snp.protein_name,
                    'gene_symbol': snp.gene_symbol,
                    'uniprot_id': snp.uniprot_id,
                    'rs_id': snp.rs_id,
                    'allele': snp.allele,
                    'defining_change': snp.defining_change,
                    'description': snp.description[:500] if snp.description else "",
                    'pubmed_id': snp.pubmed_id,
                    'source': 'DrugBank',
                })
            pd.DataFrame(snp_data).to_csv(neo4j_path / 'snp_effects.csv', index=False)
        
        # Export SNP adverse reactions
        if self.snp_adverse_reactions:
            snp_adr_data = []
            for snp in self.snp_adverse_reactions:
                snp_adr_data.append({
                    'drug_id': snp.drug_id,
                    'protein_name': snp.protein_name,
                    'gene_symbol': snp.gene_symbol,
                    'uniprot_id': snp.uniprot_id,
                    'rs_id': snp.rs_id,
                    'allele': snp.allele,
                    'adverse_reaction': snp.description,
                    'pubmed_id': snp.pubmed_id,
                    'source': 'DrugBank',
                })
            pd.DataFrame(snp_adr_data).to_csv(neo4j_path / 'snp_adverse_reactions.csv', index=False)
        
        # Save statistics with provenance tracking
        stats = self.get_statistics()
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save full graph as pickle
        G = self.build_graph()
        with open(output_path / 'knowledge_graph.pkl', 'wb') as f:
            pickle.dump({
                'drugs': self.drugs,
                'proteins': self.proteins,
                'side_effects': self.side_effects,
                'diseases': self.diseases,
                'pathways': self.pathways,
                'categories': self.categories,
                'ddi_edges': self.ddi_edges,
                'drug_protein_edges': self.drug_protein_edges,
                'drug_se_edges': self.drug_se_edges,
                'drug_disease_edges': self.drug_disease_edges,
                'drug_pathway_edges': self.drug_pathway_edges,
                'drug_category_edges': self.drug_category_edges,
                'snp_effects': self.snp_effects,
                'snp_adverse_reactions': self.snp_adverse_reactions,
                'graph': G,
                'statistics': stats,
            }, f)
        
        logger.info("  Export complete!")


# ---
# MAIN
# ---

def main():
    """Build fact-based DDI Knowledge Graph."""
    
    # Use recalibrated severity data (with empirical keyword-based classification)
    csv_path = "data/ddi_recalibrated.csv"
    xml_path = "data/full database.xml"
    output_dir = "knowledge_graph_fact_based"
    
    # Initialize builder
    builder = FactBasedKGBuilder(csv_path, xml_path)
    
    # Step 1: Load identifiers from CSV
    builder.load_csv_identifiers()
    
    # Step 2: Parse DrugBank XML (exact ID/name matches only)
    builder.parse_drugbank_xml()
    
    # Step 3: Integrate SIDER (exact drug name matches)
    builder.integrate_sider()
    
    # Step 4: Integrate CTD (CAS or drug name matches)
    builder.integrate_ctd()
    
    # Get statistics
    stats = builder.get_statistics()
    
    print("\n" + "=" * 70)
    print("FACT-BASED KNOWLEDGE GRAPH STATISTICS")
    print("(All links verified by exact identifier/name matching)")
    print("=" * 70)
    
    print("\nNode Counts:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    
    print("\nEdge Counts:")
    for edge_type, count in stats['edges'].items():
        print(f"  {edge_type}: {count:,}")
    
    print("\nSNP Data (from DrugBank):")
    for snp_type, count in stats['snp_data'].items():
        print(f"  {snp_type}: {count:,}")
    
    print("\nMatching Statistics (how data was linked):")
    for match_type, count in stats['matching'].items():
        print(f"  {match_type}: {count:,}")
    
    print("\nData Sources:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count:,}")
    
    # Export
    builder.export(output_dir)
    
    print(f"\nAll outputs saved to {output_dir}/")
    print("\nProvenance tracked for every relationship - see CSV files for match_type and match_value columns")


if __name__ == '__main__':
    main()
