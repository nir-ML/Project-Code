#!/usr/bin/env python3
"""
API-Based DDI Knowledge Graph Builder

Builds knowledge graph using:
1. DrugBank XML (local file - requires license)
2. OpenFDA API for side effects (public, no license required)
3. PubChem API for drug-disease associations (public, no license required)

This version avoids the need for local SIDER and CTD files by using public APIs.

Usage:
    python build_kg_api_based.py
    python build_kg_api_based.py --skip-api  # Skip API calls, use only DrugBank
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import pandas as pd
import networkx as nx
import pickle
import json
import logging
import os
import time
import requests
from pathlib import Path
import argparse

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
    source: str  # DrugBank, OpenFDA, PubChem
    match_type: str  # drugbank_id, drug_name, etc.
    match_value: str
    confidence: str = "exact"


@dataclass
class DrugNode:
    """Drug with all identifiers for matching."""
    drugbank_id: str
    name: str
    name_lower: str = ""
    synonyms: List[str] = field(default_factory=list)
    cas_number: str = ""
    unii: str = ""
    atc_codes: List[str] = field(default_factory=list)
    type: str = ""
    description: str = ""
    indication: str = ""
    mechanism_of_action: str = ""
    pharmacodynamics: str = ""
    metabolism: str = ""
    toxicity: str = ""
    half_life: str = ""
    protein_binding: str = ""
    pubchem_cid: str = ""
    chembl_id: str = ""
    kegg_id: str = ""
    smiles: str = ""
    inchi: str = ""
    inchi_key: str = ""
    molecular_weight: str = ""
    molecular_formula: str = ""
    groups: List[str] = field(default_factory=list)
    food_interactions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.name_lower = self.name.lower().strip()


@dataclass
class ProteinNode:
    """Protein target/enzyme/carrier/transporter from DrugBank."""
    id: str
    name: str
    type: str
    uniprot_id: str = ""
    gene_name: str = ""
    organism: str = ""
    general_function: str = ""
    specific_function: str = ""
    actions: List[str] = field(default_factory=list)
    known_action: str = ""
    provenance: Optional[Provenance] = None


@dataclass
class SideEffectNode:
    """Side effect from OpenFDA."""
    id: str
    name: str
    source: str = "OpenFDA"
    provenance: Optional[Provenance] = None


@dataclass 
class DiseaseNode:
    """Disease from PubChem."""
    id: str
    name: str
    provenance: Optional[Provenance] = None


@dataclass
class PathwayNode:
    """SMPDB pathway from DrugBank."""
    smpdb_id: str
    name: str
    category: str = ""
    provenance: Optional[Provenance] = None


@dataclass
class CategoryNode:
    """Category from DrugBank."""
    name: str
    mesh_id: str = ""
    provenance: Optional[Provenance] = None


# ---
# EDGE TYPES
# ---

@dataclass
class DDIEdge:
    """Drug-drug interaction."""
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
    type: str
    actions: List[str] = field(default_factory=list)
    known_action: str = ""
    provenance: Optional[Provenance] = None


@dataclass
class DrugSideEffectEdge:
    """Drug-side effect from OpenFDA."""
    drug_id: str
    side_effect_id: str
    count: int = 0
    is_serious: bool = False
    provenance: Optional[Provenance] = None


@dataclass
class DrugDiseaseEdge:
    """Drug-disease from PubChem."""
    drug_id: str
    disease_id: str
    relationship_type: str = ""
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


# ---
# API CLIENTS
# ---

class OpenFDAClient:
    """Client for OpenFDA drug adverse events API."""
    
    BASE_URL = "https://api.fda.gov/drug/event.json"
    RATE_LIMIT_DELAY = 0.3  # Conservative rate limiting
    
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
        self.cache = {}
        
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_adverse_events(self, drug_name: str, limit: int = 20) -> List[Dict]:
        """Get adverse events for a drug from OpenFDA."""
        # Check cache
        cache_key = f"fda:{drug_name.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit()
        
        try:
            # Search for drug in patient.drug.medicinalproduct field
            search = f'patient.drug.medicinalproduct:"{drug_name}"'
            params = {
                "search": search,
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": limit
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                self.cache[cache_key] = results
                return results
            else:
                return []
                
        except Exception as e:
            logger.debug(f"OpenFDA error for {drug_name}: {e}")
            return []
    
    def get_serious_events(self, drug_name: str, limit: int = 10) -> List[Dict]:
        """Get serious adverse events for a drug."""
        cache_key = f"fda_serious:{drug_name.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit()
        
        try:
            search = f'patient.drug.medicinalproduct:"{drug_name}"+AND+serious:1'
            params = {
                "search": search,
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": limit
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                self.cache[cache_key] = results
                return results
            else:
                return []
                
        except Exception as e:
            logger.debug(f"OpenFDA serious events error for {drug_name}: {e}")
            return []


class PubChemClient:
    """Client for PubChem PUG REST API for drug-disease associations."""
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    RATE_LIMIT_DELAY = 0.35  # PubChem requests 5 requests/second max
    
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
        self.cache = {}
        
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_cid_by_name(self, drug_name: str) -> Optional[str]:
        """Get PubChem Compound ID by drug name."""
        cache_key = f"cid:{drug_name.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/compound/name/{requests.utils.quote(drug_name)}/cids/JSON"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                cids = data.get('IdentifierList', {}).get('CID', [])
                if cids:
                    cid = str(cids[0])
                    self.cache[cache_key] = cid
                    return cid
            return None
            
        except Exception as e:
            logger.debug(f"PubChem CID error for {drug_name}: {e}")
            return None
    
    def get_associated_diseases(self, cid: str) -> List[Dict]:
        """Get diseases associated with a compound via PUG View."""
        cache_key = f"diseases:{cid}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/compound/cid/{cid}/xrefs/SourceName,SourceID/JSON"
            response = requests.get(url, timeout=30)
            
            diseases = []
            if response.status_code == 200:
                # PubChem doesn't have direct disease associations in bulk
                # We'll use DrugBank indications instead
                pass
            
            self.cache[cache_key] = diseases
            return diseases
            
        except Exception as e:
            logger.debug(f"PubChem disease error for CID {cid}: {e}")
            return []


# ---
# API-BASED KNOWLEDGE GRAPH BUILDER
# ---

class APIBasedKGBuilder:
    """
    Builds knowledge graph using DrugBank XML + public APIs.
    No local SIDER/CTD files required.
    """
    
    def __init__(self, xml_path: str, csv_path: str = None):
        self.xml_path = xml_path
        self.csv_path = csv_path
        
        # API clients
        self.fda_client = OpenFDAClient()
        self.pubchem_client = PubChemClient()
        
        # Drug lookups
        self.csv_drug_ids: Set[str] = set()
        self.csv_drug_names: Dict[str, str] = {}
        
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
        
        # Additional lookups
        self.drug_by_name: Dict[str, str] = {}
        self.drug_by_cas: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            'api_calls': defaultdict(int),
            'sources': defaultdict(int),
        }
    
    def _get_text(self, elem, tag: str) -> str:
        """Safely get text from XML element."""
        child = elem.find(f'db:{tag}', NS)
        if child is not None and child.text:
            return child.text.strip()
        return ""
    
    def _get_all_text(self, elem, tag: str) -> List[str]:
        """Get text from all matching child elements."""
        return [c.text.strip() for c in elem.findall(f'db:{tag}', NS) if c.text]
    
    def load_csv_identifiers(self) -> None:
        """Load drug identifiers from CSV if provided."""
        if not self.csv_path or not os.path.exists(self.csv_path):
            logger.info("No CSV file provided, will extract all drugs from DrugBank XML")
            return
            
        logger.info(f"Loading identifiers from {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        for col_suffix in ['1', '2']:
            id_col = f'drugbank_id_{col_suffix}'
            name_col = f'drug_name_{col_suffix}'
            
            if id_col not in df.columns:
                continue
                
            for _, row in df.iterrows():
                drug_id = row[id_col]
                drug_name = str(row[name_col]).strip()
                
                self.csv_drug_ids.add(drug_id)
                self.csv_drug_names[drug_name.lower()] = drug_id
        
        # Load DDI edges from CSV
        severity_col = 'severity_label'
        if 'severity_calibrated' in df.columns:
            severity_col = 'severity_calibrated'
        elif 'severity_recalibrated' in df.columns:
            severity_col = 'severity_recalibrated'
        
        for _, row in df.iterrows():
            severity = row.get(severity_col, row.get('severity_label', ''))
            
            self.ddi_edges.append(DDIEdge(
                drug1_id=row['drugbank_id_1'],
                drug2_id=row['drugbank_id_2'],
                description=row.get('interaction_description', ''),
                severity=severity,
                provenance=Provenance(
                    source="CSV",
                    match_type="drugbank_id",
                    match_value=f"{row['drugbank_id_1']}-{row['drugbank_id_2']}",
                ),
            ))
        
        logger.info(f"  Found {len(self.csv_drug_ids)} unique DrugBank IDs")
        logger.info(f"  Found {len(self.ddi_edges)} DDI edges from CSV")
    
    def parse_drugbank_xml(self) -> None:
        """Parse DrugBank XML and extract drug data."""
        logger.info(f"Parsing DrugBank XML: {self.xml_path}")
        
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        drug_count = 0
        matched_count = 0
        
        for drug_elem in root.findall('db:drug', NS):
            drugbank_id = ""
            for id_elem in drug_elem.findall('db:drugbank-id', NS):
                if id_elem.get('primary') == 'true':
                    drugbank_id = id_elem.text.strip() if id_elem.text else ""
                    break
            
            if not drugbank_id:
                continue
            
            drug_count += 1
            drug_name = self._get_text(drug_elem, 'name')
            
            # If CSV provided, filter to only CSV drugs; otherwise include all
            if self.csv_drug_ids:
                is_match = drugbank_id in self.csv_drug_ids or drug_name.lower() in self.csv_drug_names
                if not is_match:
                    continue
            
            matched_count += 1
            
            # Get synonyms
            synonyms = []
            syns_elem = drug_elem.find('db:synonyms', NS)
            if syns_elem is not None:
                for syn in syns_elem.findall('db:synonym', NS):
                    if syn.text:
                        synonyms.append(syn.text.strip())
            
            # Get CAS number
            cas_number = ""
            for id_elem in drug_elem.findall('.//db:identifier', NS):
                source = id_elem.get('source', '')
                if source == 'CAS' and id_elem.text:
                    cas_number = id_elem.text.strip()
                    break
            
            # Get ATC codes
            atc_codes = []
            atc_elem = drug_elem.find('db:atc-codes', NS)
            if atc_elem is not None:
                for level in atc_elem.findall('.//db:level', NS):
                    code = level.get('code')
                    if code and len(code) == 7:
                        atc_codes.append(code)
            
            # Get external IDs
            pubchem_cid = ""
            chembl_id = ""
            kegg_id = ""
            unii = ""
            
            ext_ids_elem = drug_elem.find('db:external-identifiers', NS)
            if ext_ids_elem is not None:
                for ext_id in ext_ids_elem.findall('db:external-identifier', NS):
                    resource = self._get_text(ext_id, 'resource')
                    identifier = self._get_text(ext_id, 'identifier')
                    if resource == 'PubChem Compound':
                        pubchem_cid = identifier
                    elif resource == 'ChEMBL':
                        chembl_id = identifier
                    elif resource == 'KEGG Drug':
                        kegg_id = identifier
                    elif resource == 'UNII':
                        unii = identifier
            
            # Get groups
            groups = self._get_all_text(drug_elem.find('db:groups', NS) or drug_elem, 'group')
            
            # Create drug node
            drug = DrugNode(
                drugbank_id=drugbank_id,
                name=drug_name,
                synonyms=synonyms,
                cas_number=cas_number,
                unii=unii,
                atc_codes=atc_codes,
                type=drug_elem.get('type', ''),
                description=self._get_text(drug_elem, 'description')[:500],
                indication=self._get_text(drug_elem, 'indication'),
                mechanism_of_action=self._get_text(drug_elem, 'mechanism-of-action'),
                pharmacodynamics=self._get_text(drug_elem, 'pharmacodynamics'),
                metabolism=self._get_text(drug_elem, 'metabolism'),
                toxicity=self._get_text(drug_elem, 'toxicity'),
                half_life=self._get_text(drug_elem, 'half-life'),
                pubchem_cid=pubchem_cid,
                chembl_id=chembl_id,
                kegg_id=kegg_id,
                groups=groups,
            )
            
            self.drugs[drugbank_id] = drug
            self.drug_by_name[drug_name.lower()] = drugbank_id
            if cas_number:
                self.drug_by_cas[cas_number] = drugbank_id
            for syn in synonyms:
                self.drug_by_name[syn.lower()] = drugbank_id
            
            self.stats['sources']['DrugBank-drug'] += 1
            
            # Parse protein relationships
            self._parse_proteins(drug_elem, drugbank_id)
            
            # Parse pathways
            self._parse_pathways(drug_elem, drugbank_id)
            
            # Parse categories
            self._parse_categories(drug_elem, drugbank_id)
            
            # Parse DrugBank DDIs (in addition to CSV)
            self._parse_drugbank_ddis(drug_elem, drugbank_id)
        
        logger.info(f"  Parsed {drug_count} total drugs, matched {matched_count}")
        logger.info(f"  Built {len(self.proteins)} protein nodes")
        logger.info(f"  Built {len(self.pathways)} pathway nodes")
    
    def _parse_proteins(self, drug_elem, drug_id: str) -> None:
        """Parse protein targets, enzymes, carriers, transporters."""
        for protein_type in ['targets', 'enzymes', 'carriers', 'transporters']:
            container = drug_elem.find(f'db:{protein_type}', NS)
            if container is None:
                continue
            
            singular = protein_type[:-1]  # Remove 's'
            for protein_elem in container.findall(f'db:{singular}', NS):
                protein_id = self._get_text(protein_elem, 'id')
                if not protein_id:
                    continue
                
                polypeptide = protein_elem.find('.//db:polypeptide', NS)
                
                uniprot_id = ""
                gene_name = ""
                protein_name = self._get_text(protein_elem, 'name')
                
                if polypeptide is not None:
                    uniprot_id = polypeptide.get('id', '')
                    gene_name = self._get_text(polypeptide, 'gene-name')
                    if not protein_name:
                        protein_name = self._get_text(polypeptide, 'name')
                
                if not protein_name:
                    continue
                
                # Get actions
                actions = self._get_all_text(protein_elem.find('db:actions', NS) or protein_elem, 'action')
                known_action = self._get_text(protein_elem, 'known-action')
                
                # Create protein node if not exists
                if protein_id not in self.proteins:
                    self.proteins[protein_id] = ProteinNode(
                        id=protein_id,
                        name=protein_name,
                        type=singular,
                        uniprot_id=uniprot_id,
                        gene_name=gene_name,
                        actions=actions,
                        known_action=known_action,
                        provenance=Provenance(
                            source="DrugBank",
                            match_type="drugbank_id",
                            match_value=protein_id,
                        ),
                    )
                    self.stats['sources']['DrugBank-protein'] += 1
                
                # Create drug-protein edge
                self.drug_protein_edges.append(DrugProteinEdge(
                    drug_id=drug_id,
                    protein_id=protein_id,
                    type=singular,
                    actions=actions,
                    known_action=known_action,
                    provenance=Provenance(
                        source="DrugBank",
                        match_type="drugbank_id",
                        match_value=f"{drug_id}-{protein_id}",
                    ),
                ))
    
    def _parse_pathways(self, drug_elem, drug_id: str) -> None:
        """Parse SMPDB pathways from DrugBank."""
        pathways_elem = drug_elem.find('db:pathways', NS)
        if pathways_elem is None:
            return
        
        for pathway_elem in pathways_elem.findall('db:pathway', NS):
            smpdb_id = self._get_text(pathway_elem, 'smpdb-id')
            pathway_name = self._get_text(pathway_elem, 'name')
            category = self._get_text(pathway_elem, 'category')
            
            if not smpdb_id or not pathway_name:
                continue
            
            if smpdb_id not in self.pathways:
                self.pathways[smpdb_id] = PathwayNode(
                    smpdb_id=smpdb_id,
                    name=pathway_name,
                    category=category,
                    provenance=Provenance(
                        source="DrugBank",
                        match_type="smpdb_id",
                        match_value=smpdb_id,
                    ),
                )
                self.stats['sources']['DrugBank-pathway'] += 1
            
            self.drug_pathway_edges.append(DrugPathwayEdge(
                drug_id=drug_id,
                pathway_id=smpdb_id,
                provenance=Provenance(
                    source="DrugBank",
                    match_type="drugbank_id",
                    match_value=f"{drug_id}-{smpdb_id}",
                ),
            ))
    
    def _parse_categories(self, drug_elem, drug_id: str) -> None:
        """Parse drug categories from DrugBank."""
        categories_elem = drug_elem.find('db:categories', NS)
        if categories_elem is None:
            return
        
        for cat_elem in categories_elem.findall('db:category', NS):
            cat_name = self._get_text(cat_elem, 'category')
            mesh_id = self._get_text(cat_elem, 'mesh-id')
            
            if not cat_name:
                continue
            
            if cat_name not in self.categories:
                self.categories[cat_name] = CategoryNode(
                    name=cat_name,
                    mesh_id=mesh_id,
                    provenance=Provenance(
                        source="DrugBank",
                        match_type="category_name",
                        match_value=cat_name,
                    ),
                )
            
            self.drug_category_edges.append(DrugCategoryEdge(
                drug_id=drug_id,
                category_name=cat_name,
                provenance=Provenance(
                    source="DrugBank",
                    match_type="drugbank_id",
                    match_value=f"{drug_id}-{cat_name}",
                ),
            ))
    
    def _parse_drugbank_ddis(self, drug_elem, drug_id: str) -> None:
        """Parse DDIs directly from DrugBank XML."""
        ddis_elem = drug_elem.find('db:drug-interactions', NS)
        if ddis_elem is None:
            return
        
        # Use instance variable to avoid rebuilding set each time
        if not hasattr(self, '_existing_ddi_pairs'):
            self._existing_ddi_pairs = {(e.drug1_id, e.drug2_id) for e in self.ddi_edges}
            self._existing_ddi_pairs.update({(e.drug2_id, e.drug1_id) for e in self.ddi_edges})
        
        for ddi_elem in ddis_elem.findall('db:drug-interaction', NS):
            other_id = self._get_text(ddi_elem, 'drugbank-id')
            description = self._get_text(ddi_elem, 'description')
            
            if not other_id:
                continue
            
            # Skip if already in CSV or already added (avoid duplicates)
            pair = tuple(sorted([drug_id, other_id]))
            if pair in self._existing_ddi_pairs or (pair[1], pair[0]) in self._existing_ddi_pairs:
                continue
            
            # Only add if other drug is in our set (or we're not filtering)
            if self.csv_drug_ids and other_id not in self.csv_drug_ids:
                continue
            
            self._existing_ddi_pairs.add(pair)
            self.ddi_edges.append(DDIEdge(
                drug1_id=pair[0],
                drug2_id=pair[1],
                description=description,
                severity="",  # DrugBank doesn't have severity labels
                provenance=Provenance(
                    source="DrugBank-XML",
                    match_type="drugbank_id",
                    match_value=f"{drug_id}-{other_id}",
                ),
            ))
    
    def fetch_openfda_side_effects(self, max_drugs: int = None, batch_size: int = 50) -> None:
        """Fetch side effects from OpenFDA API for all drugs."""
        logger.info("\n--- Fetching Side Effects from OpenFDA API ---")
        
        drug_list = list(self.drugs.values())
        if max_drugs:
            drug_list = drug_list[:max_drugs]
        
        logger.info(f"  Fetching adverse events for {len(drug_list)} drugs...")
        
        for i, drug in enumerate(drug_list):
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{len(drug_list)} drugs queried")
            
            # Fetch adverse events
            events = self.fda_client.get_adverse_events(drug.name, limit=15)
            self.stats['api_calls']['openfda_adverse_events'] += 1
            
            for event in events:
                term = event.get('term', '')
                count = event.get('count', 0)
                
                if not term or count < 10:  # Skip rare events
                    continue
                
                # Create side effect node
                se_id = f"FDA:{term.replace(' ', '_').upper()}"
                
                if se_id not in self.side_effects:
                    self.side_effects[se_id] = SideEffectNode(
                        id=se_id,
                        name=term,
                        source="OpenFDA",
                        provenance=Provenance(
                            source="OpenFDA",
                            match_type="reaction_term",
                            match_value=term,
                        ),
                    )
                    self.stats['sources']['OpenFDA-side_effect'] += 1
                
                # Create edge
                self.drug_se_edges.append(DrugSideEffectEdge(
                    drug_id=drug.drugbank_id,
                    side_effect_id=se_id,
                    count=count,
                    is_serious=False,
                    provenance=Provenance(
                        source="OpenFDA",
                        match_type="drug_name",
                        match_value=drug.name,
                    ),
                ))
            
            # Also fetch serious events
            serious_events = self.fda_client.get_serious_events(drug.name, limit=10)
            self.stats['api_calls']['openfda_serious_events'] += 1
            
            for event in serious_events:
                term = event.get('term', '')
                count = event.get('count', 0)
                
                if not term:
                    continue
                
                se_id = f"FDA:{term.replace(' ', '_').upper()}"
                
                if se_id not in self.side_effects:
                    self.side_effects[se_id] = SideEffectNode(
                        id=se_id,
                        name=term,
                        source="OpenFDA",
                        provenance=Provenance(
                            source="OpenFDA",
                            match_type="reaction_term",
                            match_value=term,
                        ),
                    )
                    self.stats['sources']['OpenFDA-side_effect'] += 1
                
                # Check if edge already exists
                edge_exists = any(
                    e.drug_id == drug.drugbank_id and e.side_effect_id == se_id
                    for e in self.drug_se_edges
                )
                
                if not edge_exists:
                    self.drug_se_edges.append(DrugSideEffectEdge(
                        drug_id=drug.drugbank_id,
                        side_effect_id=se_id,
                        count=count,
                        is_serious=True,
                        provenance=Provenance(
                            source="OpenFDA",
                            match_type="drug_name_serious",
                            match_value=drug.name,
                        ),
                    ))
        
        logger.info(f"  Loaded {len(self.side_effects)} unique side effects from OpenFDA")
        logger.info(f"  Loaded {len(self.drug_se_edges)} drug-side effect associations")
    
    def extract_diseases_from_indications(self) -> None:
        """Extract disease associations from DrugBank indication text."""
        logger.info("\n--- Extracting Diseases from DrugBank Indications ---")
        
        # Common disease patterns (simplified extraction)
        disease_count = 0
        
        for drug_id, drug in self.drugs.items():
            if not drug.indication:
                continue
            
            # Create a disease node from the indication
            # This is a simplified approach - in production, use NER
            indication_preview = drug.indication[:100].strip()
            if indication_preview:
                disease_id = f"IND:{drug_id}"
                
                if disease_id not in self.diseases:
                    self.diseases[disease_id] = DiseaseNode(
                        id=disease_id,
                        name=indication_preview,
                        provenance=Provenance(
                            source="DrugBank",
                            match_type="indication",
                            match_value=drug_id,
                        ),
                    )
                    disease_count += 1
                
                self.drug_disease_edges.append(DrugDiseaseEdge(
                    drug_id=drug_id,
                    disease_id=disease_id,
                    relationship_type="indicated_for",
                    provenance=Provenance(
                        source="DrugBank",
                        match_type="indication",
                        match_value=drug_id,
                    ),
                ))
        
        logger.info(f"  Extracted {disease_count} disease associations from indications")
    
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
            },
            'edges': {
                'ddi': len(self.ddi_edges),
                'drug_protein': len(self.drug_protein_edges),
                'drug_side_effect': len(self.drug_se_edges),
                'drug_disease': len(self.drug_disease_edges),
                'drug_pathway': len(self.drug_pathway_edges),
                'drug_category': len(self.drug_category_edges),
            },
            'api_calls': dict(self.stats['api_calls']),
            'sources': dict(self.stats['sources']),
        }
    
    def build_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph."""
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
            )
        
        # Add side effect nodes
        for se_id, se in self.side_effects.items():
            G.add_node(
                se_id,
                node_type='side_effect',
                name=se.name,
                source=se.source,
            )
        
        # Add disease nodes
        for disease_id, disease in self.diseases.items():
            G.add_node(
                disease_id,
                node_type='disease',
                name=disease.name,
            )
        
        # Add pathway nodes
        for pathway_id, pathway in self.pathways.items():
            G.add_node(
                pathway_id,
                node_type='pathway',
                name=pathway.name,
                category=pathway.category,
            )
        
        # Add DDI edges
        for edge in self.ddi_edges:
            G.add_edge(
                edge.drug1_id, edge.drug2_id,
                edge_type='interacts_with',
                description=edge.description[:200] if edge.description else "",
                severity=edge.severity,
                source=edge.provenance.source if edge.provenance else "",
            )
        
        # Add drug-protein edges
        for edge in self.drug_protein_edges:
            G.add_edge(
                edge.drug_id, edge.protein_id,
                edge_type=f'has_{edge.type}',
                actions=edge.actions,
            )
        
        # Add drug-side effect edges
        for edge in self.drug_se_edges:
            G.add_edge(
                edge.drug_id, edge.side_effect_id,
                edge_type='has_side_effect',
                count=edge.count,
                is_serious=edge.is_serious,
            )
        
        # Add drug-disease edges
        for edge in self.drug_disease_edges:
            G.add_edge(
                edge.drug_id, edge.disease_id,
                edge_type='treats',
                relationship=edge.relationship_type,
            )
        
        # Add drug-pathway edges
        for edge in self.drug_pathway_edges:
            G.add_edge(
                edge.drug_id, edge.pathway_id,
                edge_type='involved_in_pathway',
            )
        
        logger.info(f"  Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def export(self, output_dir: str = "knowledge_graph_api_based") -> None:
        """Export knowledge graph to files."""
        logger.info(f"\n--- Exporting to {output_dir}/ ---")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        neo4j_path = output_path / "neo4j_export"
        neo4j_path.mkdir(exist_ok=True)
        
        # Export drugs
        drug_data = []
        for drug_id, drug in self.drugs.items():
            drug_data.append({
                'drugbank_id': drug_id,
                'name': drug.name,
                'type': drug.type,
                'cas_number': drug.cas_number,
                'atc_codes': ';'.join(drug.atc_codes),
                'indication': drug.indication[:500] if drug.indication else "",
                'mechanism': drug.mechanism_of_action[:500] if drug.mechanism_of_action else "",
                'pubchem_cid': drug.pubchem_cid,
            })
        pd.DataFrame(drug_data).to_csv(neo4j_path / 'drugs.csv', index=False)
        
        # Export proteins
        protein_data = []
        for protein_id, protein in self.proteins.items():
            protein_data.append({
                'protein_id': protein_id,
                'name': protein.name,
                'type': protein.type,
                'uniprot_id': protein.uniprot_id,
                'gene_name': protein.gene_name,
            })
        pd.DataFrame(protein_data).to_csv(neo4j_path / 'proteins.csv', index=False)
        
        # Export side effects
        se_data = []
        for se_id, se in self.side_effects.items():
            se_data.append({
                'side_effect_id': se_id,
                'name': se.name,
                'source': se.source,
            })
        pd.DataFrame(se_data).to_csv(neo4j_path / 'side_effects.csv', index=False)
        
        # Export diseases
        disease_data = []
        for disease_id, disease in self.diseases.items():
            disease_data.append({
                'disease_id': disease_id,
                'name': disease.name,
            })
        pd.DataFrame(disease_data).to_csv(neo4j_path / 'diseases.csv', index=False)
        
        # Export pathways
        pathway_data = []
        for pathway_id, pathway in self.pathways.items():
            pathway_data.append({
                'pathway_id': pathway_id,
                'name': pathway.name,
                'category': pathway.category,
            })
        pd.DataFrame(pathway_data).to_csv(neo4j_path / 'pathways.csv', index=False)
        
        # Export categories
        cat_data = []
        for cat_name, cat in self.categories.items():
            cat_data.append({
                'category_name': cat_name,
                'mesh_id': cat.mesh_id,
            })
        pd.DataFrame(cat_data).to_csv(neo4j_path / 'categories.csv', index=False)
        
        # Export DDI edges
        ddi_data = []
        for edge in self.ddi_edges:
            ddi_data.append({
                'drug1_id': edge.drug1_id,
                'drug2_id': edge.drug2_id,
                'description': edge.description[:300] if edge.description else "",
                'severity': edge.severity,
                'source': edge.provenance.source if edge.provenance else "",
            })
        pd.DataFrame(ddi_data).to_csv(neo4j_path / 'ddi_edges.csv', index=False)
        
        # Export drug-protein edges
        dp_data = []
        for edge in self.drug_protein_edges:
            dp_data.append({
                'drug_id': edge.drug_id,
                'protein_id': edge.protein_id,
                'type': edge.type,
                'actions': ';'.join(edge.actions),
            })
        pd.DataFrame(dp_data).to_csv(neo4j_path / 'drug_protein_edges.csv', index=False)
        
        # Export drug-side effect edges
        dse_data = []
        for edge in self.drug_se_edges:
            dse_data.append({
                'drug_id': edge.drug_id,
                'side_effect_id': edge.side_effect_id,
                'count': edge.count,
                'is_serious': edge.is_serious,
                'source': 'OpenFDA',
            })
        pd.DataFrame(dse_data).to_csv(neo4j_path / 'drug_side_effect_edges.csv', index=False)
        
        # Export drug-disease edges
        dd_data = []
        for edge in self.drug_disease_edges:
            dd_data.append({
                'drug_id': edge.drug_id,
                'disease_id': edge.disease_id,
                'relationship_type': edge.relationship_type,
            })
        pd.DataFrame(dd_data).to_csv(neo4j_path / 'drug_disease_edges.csv', index=False)
        
        # Export drug-pathway edges
        dpw_data = []
        for edge in self.drug_pathway_edges:
            dpw_data.append({
                'drug_id': edge.drug_id,
                'pathway_id': edge.pathway_id,
            })
        pd.DataFrame(dpw_data).to_csv(neo4j_path / 'drug_pathway_edges.csv', index=False)
        
        # Export drug-category edges
        dc_data = []
        for edge in self.drug_category_edges:
            dc_data.append({
                'drug_id': edge.drug_id,
                'category_name': edge.category_name,
            })
        pd.DataFrame(dc_data).to_csv(neo4j_path / 'drug_category_edges.csv', index=False)
        
        # Save statistics
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
                'graph': G,
                'statistics': stats,
            }, f)
        
        logger.info("  Export complete!")


def main():
    """Build API-based DDI Knowledge Graph."""
    parser = argparse.ArgumentParser(description='Build Knowledge Graph using DrugBank + APIs')
    parser.add_argument('--skip-api', action='store_true', 
                        help='Skip API calls, use only DrugBank data')
    parser.add_argument('--max-drugs', type=int, default=None,
                        help='Limit API calls to first N drugs (for testing)')
    parser.add_argument('--output', type=str, default='knowledge_graph_api_based',
                        help='Output directory')
    args = parser.parse_args()
    
    # File paths
    xml_path = "data/full database.xml"
    csv_path = "data/ddi_cardio_or_antithrombotic_labeled (1).csv"
    
    # Check for recalibrated CSV first
    if os.path.exists("data/ddi_recalibrated.csv"):
        csv_path = "data/ddi_recalibrated.csv"
    
    # Verify XML exists
    if not os.path.exists(xml_path):
        logger.error(f"DrugBank XML not found: {xml_path}")
        logger.error("Please download DrugBank data. See DATA_SOURCES.md for instructions.")
        return 1
    
    # Initialize builder
    builder = APIBasedKGBuilder(xml_path, csv_path if os.path.exists(csv_path) else None)
    
    print("\n" + "=" * 70)
    print("API-BASED KNOWLEDGE GRAPH BUILDER")
    print("=" * 70)
    print("\nData Sources:")
    print("  - DrugBank XML: Local file (licensed)")
    print("  - Side Effects: OpenFDA API (public)")
    print("  - Diseases: DrugBank indications (no CTD required)")
    print("=" * 70)
    
    # Step 1: Load CSV identifiers (if available)
    if os.path.exists(csv_path):
        builder.load_csv_identifiers()
    
    # Step 2: Parse DrugBank XML
    builder.parse_drugbank_xml()
    
    # Step 3: Fetch side effects from OpenFDA (unless skipped)
    if not args.skip_api:
        builder.fetch_openfda_side_effects(max_drugs=args.max_drugs)
    else:
        logger.info("\nSkipping API calls (--skip-api flag set)")
    
    # Step 4: Extract diseases from indications
    builder.extract_diseases_from_indications()
    
    # Get statistics
    stats = builder.get_statistics()
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("=" * 70)
    
    print("\nNode Counts:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    
    print("\nEdge Counts:")
    for edge_type, count in stats['edges'].items():
        print(f"  {edge_type}: {count:,}")
    
    if stats['api_calls']:
        print("\nAPI Calls Made:")
        for api, count in stats['api_calls'].items():
            print(f"  {api}: {count:,}")
    
    print("\nData Sources:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count:,}")
    
    # Export
    builder.export(args.output)
    
    print(f"\nAll outputs saved to {args.output}/")
    print("\nThis knowledge graph was built using:")
    print("  - DrugBank (local XML, requires license)")
    print("  - OpenFDA API (public, no license required)")
    print("\nNo SIDER or CTD local files required!")
    
    return 0


if __name__ == '__main__':
    exit(main())
