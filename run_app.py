#!/usr/bin/env python3
"""
DDI Risk Analysis Application Launcher
Filtered to Cardiovascular and Antithrombotic Drugs

This script:
1. Prompts user for DrugBank XML path (required - licensed data)
2. Filters to cardiovascular (ATC C*) and antithrombotic (ATC B01*) drugs
3. Uses existing shareable reference data (FAERS, high-risk drug classes)
4. Builds knowledge graph and launches the application

Filtering logic (from drugbankDataset_wrangling notebook):
- Cardiovascular: ATC code starts with "C"
- Antithrombotic: ATC code starts with "B01"
- DDI included if at least one drug is cardiovascular OR antithrombotic

Shareable data (included in repo):
- external_data/faers_comprehensive_reports.json (Public Domain - FDA)
- external_data/high_risk_drug_classes_reference.json (Wikipedia sources)

User must provide:
- DrugBank full database XML (requires academic/commercial license)
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Drug:
    drugbank_id: str
    name: str
    description: str = ""
    cas_number: str = ""
    categories: List[str] = field(default_factory=list)
    atc_codes: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    enzymes: List[str] = field(default_factory=list)
    carriers: List[str] = field(default_factory=list)
    transporters: List[str] = field(default_factory=list)
    indication: str = ""
    mechanism: str = ""
    half_life: str = ""
    synonyms: List[str] = field(default_factory=list)
    is_cardiovascular: bool = False
    is_antithrombotic: bool = False


@dataclass 
class DDI:
    drug1_id: str
    drug2_id: str
    drug1_name: str
    drug2_name: str
    description: str
    severity: str = "Unknown"


@dataclass
class Protein:
    uniprot_id: str
    name: str
    gene_name: str = ""
    organism: str = ""


# =============================================================================
# Knowledge Graph Builder - Cardiovascular & Antithrombotic Filter
# =============================================================================

class CardioKnowledgeGraphBuilder:
    """
    Builds KG from DrugBank XML filtered to cardiovascular and antithrombotic drugs.
    
    Filtering logic (matching the notebook):
    - Cardiovascular: ATC code starts with "C"
    - Antithrombotic: ATC code starts with "B01"
    - DDI included if at least one drug is cardiovascular OR antithrombotic
    """
    
    NAMESPACE = {'db': 'http://www.drugbank.ca'}
    
    def __init__(self, drugbank_xml_path: str):
        self.xml_path = drugbank_xml_path
        
        # ATC lookup: drug_id -> list of ATC codes
        self.atc_lookup: Dict[str, List[str]] = {}
        
        # Filtered data
        self.drugs: Dict[str, Drug] = {}
        self.ddis: List[DDI] = []
        self.proteins: Dict[str, Protein] = {}
        self.categories: Set[str] = set()
        
        # Stats
        self.stats = {
            'total_drugs': 0,
            'cardiovascular_drugs': 0,
            'antithrombotic_drugs': 0,
            'total_ddis': 0,
            'filtered_ddis': 0,
        }
        
        # Reference data paths (shareable - included in repo)
        self.faers_path = "external_data/faers_comprehensive_reports.json"
        self.high_risk_path = "external_data/high_risk_drug_classes_reference.json"
        
        # Loaded reference data
        self.faers_data = {}
        self.high_risk_classes = {}
    
    def classify_atc(self, atc_codes: List[str]) -> Tuple[bool, bool]:
        """
        Classify drug based on ATC codes.
        Returns (is_cardiovascular, is_antithrombotic)
        """
        if not atc_codes:
            return False, False
        
        is_cardiovascular = any(code.startswith("C") for code in atc_codes)
        is_antithrombotic = any(code.startswith("B01") for code in atc_codes)
        
        return is_cardiovascular, is_antithrombotic
    
    def load_reference_data(self):
        """Load shareable reference data files"""
        print("\n📚 Loading reference data (shareable - no license conflict)...")
        
        # Load FAERS data
        if os.path.exists(self.faers_path):
            with open(self.faers_path, 'r') as f:
                faers_raw = json.load(f)
            self.faers_data = defaultdict(list)
            for record in faers_raw:
                drug = record.get('query_drug', '').lower()
                self.faers_data[drug].append(record)
            print(f"  ✓ FAERS adverse events: {len(faers_raw):,} records")
        else:
            print(f"  ⚠ FAERS data not found: {self.faers_path}")
            
        # Load high-risk drug classes
        if os.path.exists(self.high_risk_path):
            with open(self.high_risk_path, 'r') as f:
                self.high_risk_classes = json.load(f)
            n_drugs = sum(
                len(cat.get('drugs', []))
                for section in self.high_risk_classes.values()
                if isinstance(section, dict)
                for cat in section.get('categories', {}).values()
                if isinstance(cat, dict)
            )
            print(f"  ✓ High-risk drug classes: {n_drugs} drugs across categories")
        else:
            print(f"  ⚠ High-risk classes not found: {self.high_risk_path}")
    
    def build_atc_lookup(self):
        """
        First pass: Build ATC lookup dictionary for all drugs.
        This is needed to classify DDI partners.
        """
        print(f"\n📖 Building ATC lookup from: {self.xml_path}")
        
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"DrugBank XML not found: {self.xml_path}")
        
        ns = self.NAMESPACE
        drug_tag = f"{{{ns['db']}}}drug"
        
        # Use iterparse for memory efficiency
        # Get reference to root first for proper memory management
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        root = None
        drug_count = 0
        
        for event, elem in context:
            # Get root on first start event
            if event == 'start' and root is None:
                root = elem
                continue
            
            if event == 'end' and elem.tag == drug_tag:
                # Get drug ID
                drug_id = None
                for db_id in elem.findall('db:drugbank-id', ns):
                    if db_id.get('primary') == 'true':
                        drug_id = db_id.text
                        break
                if not drug_id:
                    ids = elem.findall('db:drugbank-id', ns)
                    if ids:
                        drug_id = ids[0].text
                
                if drug_id:
                    # Get ATC codes - extract BEFORE clearing
                    atc_codes = []
                    for atc in elem.findall('.//db:atc-code', ns):
                        code = atc.get('code')
                        if code:
                            atc_codes.append(code)
                    
                    # Only store if we found ATC codes, or if this is first time seeing this drug
                    # (prevents later empty entries from overwriting good data)
                    if atc_codes or drug_id not in self.atc_lookup:
                        # Merge with existing if we have both
                        if drug_id in self.atc_lookup and self.atc_lookup[drug_id]:
                            existing = set(self.atc_lookup[drug_id])
                            existing.update(atc_codes)
                            self.atc_lookup[drug_id] = list(existing)
                        else:
                            self.atc_lookup[drug_id] = atc_codes
                    
                    drug_count += 1
                    
                    if drug_count % 5000 == 0:
                        print(f"   Scanned {drug_count:,} drugs for ATC codes...")
                
                # Clear this element and remove from root to free memory
                elem.clear()
                if root is not None:
                    # Remove processed child from root
                    try:
                        root.remove(elem)
                    except ValueError:
                        pass
        
        self.stats['total_drugs'] = drug_count
        print(f"  ✓ ATC lookup built: {drug_count:,} drugs")
        
        # Count cardiovascular and antithrombotic
        for drug_id, atc_codes in self.atc_lookup.items():
            is_cardio, is_antithrombotic = self.classify_atc(atc_codes)
            if is_cardio:
                self.stats['cardiovascular_drugs'] += 1
            if is_antithrombotic:
                self.stats['antithrombotic_drugs'] += 1
        
        print(f"  ✓ Cardiovascular drugs (ATC C*): {self.stats['cardiovascular_drugs']:,}")
        print(f"  ✓ Antithrombotic drugs (ATC B01*): {self.stats['antithrombotic_drugs']:,}")
    
    def parse_filtered_drugs(self):
        """
        Second pass: Parse drugs and DDIs, filtering to cardio/antithrombotic.
        """
        print(f"\n🔍 Parsing and filtering to cardiovascular & antithrombotic drugs...")
        
        ns = self.NAMESPACE
        drug_tag = f"{{{ns['db']}}}drug"
        
        # Use iterparse with proper memory management
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        root = None
        
        drug_count = 0
        ddi_count = 0
        protein_count = 0
        filtered_drugs = 0
        filtered_ddis = 0
        
        for event, elem in context:
            # Get root on first start event
            if event == 'start' and root is None:
                root = elem
                continue
                
            if event == 'end' and elem.tag == drug_tag:
                drug = self._parse_drug_element(elem, ns)
                
                if drug:
                    drug_count += 1
                    
                    # Check if this drug is cardio or antithrombotic
                    is_cardio_1, is_anti_1 = self.classify_atc(drug.atc_codes)
                    drug.is_cardiovascular = is_cardio_1
                    drug.is_antithrombotic = is_anti_1
                    
                    # Process DDIs
                    for ddi_elem in elem.findall('.//db:drug-interaction', ns):
                        ddi = self._parse_ddi_element(ddi_elem, drug, ns)
                        if ddi:
                            ddi_count += 1
                            
                            # Check if other drug is cardio or antithrombotic
                            other_atc = self.atc_lookup.get(ddi.drug2_id, [])
                            is_cardio_2, is_anti_2 = self.classify_atc(other_atc)
                            
                            # Include DDI if at least one drug is cardio OR antithrombotic
                            if is_cardio_1 or is_anti_1 or is_cardio_2 or is_anti_2:
                                self.ddis.append(ddi)
                                filtered_ddis += 1
                                
                                # Add both drugs to our dataset
                                if drug.drugbank_id not in self.drugs:
                                    self.drugs[drug.drugbank_id] = drug
                                    filtered_drugs += 1
                    
                    # Extract proteins for filtered drugs
                    if drug.drugbank_id in self.drugs:
                        for target in elem.findall('.//db:target', ns):
                            protein = self._parse_protein_element(target, ns)
                            if protein and protein.uniprot_id not in self.proteins:
                                self.proteins[protein.uniprot_id] = protein
                                protein_count += 1
                        
                        for cat in drug.categories:
                            self.categories.add(cat)
                    
                    if drug_count % 2000 == 0:
                        print(f"   Processed {drug_count:,} drugs, kept {filtered_drugs:,} drugs, {filtered_ddis:,} DDIs...")
                
                # Clear this element and remove from root to free memory
                elem.clear()
                if root is not None:
                    try:
                        root.remove(elem)
                    except ValueError:
                        pass
        
        self.stats['total_ddis'] = ddi_count
        self.stats['filtered_ddis'] = filtered_ddis
        
        print(f"\n✅ Filtering complete:")
        print(f"   Total drugs scanned: {drug_count:,}")
        print(f"   Filtered drugs (cardio/antithrombotic): {len(self.drugs):,}")
        print(f"   Total DDIs: {ddi_count:,}")
        print(f"   Filtered DDIs: {filtered_ddis:,}")
        print(f"   Proteins: {len(self.proteins):,}")
        print(f"   Categories: {len(self.categories):,}")
    
    def _parse_drug_element(self, elem, ns) -> Optional[Drug]:
        """Parse a single drug element"""
        drugbank_id = None
        for db_id in elem.findall('db:drugbank-id', ns):
            if db_id.get('primary') == 'true':
                drugbank_id = db_id.text
                break
        if not drugbank_id:
            ids = elem.findall('db:drugbank-id', ns)
            if ids:
                drugbank_id = ids[0].text
        
        if not drugbank_id:
            return None
            
        name_elem = elem.find('db:name', ns)
        name = name_elem.text if name_elem is not None else ""
        
        desc_elem = elem.find('db:description', ns)
        description = desc_elem.text if desc_elem is not None and desc_elem.text else ""
        
        cas_elem = elem.find('db:cas-number', ns)
        cas_number = cas_elem.text if cas_elem is not None and cas_elem.text else ""
        
        # Categories
        categories = []
        for cat in elem.findall('.//db:category/db:category', ns):
            if cat.text:
                categories.append(cat.text)
        
        # ATC codes
        atc_codes = self.atc_lookup.get(drugbank_id, [])
        
        # Targets
        targets = []
        for target in elem.findall('.//db:target/db:polypeptide', ns):
            uniprot = target.find('db:external-identifiers/db:external-identifier[db:resource="UniProtKB"]/db:identifier', ns)
            if uniprot is not None and uniprot.text:
                targets.append(uniprot.text)
        
        # Enzymes
        enzymes = []
        for enzyme in elem.findall('.//db:enzyme/db:polypeptide', ns):
            uniprot = enzyme.find('db:external-identifiers/db:external-identifier[db:resource="UniProtKB"]/db:identifier', ns)
            if uniprot is not None and uniprot.text:
                enzymes.append(uniprot.text)
        
        # Indication
        ind_elem = elem.find('db:indication', ns)
        indication = ind_elem.text if ind_elem is not None and ind_elem.text else ""
        
        # Mechanism
        mech_elem = elem.find('db:mechanism-of-action', ns)
        mechanism = mech_elem.text if mech_elem is not None and mech_elem.text else ""
        
        # Half-life
        hl_elem = elem.find('db:half-life', ns)
        half_life = hl_elem.text if hl_elem is not None and hl_elem.text else ""
        
        # Synonyms
        synonyms = []
        for syn in elem.findall('.//db:synonym', ns):
            if syn.text:
                synonyms.append(syn.text)
        
        return Drug(
            drugbank_id=drugbank_id,
            name=name,
            description=description[:500] if description else "",
            cas_number=cas_number,
            categories=categories,
            atc_codes=atc_codes,
            targets=targets,
            enzymes=enzymes,
            indication=indication[:500] if indication else "",
            mechanism=mechanism[:500] if mechanism else "",
            half_life=half_life,
            synonyms=synonyms[:10]
        )
    
    def _parse_ddi_element(self, ddi_elem, drug: Drug, ns) -> Optional[DDI]:
        """Parse a DDI element"""
        other_id_elem = ddi_elem.find('db:drugbank-id', ns)
        other_name_elem = ddi_elem.find('db:name', ns)
        desc_elem = ddi_elem.find('db:description', ns)
        
        if other_id_elem is None or other_name_elem is None:
            return None
            
        other_id = other_id_elem.text
        other_name = other_name_elem.text
        description = desc_elem.text if desc_elem is not None and desc_elem.text else ""
        
        severity = self._classify_severity(description)
        
        return DDI(
            drug1_id=drug.drugbank_id,
            drug2_id=other_id,
            drug1_name=drug.name,
            drug2_name=other_name,
            description=description[:500] if description else "",
            severity=severity
        )
    
    def _classify_severity(self, description: str) -> str:
        """Classify DDI severity based on description keywords"""
        desc_lower = description.lower()
        
        # Major indicators
        major_keywords = [
            'contraindicated', 'avoid', 'do not use', 'life-threatening',
            'fatal', 'death', 'serious', 'severe', 'dangerous',
            'serotonin syndrome', 'qt prolongation', 'torsades',
            'bleeding', 'hemorrhage', 'cardiac arrest', 'seizure'
        ]
        
        # Moderate indicators
        moderate_keywords = [
            'monitor', 'caution', 'may increase', 'may decrease',
            'adjust dose', 'reduce dose', 'increased risk', 'enhanced effect',
            'toxicity', 'adverse effect'
        ]
        
        for kw in major_keywords:
            if kw in desc_lower:
                return 'Major'
        
        for kw in moderate_keywords:
            if kw in desc_lower:
                return 'Moderate'
        
        return 'Minor'
    
    def _parse_protein_element(self, elem, ns) -> Optional[Protein]:
        """Parse a protein/target element"""
        polypeptide = elem.find('db:polypeptide', ns)
        if polypeptide is None:
            return None
            
        uniprot_elem = polypeptide.find(
            'db:external-identifiers/db:external-identifier[db:resource="UniProtKB"]/db:identifier', ns
        )
        if uniprot_elem is None:
            return None
            
        uniprot_id = uniprot_elem.text
        
        name_elem = polypeptide.find('db:name', ns)
        name = name_elem.text if name_elem is not None else ""
        
        gene_elem = polypeptide.find('db:gene-name', ns)
        gene_name = gene_elem.text if gene_elem is not None and gene_elem.text else ""
        
        organism_elem = polypeptide.find('db:organism', ns)
        organism = organism_elem.text if organism_elem is not None else ""
        
        return Protein(
            uniprot_id=uniprot_id,
            name=name,
            gene_name=gene_name,
            organism=organism
        )
    
    def enrich_with_faers(self):
        """Enrich drugs with FAERS adverse event data"""
        if not self.faers_data:
            return
            
        print("\n🔬 Enriching with FAERS adverse event data...")
        matched = 0
        for drug in self.drugs.values():
            drug_name_lower = drug.name.lower()
            if drug_name_lower in self.faers_data:
                matched += 1
        
        print(f"  ✓ Matched {matched} drugs with FAERS data")
    
    def export_for_app(self, output_dir: str):
        """Export knowledge graph for the DDI app"""
        os.makedirs(output_dir, exist_ok=True)
        neo4j_dir = os.path.join(output_dir, 'neo4j_export')
        os.makedirs(neo4j_dir, exist_ok=True)
        
        print(f"\n💾 Exporting knowledge graph to {output_dir}/")
        
        # Export drugs
        drugs_file = os.path.join(neo4j_dir, 'drugs.csv')
        with open(drugs_file, 'w') as f:
            f.write('drugbank_id,name,description,cas_number,indication,mechanism,half_life,atc_codes,is_cardiovascular,is_antithrombotic\n')
            for drug in self.drugs.values():
                desc = drug.description.replace('"', '""')[:200]
                ind = drug.indication.replace('"', '""')[:200]
                mech = drug.mechanism.replace('"', '""')[:200]
                atc = '|'.join(drug.atc_codes)
                f.write(f'"{drug.drugbank_id}","{drug.name}","{desc}","{drug.cas_number}","{ind}","{mech}","{drug.half_life}","{atc}",{drug.is_cardiovascular},{drug.is_antithrombotic}\n')
        print(f"  ✓ drugs.csv: {len(self.drugs):,} drugs")
        
        # Export DDIs
        ddi_file = os.path.join(neo4j_dir, 'ddi_edges.csv')
        with open(ddi_file, 'w') as f:
            f.write('drug1_id,drug2_id,drug1_name,drug2_name,description,severity\n')
            for ddi in self.ddis:
                desc = ddi.description.replace('"', '""')[:200]
                f.write(f'"{ddi.drug1_id}","{ddi.drug2_id}","{ddi.drug1_name}","{ddi.drug2_name}","{desc}","{ddi.severity}"\n')
        print(f"  ✓ ddi_edges.csv: {len(self.ddis):,} interactions")
        
        # Export proteins
        proteins_file = os.path.join(neo4j_dir, 'proteins.csv')
        with open(proteins_file, 'w') as f:
            f.write('uniprot_id,name,gene_name,organism\n')
            for protein in self.proteins.values():
                name = protein.name.replace('"', '""')
                f.write(f'"{protein.uniprot_id}","{name}","{protein.gene_name}","{protein.organism}"\n')
        print(f"  ✓ proteins.csv: {len(self.proteins):,} proteins")
        
        # Export categories
        categories_file = os.path.join(neo4j_dir, 'categories.csv')
        with open(categories_file, 'w') as f:
            f.write('category_id,name\n')
            for i, cat in enumerate(sorted(self.categories)):
                cat_clean = cat.replace('"', '""')
                f.write(f'"CAT{i:05d}","{cat_clean}"\n')
        print(f"  ✓ categories.csv: {len(self.categories):,} categories")
        
        # Export drug-protein edges
        drug_protein_file = os.path.join(neo4j_dir, 'drug_protein_edges.csv')
        with open(drug_protein_file, 'w') as f:
            f.write('drug_id,protein_id,interaction_type\n')
            count = 0
            for drug in self.drugs.values():
                for target in drug.targets:
                    f.write(f'"{drug.drugbank_id}","{target}","target"\n')
                    count += 1
                for enzyme in drug.enzymes:
                    f.write(f'"{drug.drugbank_id}","{enzyme}","enzyme"\n')
                    count += 1
        print(f"  ✓ drug_protein_edges.csv: {count:,} edges")
        
        # Export drug-category edges
        drug_cat_file = os.path.join(neo4j_dir, 'drug_category_edges.csv')
        cat_to_id = {cat: f"CAT{i:05d}" for i, cat in enumerate(sorted(self.categories))}
        with open(drug_cat_file, 'w') as f:
            f.write('drug_id,category_id\n')
            count = 0
            for drug in self.drugs.values():
                for cat in drug.categories:
                    if cat in cat_to_id:
                        f.write(f'"{drug.drugbank_id}","{cat_to_id[cat]}"\n')
                        count += 1
        print(f"  ✓ drug_category_edges.csv: {count:,} edges")
        
        # Export statistics
        stats = {
            'total_drugs_scanned': self.stats['total_drugs'],
            'cardiovascular_drugs': self.stats['cardiovascular_drugs'],
            'antithrombotic_drugs': self.stats['antithrombotic_drugs'],
            'filtered_drugs': len(self.drugs),
            'total_ddis_scanned': self.stats['total_ddis'],
            'filtered_ddis': len(self.ddis),
            'proteins': len(self.proteins),
            'categories': len(self.categories),
            'faers_matched': len([d for d in self.drugs.values() if d.name.lower() in self.faers_data]),
            'filter_criteria': 'At least one drug is cardiovascular (ATC C*) OR antithrombotic (ATC B01*)'
        }
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ statistics.json")
        
        return output_dir


# =============================================================================
# Main Application
# =============================================================================

def get_drugbank_path() -> str:
    """Get DrugBank XML path from args, env, or prompt"""
    print("=" * 70)
    print("  DDI Risk Analysis - Cardiovascular & Antithrombotic")
    print("=" * 70)
    print()
    print("This application builds a knowledge graph filtered to:")
    print("  • Cardiovascular drugs (ATC code starts with C)")
    print("  • Antithrombotic drugs (ATC code starts with B01)")
    print()
    print("DrugBank XML required (academic/commercial license):")
    print("  → https://go.drugbank.com/releases/latest")
    print()
    print("Shareable data (FAERS, high-risk classes) included in repo.")
    print()
    
    # Check command line arguments first
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            print(f"Using DrugBank from argument: {path}")
            return path
        else:
            print(f"  ✗ File not found: {path}")
            sys.exit(1)
    
    # Check environment variable
    env_path = os.environ.get('DRUGBANK_XML')
    if env_path and os.path.exists(env_path):
        print(f"Using DrugBank from DRUGBANK_XML env: {env_path}")
        return env_path
    
    # Check common locations
    common_paths = [
        "data/full database.xml",
        "../data/full database.xml",
        os.path.expanduser("~/drugbank/full database.xml"),
        os.path.expanduser("~/Downloads/full database.xml"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"Found DrugBank at: {path}")
            try:
                response = input("Use this file? [Y/n]: ").strip().lower()
                if response in ('', 'y', 'yes'):
                    return path
            except (EOFError, KeyboardInterrupt):
                return path
    
    # Ask for path interactively
    try:
        while True:
            path = input("\nEnter path to DrugBank XML file: ").strip()
            path = path.strip('"').strip("'")
            
            if os.path.exists(path):
                return path
            else:
                print(f"  ✗ File not found: {path}")
                print("    Please check the path and try again.")
    except (EOFError, KeyboardInterrupt):
        print("\n\nUsage: python run_app.py <path_to_drugbank.xml>")
        print("   or: DRUGBANK_XML=/path/to/file.xml python run_app.py")
        sys.exit(1)


def build_and_run():
    """Build filtered knowledge graph and run the app"""
    # Get DrugBank path from user
    drugbank_path = get_drugbank_path()
    
    # Output directory for generated KG
    output_dir = "knowledge_graph_cardio"
    
    # Build filtered knowledge graph
    print()
    builder = CardioKnowledgeGraphBuilder(drugbank_path)
    builder.load_reference_data()
    builder.build_atc_lookup()
    builder.parse_filtered_drugs()
    builder.enrich_with_faers()
    builder.export_for_app(output_dir)
    
    print()
    print("=" * 70)
    print("  Knowledge Graph Built Successfully!")
    print("=" * 70)
    print()
    print(f"  Filter: Cardiovascular (ATC C*) OR Antithrombotic (ATC B01*)")
    print(f"  Drugs: {len(builder.drugs):,}")
    print(f"  DDIs: {len(builder.ddis):,}")
    print()
    
    # Copy generated KG to the location expected by ddi_app.py
    import shutil
    expected_dir = "knowledge_graph_fact_based"
    
    # Backup existing if present
    neo4j_exists = os.path.exists(f"{expected_dir}/neo4j_export/drugs.csv")
    if neo4j_exists:
        backup_dir = f"{expected_dir}_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        print(f"  Backing up existing KG to {backup_dir}/")
        shutil.move(expected_dir, backup_dir)
    
    # Copy to expected location
    if os.path.exists(expected_dir):
        shutil.rmtree(expected_dir)
    shutil.copytree(output_dir, expected_dir)
    print(f"✓ Knowledge graph ready at {expected_dir}/")
    
    # Launch the app
    print()
    print("🚀 Launching DDI Analysis Application...")
    print("   URL: http://localhost:7860")
    print("   Press Ctrl+C to stop")
    print()
    
    # Import and run ddi_app
    import ddi_app
    
    print("Loading Knowledge Graph...")
    result = ddi_app.kg.load()
    print(f"   {result}")
    
    print("\nStarting application...")
    app = ddi_app.create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    build_and_run()
