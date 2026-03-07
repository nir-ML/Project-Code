#!/usr/bin/env python3
"""
Interactive Knowledge Graph Builder for Cardiovascular & Antithrombotic Drugs

Filters DrugBank XML to include only:
- Cardiovascular drugs (ATC codes starting with 'C')
- Antithrombotic drugs (ATC codes starting with 'B01')

Then builds a knowledge graph using OpenFDA API for side effects.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DrugBank XML namespace
NS = {'db': 'http://www.drugbank.ca'}


def classify_atc(atc_list: List[str]) -> tuple:
    """Classify drug based on ATC codes."""
    if not atc_list:
        return False, False
    
    is_cardiovascular = any(code.startswith("C") for code in atc_list)
    is_antithrombotic = any(code.startswith("B01") for code in atc_list)
    
    return is_cardiovascular, is_antithrombotic


class CardioKGBuilder:
    """
    Knowledge Graph Builder filtered for Cardiovascular & Antithrombotic drugs.
    """
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        
        # Drug data
        self.drugs: Dict[str, dict] = {}
        self.drug_atc_lookup: Dict[str, List[str]] = {}
        self.cardio_drug_ids: Set[str] = set()
        
        # DDI data
        self.ddi_edges: List[dict] = []
        
        # Protein data
        self.proteins: Dict[str, dict] = {}
        self.drug_protein_edges: List[dict] = []
        
        # Pathway data
        self.pathways: Dict[str, dict] = {}
        self.drug_pathway_edges: List[dict] = []
        
        # Side effects (from OpenFDA)
        self.side_effects: Dict[str, dict] = {}
        self.drug_se_edges: List[dict] = []
        
        # Categories
        self.categories: Dict[str, dict] = {}
        self.drug_category_edges: List[dict] = []
        
        # Disease associations
        self.diseases: Dict[str, dict] = {}
        self.drug_disease_edges: List[dict] = []
        
        # Statistics
        self.stats = {
            'total_drugs_in_xml': 0,
            'cardiovascular_drugs': 0,
            'antithrombotic_drugs': 0,
            'cardio_or_antithrombotic': 0,
        }
    
    def _get_text(self, elem, tag: str) -> str:
        """Safely get text from XML element."""
        child = elem.find(f'db:{tag}', NS)
        if child is not None and child.text:
            return child.text.strip()
        return ""
    
    def build_atc_lookup(self) -> None:
        """First pass: Build ATC code lookup for all drugs."""
        logger.info("Building ATC code lookup...")
        
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        for drug_elem in root.findall('db:drug', NS):
            drug_id_elem = drug_elem.find("db:drugbank-id[@primary='true']", NS)
            if drug_id_elem is None or not drug_id_elem.text:
                continue
            
            drug_id = drug_id_elem.text.strip()
            self.stats['total_drugs_in_xml'] += 1
            
            # Get all ATC codes
            atc_codes = []
            atc_elem = drug_elem.find('db:atc-codes', NS)
            if atc_elem is not None:
                for level in atc_elem.findall('.//db:level', NS):
                    code = level.get('code')
                    if code and len(code) == 7:  # Full ATC code
                        atc_codes.append(code)
            
            self.drug_atc_lookup[drug_id] = atc_codes
            
            # Classify
            is_cardio, is_antithrombotic = classify_atc(atc_codes)
            
            if is_cardio:
                self.stats['cardiovascular_drugs'] += 1
            if is_antithrombotic:
                self.stats['antithrombotic_drugs'] += 1
            if is_cardio or is_antithrombotic:
                self.cardio_drug_ids.add(drug_id)
                self.stats['cardio_or_antithrombotic'] += 1
        
        logger.info(f"  Total drugs in XML: {self.stats['total_drugs_in_xml']}")
        logger.info(f"  Cardiovascular drugs (ATC C*): {self.stats['cardiovascular_drugs']}")
        logger.info(f"  Antithrombotic drugs (ATC B01*): {self.stats['antithrombotic_drugs']}")
        logger.info(f"  Cardio OR Antithrombotic: {self.stats['cardio_or_antithrombotic']}")
    
    def parse_filtered_drugs(self) -> None:
        """Second pass: Parse full drug data for filtered drugs only."""
        logger.info("\nParsing drug data for cardiovascular/antithrombotic drugs...")
        
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        for drug_elem in root.findall('db:drug', NS):
            drug_id_elem = drug_elem.find("db:drugbank-id[@primary='true']", NS)
            if drug_id_elem is None or not drug_id_elem.text:
                continue
            
            drug_id = drug_id_elem.text.strip()
            
            # Only process cardiovascular/antithrombotic drugs
            if drug_id not in self.cardio_drug_ids:
                continue
            
            drug_name = self._get_text(drug_elem, 'name')
            atc_codes = self.drug_atc_lookup.get(drug_id, [])
            is_cardio, is_antithrombotic = classify_atc(atc_codes)
            
            # Get synonyms
            synonyms = []
            syns_elem = drug_elem.find('db:synonyms', NS)
            if syns_elem is not None:
                for syn in syns_elem.findall('db:synonym', NS):
                    if syn.text:
                        synonyms.append(syn.text.strip())
            
            # Get CAS number
            cas_number = ""
            calc_props = drug_elem.find('db:calculated-properties', NS)
            if calc_props is not None:
                for prop in calc_props.findall('db:property', NS):
                    kind = self._get_text(prop, 'kind')
                    if kind == 'CAS':
                        cas_number = self._get_text(prop, 'value')
                        break
            
            # Get groups
            groups = []
            groups_elem = drug_elem.find('db:groups', NS)
            if groups_elem is not None:
                for grp in groups_elem.findall('db:group', NS):
                    if grp.text:
                        groups.append(grp.text.strip())
            
            self.drugs[drug_id] = {
                'drugbank_id': drug_id,
                'name': drug_name,
                'synonyms': synonyms,
                'cas_number': cas_number,
                'atc_codes': atc_codes,
                'is_cardiovascular': is_cardio,
                'is_antithrombotic': is_antithrombotic,
                'type': drug_elem.get('type', ''),
                'description': self._get_text(drug_elem, 'description')[:500],
                'indication': self._get_text(drug_elem, 'indication'),
                'mechanism': self._get_text(drug_elem, 'mechanism-of-action'),
                'pharmacodynamics': self._get_text(drug_elem, 'pharmacodynamics'),
                'groups': groups,
            }
            
            # Parse proteins (targets, enzymes, carriers, transporters)
            self._parse_proteins(drug_elem, drug_id)
            
            # Parse pathways
            self._parse_pathways(drug_elem, drug_id)
            
            # Parse categories
            self._parse_categories(drug_elem, drug_id)
            
            # Parse DDIs (filter to cardio drugs only)
            self._parse_ddis(drug_elem, drug_id)
        
        logger.info(f"  Parsed {len(self.drugs)} drugs")
        logger.info(f"  Parsed {len(self.proteins)} proteins")
        logger.info(f"  Parsed {len(self.pathways)} pathways")
        logger.info(f"  Parsed {len(self.ddi_edges)} DDI edges (filtered)")
    
    def _parse_proteins(self, drug_elem, drug_id: str) -> None:
        """Parse protein targets, enzymes, carriers, transporters."""
        for protein_type in ['targets', 'enzymes', 'carriers', 'transporters']:
            container = drug_elem.find(f'db:{protein_type}', NS)
            if container is None:
                continue
            
            singular = protein_type[:-1]
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
                actions = []
                actions_elem = protein_elem.find('db:actions', NS)
                if actions_elem is not None:
                    for action in actions_elem.findall('db:action', NS):
                        if action.text:
                            actions.append(action.text.strip())
                
                if protein_id not in self.proteins:
                    self.proteins[protein_id] = {
                        'id': protein_id,
                        'name': protein_name,
                        'type': singular,
                        'uniprot_id': uniprot_id,
                        'gene_name': gene_name,
                    }
                
                self.drug_protein_edges.append({
                    'drug_id': drug_id,
                    'protein_id': protein_id,
                    'type': singular,
                    'actions': actions,
                })
    
    def _parse_pathways(self, drug_elem, drug_id: str) -> None:
        """Parse SMPDB pathways."""
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
                self.pathways[smpdb_id] = {
                    'id': smpdb_id,
                    'name': pathway_name,
                    'category': category,
                }
            
            self.drug_pathway_edges.append({
                'drug_id': drug_id,
                'pathway_id': smpdb_id,
            })
    
    def _parse_categories(self, drug_elem, drug_id: str) -> None:
        """Parse drug categories."""
        categories_elem = drug_elem.find('db:categories', NS)
        if categories_elem is None:
            return
        
        for cat_elem in categories_elem.findall('db:category', NS):
            cat_name = self._get_text(cat_elem, 'category')
            mesh_id = self._get_text(cat_elem, 'mesh-id')
            
            if not cat_name:
                continue
            
            if cat_name not in self.categories:
                self.categories[cat_name] = {
                    'name': cat_name,
                    'mesh_id': mesh_id,
                }
            
            self.drug_category_edges.append({
                'drug_id': drug_id,
                'category_name': cat_name,
            })
    
    def _parse_ddis(self, drug_elem, drug_id: str) -> None:
        """Parse DDIs, keeping only those involving cardio/antithrombotic drugs."""
        ddis_elem = drug_elem.find('db:drug-interactions', NS)
        if ddis_elem is None:
            return
        
        for ddi_elem in ddis_elem.findall('db:drug-interaction', NS):
            other_id_elem = ddi_elem.find('db:drugbank-id', NS)
            other_name_elem = ddi_elem.find('db:name', NS)
            description_elem = ddi_elem.find('db:description', NS)
            
            if other_id_elem is None or not other_id_elem.text:
                continue
            
            other_id = other_id_elem.text.strip()
            other_name = other_name_elem.text.strip() if other_name_elem is not None and other_name_elem.text else ""
            description = description_elem.text.strip() if description_elem is not None and description_elem.text else ""
            
            # Keep DDI if the other drug is also cardio/antithrombotic, OR
            # if we want to keep all DDIs involving our filtered drugs
            # Here we keep all DDIs where at least one drug is cardio/antithrombotic
            
            # Check if other drug is cardio/antithrombotic
            other_atc = self.drug_atc_lookup.get(other_id, [])
            other_is_cardio, other_is_antithrombotic = classify_atc(other_atc)
            
            self.ddi_edges.append({
                'drug1_id': drug_id,
                'drug1_name': self.drugs[drug_id]['name'],
                'drug1_is_cardio': self.drugs[drug_id]['is_cardiovascular'],
                'drug1_is_antithrombotic': self.drugs[drug_id]['is_antithrombotic'],
                'drug2_id': other_id,
                'drug2_name': other_name,
                'drug2_is_cardio': other_is_cardio,
                'drug2_is_antithrombotic': other_is_antithrombotic,
                'description': description,
            })
    
    def fetch_openfda_side_effects(self, max_drugs: int = None) -> None:
        """Fetch side effects from OpenFDA API."""
        logger.info("\nFetching side effects from OpenFDA API...")
        
        drug_list = list(self.drugs.values())
        if max_drugs:
            drug_list = drug_list[:max_drugs]
        
        logger.info(f"  Querying {len(drug_list)} drugs...")
        
        for i, drug in enumerate(drug_list):
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{len(drug_list)}")
            
            drug_name = drug['name']
            
            try:
                # Rate limit
                time.sleep(0.3)
                
                # Query OpenFDA
                url = "https://api.fda.gov/drug/event.json"
                params = {
                    "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                    "count": "patient.reaction.reactionmeddrapt.exact",
                    "limit": 15
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    for event in results:
                        term = event.get('term', '')
                        count = event.get('count', 0)
                        
                        if not term or count < 10:
                            continue
                        
                        se_id = f"FDA:{term.replace(' ', '_').upper()}"
                        
                        if se_id not in self.side_effects:
                            self.side_effects[se_id] = {
                                'id': se_id,
                                'name': term,
                                'source': 'OpenFDA',
                            }
                        
                        self.drug_se_edges.append({
                            'drug_id': drug['drugbank_id'],
                            'side_effect_id': se_id,
                            'count': count,
                        })
            
            except Exception as e:
                logger.debug(f"Error for {drug_name}: {e}")
        
        logger.info(f"  Loaded {len(self.side_effects)} unique side effects")
        logger.info(f"  Loaded {len(self.drug_se_edges)} drug-side effect links")
    
    def extract_diseases_from_indications(self) -> None:
        """Extract disease associations from indications."""
        logger.info("\nExtracting diseases from indications...")
        
        for drug_id, drug in self.drugs.items():
            indication = drug.get('indication', '')
            if not indication:
                continue
            
            disease_id = f"IND:{drug_id}"
            indication_preview = indication[:150].strip()
            
            self.diseases[disease_id] = {
                'id': disease_id,
                'name': indication_preview,
            }
            
            self.drug_disease_edges.append({
                'drug_id': drug_id,
                'disease_id': disease_id,
                'relationship': 'indicated_for',
            })
        
        logger.info(f"  Extracted {len(self.diseases)} disease associations")
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        return {
            'filter_stats': self.stats,
            'nodes': {
                'drugs': len(self.drugs),
                'proteins': len(self.proteins),
                'pathways': len(self.pathways),
                'categories': len(self.categories),
                'side_effects': len(self.side_effects),
                'diseases': len(self.diseases),
            },
            'edges': {
                'ddi': len(self.ddi_edges),
                'drug_protein': len(self.drug_protein_edges),
                'drug_pathway': len(self.drug_pathway_edges),
                'drug_category': len(self.drug_category_edges),
                'drug_side_effect': len(self.drug_se_edges),
                'drug_disease': len(self.drug_disease_edges),
            },
        }
    
    def build_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph."""
        logger.info("\nBuilding NetworkX graph...")
        
        G = nx.MultiDiGraph()
        
        # Add drug nodes
        for drug_id, drug in self.drugs.items():
            G.add_node(
                drug_id,
                node_type='drug',
                name=drug['name'],
                is_cardiovascular=drug['is_cardiovascular'],
                is_antithrombotic=drug['is_antithrombotic'],
                atc_codes=drug['atc_codes'],
            )
        
        # Add protein nodes
        for protein_id, protein in self.proteins.items():
            G.add_node(
                protein_id,
                node_type='protein',
                name=protein['name'],
                uniprot_id=protein['uniprot_id'],
            )
        
        # Add side effect nodes
        for se_id, se in self.side_effects.items():
            G.add_node(
                se_id,
                node_type='side_effect',
                name=se['name'],
            )
        
        # Add DDI edges
        for edge in self.ddi_edges:
            G.add_edge(
                edge['drug1_id'], edge['drug2_id'],
                edge_type='interacts_with',
                description=edge['description'][:200],
            )
        
        # Add drug-protein edges
        for edge in self.drug_protein_edges:
            G.add_edge(
                edge['drug_id'], edge['protein_id'],
                edge_type=f"has_{edge['type']}",
            )
        
        # Add drug-side effect edges
        for edge in self.drug_se_edges:
            G.add_edge(
                edge['drug_id'], edge['side_effect_id'],
                edge_type='has_side_effect',
            )
        
        logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def export(self, output_dir: str) -> None:
        """Export to files."""
        logger.info(f"\nExporting to {output_dir}/...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        neo4j_path = output_path / "neo4j_export"
        neo4j_path.mkdir(exist_ok=True)
        
        # Export drugs
        drug_data = []
        for drug_id, drug in self.drugs.items():
            drug_data.append({
                'drugbank_id': drug_id,
                'name': drug['name'],
                'is_cardiovascular': drug['is_cardiovascular'],
                'is_antithrombotic': drug['is_antithrombotic'],
                'atc_codes': ';'.join(drug['atc_codes']),
                'type': drug['type'],
                'indication': drug['indication'][:500] if drug['indication'] else "",
            })
        pd.DataFrame(drug_data).to_csv(neo4j_path / 'drugs.csv', index=False)
        
        # Export proteins
        pd.DataFrame(list(self.proteins.values())).to_csv(neo4j_path / 'proteins.csv', index=False)
        
        # Export pathways
        pd.DataFrame(list(self.pathways.values())).to_csv(neo4j_path / 'pathways.csv', index=False)
        
        # Export side effects
        pd.DataFrame(list(self.side_effects.values())).to_csv(neo4j_path / 'side_effects.csv', index=False)
        
        # Export DDI edges
        ddi_df = pd.DataFrame(self.ddi_edges)
        ddi_df.to_csv(neo4j_path / 'ddi_edges.csv', index=False)
        
        # Also save as the standard CSV format for the app
        ddi_app_df = ddi_df.rename(columns={
            'drug1_id': 'drugbank_id_1',
            'drug1_name': 'drug_name_1',
            'drug2_id': 'drugbank_id_2',
            'drug2_name': 'drug_name_2',
            'description': 'interaction_description',
        })
        ddi_app_df.to_csv(output_path / 'ddi_cardio_antithrombotic.csv', index=False)
        
        # Export other edges
        pd.DataFrame(self.drug_protein_edges).to_csv(neo4j_path / 'drug_protein_edges.csv', index=False)
        pd.DataFrame(self.drug_pathway_edges).to_csv(neo4j_path / 'drug_pathway_edges.csv', index=False)
        pd.DataFrame(self.drug_se_edges).to_csv(neo4j_path / 'drug_side_effect_edges.csv', index=False)
        pd.DataFrame(self.drug_category_edges).to_csv(neo4j_path / 'drug_category_edges.csv', index=False)
        pd.DataFrame(self.drug_disease_edges).to_csv(neo4j_path / 'drug_disease_edges.csv', index=False)
        
        # Save statistics
        stats = self.get_statistics()
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save full graph
        G = self.build_graph()
        with open(output_path / 'knowledge_graph.pkl', 'wb') as f:
            pickle.dump({
                'drugs': self.drugs,
                'proteins': self.proteins,
                'pathways': self.pathways,
                'categories': self.categories,
                'side_effects': self.side_effects,
                'diseases': self.diseases,
                'ddi_edges': self.ddi_edges,
                'drug_protein_edges': self.drug_protein_edges,
                'drug_pathway_edges': self.drug_pathway_edges,
                'drug_category_edges': self.drug_category_edges,
                'drug_se_edges': self.drug_se_edges,
                'drug_disease_edges': self.drug_disease_edges,
                'graph': G,
                'statistics': stats,
            }, f)
        
        logger.info("  Export complete!")


def main():
    """Interactive main function."""
    print("=" * 70)
    print("  Cardiovascular & Antithrombotic Drug Knowledge Graph Builder")
    print("=" * 70)
    print()
    print("This tool filters DrugBank to include only:")
    print("  - Cardiovascular drugs (ATC codes starting with 'C')")
    print("  - Antithrombotic drugs (ATC codes starting with 'B01')")
    print()
    print("Then builds a knowledge graph with OpenFDA side effects.")
    print()
    
    # Ask for XML path
    default_xml = "data/full database.xml"
    print(f"Enter path to DrugBank XML file:")
    print(f"  [Press Enter for default: {default_xml}]")
    xml_path = input("> ").strip()
    
    if not xml_path:
        xml_path = default_xml
    
    if not os.path.exists(xml_path):
        print(f"\n❌ Error: File not found: {xml_path}")
        return 1
    
    print(f"\n✓ Found: {xml_path}")
    
    # Ask for output directory
    print("\nEnter output directory name:")
    print("  [Press Enter for default: knowledge_graph_cardio]")
    output_dir = input("> ").strip()
    
    if not output_dir:
        output_dir = "knowledge_graph_cardio"
    
    # Ask about API
    print("\nFetch side effects from OpenFDA API?")
    print("  [y/N] (takes ~15 sec per 20 drugs)")
    use_api = input("> ").strip().lower()
    
    max_drugs = None
    if use_api in ['y', 'yes']:
        print("\nLimit API calls to how many drugs?")
        print("  [Press Enter for all, or enter a number like 100]")
        max_input = input("> ").strip()
        if max_input:
            try:
                max_drugs = int(max_input)
            except:
                pass
    
    # Confirm
    print("\n" + "=" * 70)
    print("Configuration:")
    print(f"  DrugBank XML: {xml_path}")
    print(f"  Output:       {output_dir}/")
    print(f"  OpenFDA API:  {'Yes' if use_api in ['y', 'yes'] else 'No'}")
    if max_drugs:
        print(f"  Max drugs:    {max_drugs}")
    print("=" * 70)
    
    print("\nProceed? [Y/n]")
    if input("> ").strip().lower() in ['n', 'no']:
        print("Cancelled.")
        return 0
    
    print("\n" + "=" * 70)
    print("Building Knowledge Graph...")
    print("=" * 70 + "\n")
    
    # Build
    builder = CardioKGBuilder(xml_path)
    
    # Step 1: Build ATC lookup and identify cardio/antithrombotic drugs
    builder.build_atc_lookup()
    
    # Step 2: Parse filtered drugs
    builder.parse_filtered_drugs()
    
    # Step 3: Fetch side effects (optional)
    if use_api in ['y', 'yes']:
        builder.fetch_openfda_side_effects(max_drugs=max_drugs)
    
    # Step 4: Extract diseases
    builder.extract_diseases_from_indications()
    
    # Export
    builder.export(output_dir)
    
    # Final stats
    stats = builder.get_statistics()
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH COMPLETE")
    print("=" * 70)
    
    print("\nFilter Results:")
    print(f"  Total drugs in XML:        {stats['filter_stats']['total_drugs_in_xml']:,}")
    print(f"  Cardiovascular (ATC C*):   {stats['filter_stats']['cardiovascular_drugs']:,}")
    print(f"  Antithrombotic (ATC B01*): {stats['filter_stats']['antithrombotic_drugs']:,}")
    print(f"  Selected (C* OR B01*):     {stats['filter_stats']['cardio_or_antithrombotic']:,}")
    
    print("\nKnowledge Graph Nodes:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    
    print("\nKnowledge Graph Edges:")
    for edge_type, count in stats['edges'].items():
        print(f"  {edge_type}: {count:,}")
    
    print(f"\nOutput saved to: {output_dir}/")
    print("  - knowledge_graph.pkl")
    print("  - ddi_cardio_antithrombotic.csv (for the main app)")
    print("  - neo4j_export/*.csv")
    
    return 0


if __name__ == "__main__":
    exit(main())
