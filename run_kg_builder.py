#!/usr/bin/env python3
"""
Interactive Knowledge Graph Builder

Prompts for DrugBank XML path and builds the knowledge graph using APIs.
"""

import os
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("  Drug Interaction Knowledge Graph Builder")
    print("=" * 60)
    print()
    print("This tool builds a knowledge graph using:")
    print("  - DrugBank XML (local file, requires license)")
    print("  - OpenFDA API (public, for side effects)")
    print()
    
    # Ask for DrugBank XML path
    default_xml = "data/full database.xml"
    
    print(f"Enter path to DrugBank XML file")
    print(f"  [Press Enter for default: {default_xml}]")
    xml_path = input("> ").strip()
    
    if not xml_path:
        xml_path = default_xml
    
    # Validate path
    if not os.path.exists(xml_path):
        print(f"\n❌ Error: File not found: {xml_path}")
        print("\nPlease download DrugBank XML from:")
        print("  https://go.drugbank.com/releases/latest")
        return 1
    
    print(f"\n✓ Found DrugBank XML: {xml_path}")
    
    # Ask for CSV path (optional)
    print("\nEnter path to DDI CSV file (optional, filters drugs)")
    print("  [Press Enter to skip - will include all DrugBank drugs]")
    csv_path = input("> ").strip()
    
    if csv_path and not os.path.exists(csv_path):
        print(f"  ⚠ Warning: CSV not found, proceeding without filter")
        csv_path = None
    elif csv_path:
        print(f"  ✓ Using CSV filter: {csv_path}")
    
    # Ask for output directory
    print("\nEnter output directory name")
    print("  [Press Enter for default: knowledge_graph_output]")
    output_dir = input("> ").strip()
    
    if not output_dir:
        output_dir = "knowledge_graph_output"
    
    # Ask about API calls
    print("\nFetch side effects from OpenFDA API?")
    print("  [y/N] (API calls take ~15 seconds per 20 drugs)")
    use_api = input("> ").strip().lower()
    
    max_drugs = None
    if use_api in ['y', 'yes']:
        print("\nLimit API calls to how many drugs? (saves time)")
        print("  [Press Enter for all drugs, or enter a number like 50]")
        max_input = input("> ").strip()
        if max_input:
            try:
                max_drugs = int(max_input)
                print(f"  ✓ Will fetch side effects for {max_drugs} drugs")
            except ValueError:
                print("  ⚠ Invalid number, will process all drugs")
    
    # Confirm
    print("\n" + "=" * 60)
    print("Configuration:")
    print(f"  DrugBank XML: {xml_path}")
    print(f"  CSV Filter:   {csv_path if csv_path else 'None (all drugs)'}")
    print(f"  Output:       {output_dir}/")
    print(f"  OpenFDA API:  {'Yes' if use_api in ['y', 'yes'] else 'No'}")
    if max_drugs:
        print(f"  Max Drugs:    {max_drugs}")
    print("=" * 60)
    
    print("\nProceed? [Y/n]")
    confirm = input("> ").strip().lower()
    
    if confirm in ['n', 'no']:
        print("Cancelled.")
        return 0
    
    print("\n" + "=" * 60)
    print("Building Knowledge Graph...")
    print("=" * 60 + "\n")
    
    # Import and run
    from build_kg_api_based import APIBasedKGBuilder
    
    builder = APIBasedKGBuilder(xml_path, csv_path)
    
    # Load CSV identifiers if provided
    if csv_path:
        builder.load_csv_identifiers()
    
    # Parse DrugBank XML
    print("Parsing DrugBank XML (this may take 1-2 minutes)...")
    builder.parse_drugbank_xml()
    
    print(f"\n✓ Parsed {len(builder.drugs)} drugs")
    print(f"✓ Parsed {len(builder.proteins)} proteins")
    print(f"✓ Parsed {len(builder.pathways)} pathways")
    print(f"✓ Found {len(builder.ddi_edges)} drug-drug interactions")
    
    # Fetch side effects from API
    if use_api in ['y', 'yes']:
        print("\nFetching side effects from OpenFDA API...")
        builder.fetch_openfda_side_effects(max_drugs=max_drugs)
        print(f"✓ Loaded {len(builder.side_effects)} side effects")
    
    # Extract diseases from indications
    print("\nExtracting diseases from indications...")
    builder.extract_diseases_from_indications()
    print(f"✓ Extracted {len(builder.diseases)} disease associations")
    
    # Export
    print(f"\nExporting to {output_dir}/...")
    builder.export(output_dir)
    
    # Final statistics
    stats = builder.get_statistics()
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH COMPLETE")
    print("=" * 60)
    
    print("\nNodes:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    
    print("\nEdges:")
    for edge_type, count in stats['edges'].items():
        print(f"  {edge_type}: {count:,}")
    
    print(f"\nOutput saved to: {output_dir}/")
    print("  - knowledge_graph.pkl (full graph)")
    print("  - neo4j_export/*.csv (for Neo4j import)")
    print("  - statistics.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
