#!/usr/bin/env python3
"""
Build Cardiovascular & Antithrombotic Knowledge Graph
Non-interactive version - specify paths as arguments or uses defaults.

Usage:
    python build_cardio_kg_now.py
    python build_cardio_kg_now.py /path/to/drugbank.xml
    python build_cardio_kg_now.py /path/to/drugbank.xml output_dir
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_kg_cardio import CardioKGBuilder

def main():
    # Get paths from args or use defaults
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        # Try common locations
        candidates = [
            "data/full database.xml",
            "/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/data/full database.xml",
            "/home/nbhatta1/Desktop/Project-repo/data/full database.xml",
        ]
        xml_path = None
        for c in candidates:
            if os.path.exists(c):
                xml_path = c
                break
        
        if not xml_path:
            print("ERROR: DrugBank XML not found. Please provide path as argument.")
            print("Usage: python build_cardio_kg_now.py /path/to/drugbank.xml")
            return 1
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "knowledge_graph_cardio"
    
    # Verify XML exists
    if not os.path.exists(xml_path):
        print(f"ERROR: File not found: {xml_path}")
        return 1
    
    print("=" * 70)
    print("  Cardiovascular & Antithrombotic Knowledge Graph Builder")
    print("=" * 70)
    print()
    print(f"DrugBank XML: {xml_path}")
    print(f"Output:       {output_dir}/")
    print()
    print("Filtering to drugs with:")
    print("  - ATC codes starting with 'C' (Cardiovascular)")
    print("  - ATC codes starting with 'B01' (Antithrombotic)")
    print()
    
    # Build
    builder = CardioKGBuilder(xml_path)
    
    # Step 1: Build ATC lookup
    builder.build_atc_lookup()
    
    # Step 2: Parse filtered drugs
    builder.parse_filtered_drugs()
    
    # Step 3: Extract diseases from indications
    builder.extract_diseases_from_indications()
    
    # Export
    builder.export(output_dir)
    
    # Final stats
    stats = builder.get_statistics()
    
    print()
    print("=" * 70)
    print("KNOWLEDGE GRAPH COMPLETE")
    print("=" * 70)
    print()
    print("Filter Results:")
    print(f"  Total drugs in XML:        {stats['filter_stats']['total_drugs_in_xml']:,}")
    print(f"  Cardiovascular (ATC C*):   {stats['filter_stats']['cardiovascular_drugs']:,}")
    print(f"  Antithrombotic (ATC B01*): {stats['filter_stats']['antithrombotic_drugs']:,}")
    print(f"  Selected (C* OR B01*):     {stats['filter_stats']['cardio_or_antithrombotic']:,}")
    print()
    print("Knowledge Graph Nodes:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    print()
    print("Knowledge Graph Edges:")
    for edge_type, count in stats['edges'].items():
        print(f"  {edge_type}: {count:,}")
    print()
    print(f"Output saved to: {output_dir}/")
    print("  - knowledge_graph.pkl (full graph)")
    print("  - ddi_cardio_antithrombotic.csv (DDI data for main app)")
    print("  - neo4j_export/*.csv (for Neo4j)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
