#!/usr/bin/env python3
"""
Data Download Helper Script

This script helps users obtain the required datasets for the DDI Risk Analysis project.
Due to licensing restrictions, most datasets cannot be automatically downloaded.
This script provides instructions and validates that files are properly placed.
"""

import os
import sys
from pathlib import Path

# Expected data files and their sources
REQUIRED_FILES = {
    "data/full database.xml": {
        "source": "DrugBank",
        "url": "https://go.drugbank.com/releases/latest",
        "license": "Academic/Commercial License",
        "instructions": [
            "1. Create an account at https://go.drugbank.com/",
            "2. Apply for an Academic License (free for research)",
            "3. Download 'Full Database' in XML format",
            "4. Place the file in the 'data/' directory"
        ],
        "required": True
    },
    "data/drugbank.xsd": {
        "source": "DrugBank",
        "url": "https://go.drugbank.com/releases/latest",
        "license": "Academic/Commercial License",
        "instructions": [
            "Downloaded alongside the full database XML"
        ],
        "required": False
    },
    "external_data/ddinter/ddinter_all.csv": {
        "source": "DDInter",
        "url": "http://ddinter.scbdd.com/download/",
        "license": "CC BY-NC-SA 4.0",
        "instructions": [
            "1. Visit http://ddinter.scbdd.com/download/",
            "2. Download 'All DDIs' file",
            "3. Place in 'external_data/ddinter/'"
        ],
        "required": True
    },
    "external_data/sider/drug_names.tsv": {
        "source": "SIDER",
        "url": "http://sideeffects.embl.de/download/",
        "license": "CC BY-NC-SA 4.0",
        "instructions": [
            "1. Visit http://sideeffects.embl.de/download/",
            "2. Download 'drug_names.tsv'",
            "3. Place in 'external_data/sider/'"
        ],
        "required": True
    },
    "external_data/sider/meddra_all_se.tsv.gz": {
        "source": "SIDER",
        "url": "http://sideeffects.embl.de/download/",
        "license": "CC BY-NC-SA 4.0",
        "instructions": [
            "Download from SIDER and place in 'external_data/sider/'"
        ],
        "required": True
    },
    "external_data/sider/meddra_freq.tsv.gz": {
        "source": "SIDER",
        "url": "http://sideeffects.embl.de/download/",
        "license": "CC BY-NC-SA 4.0",
        "instructions": [
            "Download from SIDER and place in 'external_data/sider/'"
        ],
        "required": False
    },
    "external_data/ctd/CTD_chemicals_diseases.tsv.gz": {
        "source": "CTD",
        "url": "https://ctdbase.org/downloads/",
        "license": "Free for academic use",
        "instructions": [
            "1. Visit https://ctdbase.org/downloads/",
            "2. Download 'Chemical-disease associations'",
            "3. Place in 'external_data/ctd/'"
        ],
        "required": False
    }
}


def get_project_root():
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    # Script is in scripts/, so parent is project root
    if script_dir.name == "scripts":
        return script_dir.parent
    return script_dir


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists."""
    return filepath.exists() and filepath.is_file()


def print_header():
    """Print script header."""
    print("=" * 60)
    print("  DDI Risk Analysis - Data Setup Helper")
    print("=" * 60)
    print()
    print("This script helps you obtain the required datasets.")
    print("Due to licensing, most files must be downloaded manually.")
    print()
    print("For detailed instructions, see: DATA_SOURCES.md")
    print()


def check_all_files(root: Path) -> dict:
    """Check status of all required files."""
    status = {}
    for filepath, info in REQUIRED_FILES.items():
        full_path = root / filepath
        status[filepath] = {
            "exists": check_file_exists(full_path),
            "info": info,
            "path": full_path
        }
    return status


def print_status(status: dict):
    """Print status of all files."""
    print("-" * 60)
    print("FILE STATUS")
    print("-" * 60)
    
    found = []
    missing_required = []
    missing_optional = []
    
    for filepath, data in status.items():
        exists = data["exists"]
        required = data["info"]["required"]
        
        if exists:
            found.append(filepath)
        elif required:
            missing_required.append(filepath)
        else:
            missing_optional.append(filepath)
    
    # Print found files
    if found:
        print("\n✓ FOUND:")
        for f in found:
            print(f"  • {f}")
    
    # Print missing required files
    if missing_required:
        print("\n✗ MISSING (REQUIRED):")
        for f in missing_required:
            print(f"  • {f}")
    
    # Print missing optional files
    if missing_optional:
        print("\n○ MISSING (OPTIONAL):")
        for f in missing_optional:
            print(f"  • {f}")
    
    print()
    return missing_required, missing_optional


def print_instructions(status: dict, missing: list):
    """Print download instructions for missing files."""
    if not missing:
        return
    
    print("-" * 60)
    print("DOWNLOAD INSTRUCTIONS")
    print("-" * 60)
    
    # Group by source
    sources = {}
    for filepath in missing:
        source = status[filepath]["info"]["source"]
        if source not in sources:
            sources[source] = []
        sources[source].append(filepath)
    
    for source, files in sources.items():
        info = status[files[0]]["info"]
        print(f"\n{source}")
        print(f"  URL: {info['url']}")
        print(f"  License: {info['license']}")
        print(f"  Files needed: {', '.join(Path(f).name for f in files)}")
        print("  Steps:")
        for step in info["instructions"]:
            print(f"    {step}")


def create_directories(root: Path):
    """Create required directories if they don't exist."""
    dirs = [
        root / "data",
        root / "external_data" / "ddinter",
        root / "external_data" / "sider",
        root / "external_data" / "ctd",
        root / "knowledge_graph_fact_based" / "neo4j_export"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print("✓ Created required directories")


def create_gitkeep_files(root: Path):
    """Create .gitkeep files to preserve directory structure."""
    gitkeep_dirs = [
        root / "data",
        root / "external_data" / "ddinter",
        root / "external_data" / "sider",
        root / "external_data" / "ctd",
        root / "knowledge_graph_fact_based",
    ]
    
    for d in gitkeep_dirs:
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("✓ Created .gitkeep placeholder files")


def main():
    """Main entry point."""
    root = get_project_root()
    
    print_header()
    
    # Create directories
    print("-" * 60)
    print("SETUP")
    print("-" * 60)
    create_directories(root)
    create_gitkeep_files(root)
    print()
    
    # Check file status
    status = check_all_files(root)
    missing_required, missing_optional = print_status(status)
    
    # Print instructions for missing files
    all_missing = missing_required + missing_optional
    if all_missing:
        print_instructions(status, all_missing)
    
    # Summary
    print()
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    
    if not missing_required:
        print("\n✓ All required data files are present!")
        print("  You can now run: python build_fact_based_kg.py")
        return 0
    else:
        print(f"\n✗ {len(missing_required)} required file(s) missing")
        print("  Please download the files and run this script again.")
        print("\n  For detailed instructions, see: DATA_SOURCES.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
