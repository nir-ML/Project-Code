#!/usr/bin/env python3
"""
Generate publication-quality figures for the DDI Knowledge Graph.

Creates:
1. Node distribution pie chart
2. Edge distribution bar chart
3. Severity distribution comparison
4. Data source contribution chart
5. Knowledge graph schema diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    'drugs': '#1f77b4',
    'proteins': '#ff7f0e',
    'side_effects': '#2ca02c',
    'diseases': '#d62728',
    'pathways': '#9467bd',
    'categories': '#8c564b',
    'ddi': '#1f77b4',
    'drug_se': '#2ca02c',
    'drug_disease': '#d62728',
    'drug_pathway': '#9467bd',
    'drug_protein': '#ff7f0e',
    'drug_category': '#8c564b',
}

SEVERITY_COLORS = {
    'Contraindicated': '#d62728',
    'Major': '#ff7f0e',
    'Moderate': '#2ca02c',
    'Minor': '#1f77b4',
}


def load_statistics():
    """Load knowledge graph statistics."""
    stats_path = Path(__file__).parent.parent / 'knowledge_graph_fact_based' / 'statistics.json'
    with open(stats_path, 'r') as f:
        return json.load(f)


def fig1_node_distribution():
    """Create node type distribution pie chart."""
    stats = load_statistics()
    nodes = stats['nodes']
    
    # Remove total
    node_types = {k: v for k, v in nodes.items() if k != 'total'}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = [k.replace('_', ' ').title() for k in node_types.keys()]
    sizes = list(node_types.values())
    colors = [COLORS.get(k, '#888888') for k in node_types.keys()]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes)):,})',
        startangle=90,
        pctdistance=0.75,
        explode=[0.02] * len(sizes)
    )
    
    # Style
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    ax.set_title(f'Knowledge Graph Node Distribution\n(Total: {nodes["total"]:,} nodes)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_node_distribution.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_node_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Created fig1_node_distribution")


def fig2_edge_distribution():
    """Create edge type distribution bar chart."""
    stats = load_statistics()
    edges = stats['edges']
    
    # Remove total and rename
    edge_data = {
        'DDI': edges['ddi'],
        'Drug-Side Effect': edges['drug_side_effect'],
        'Drug-Category': edges['drug_category'],
        'Drug-Disease': edges['drug_disease'],
        'Drug-Pathway': edges['drug_pathway'],
        'Drug-Protein': edges['drug_protein'],
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(edge_data.keys())
    values = list(edge_data.values())
    colors = ['#1f77b4', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#ff7f0e']
    
    bars = ax.barh(labels, values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2, 
                f'{val:,} ({val/sum(values)*100:.1f}%)',
                va='center', fontsize=9)
    
    ax.set_xlabel('Number of Edges')
    ax.set_title(f'Knowledge Graph Edge Distribution\n(Total: {edges["total"]:,} edges)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_edge_distribution.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_edge_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Created fig2_edge_distribution")


def fig3_severity_distribution():
    """Create severity distribution chart."""
    # Load severity data
    severity_data = pd.read_csv(Path(__file__).parent / 'data' / 'severity_distribution.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original distribution (from recalibration stats)
    original = {
        'Contraindicated': 432226,
        'Major': 326716,
        'Moderate': 24,
        'Minor': 808,
    }
    
    # Recalibrated distribution
    recalibrated = {
        'Moderate': 608742,
        'Minor': 70682,
        'Contraindicated': 44306,
        'Major': 36044,
    }
    
    # Left subplot: Original
    ax1 = axes[0]
    labels1 = list(original.keys())
    values1 = list(original.values())
    colors1 = [SEVERITY_COLORS[l] for l in labels1]
    
    wedges1, texts1, autotexts1 = ax1.pie(
        values1, labels=labels1, colors=colors1,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90
    )
    ax1.set_title('Original Distribution\n(Zero-Shot BART)', fontsize=12, fontweight='bold')
    
    # Right subplot: Recalibrated
    ax2 = axes[1]
    labels2 = list(recalibrated.keys())
    values2 = list(recalibrated.values())
    colors2 = [SEVERITY_COLORS[l] for l in labels2]
    
    wedges2, texts2, autotexts2 = ax2.pie(
        values2, labels=labels2, colors=colors2,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90
    )
    ax2.set_title('Recalibrated Distribution\n(Empirical Keyword-Based)', fontsize=12, fontweight='bold')
    
    plt.suptitle('DDI Severity Distribution Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_severity_distribution.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_severity_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Created fig3_severity_distribution")


def fig4_data_sources():
    """Create data source contribution chart."""
    stats = load_statistics()
    sources = stats['sources']
    
    # Group by database
    source_groups = {
        'DrugBank': sum(v for k, v in sources.items() if 'DrugBank' in k),
        'SIDER': sum(v for k, v in sources.items() if 'SIDER' in k),
        'CTD': sum(v for k, v in sources.items() if 'CTD' in k),
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = list(source_groups.keys())
    values = list(source_groups.values())
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(values)):,})',
        startangle=90,
        explode=[0.02] * len(values)
    )
    
    ax.set_title('Data Source Contributions\n(Edge Count by Source)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_data_sources.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_data_sources.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Created fig4_data_sources")


def fig5_kg_schema():
    """Create knowledge graph schema diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Node positions (circular layout)
    positions = {
        'Drug': (0.5, 0.5),
        'Protein': (0.15, 0.7),
        'Side Effect': (0.85, 0.7),
        'Disease': (0.15, 0.3),
        'Pathway': (0.85, 0.3),
        'Category': (0.5, 0.1),
    }
    
    node_colors = {
        'Drug': '#1f77b4',
        'Protein': '#ff7f0e',
        'Side Effect': '#2ca02c',
        'Disease': '#d62728',
        'Pathway': '#9467bd',
        'Category': '#8c564b',
    }
    
    node_counts = {
        'Drug': '4,313',
        'Protein': '3,176',
        'Side Effect': '5,548',
        'Disease': '3,041',
        'Pathway': '25,958',
        'Category': '3,619',
    }
    
    # Draw edges first
    edges = [
        ('Drug', 'Drug', 'INTERACTS_WITH\n759,774'),
        ('Drug', 'Protein', 'TARGETS\n21,559'),
        ('Drug', 'Side Effect', 'CAUSES\n265,238'),
        ('Drug', 'Disease', 'ASSOCIATED\n63,278'),
        ('Drug', 'Pathway', 'PARTICIPATES\n31,207'),
        ('Drug', 'Category', 'BELONGS_TO\n70,618'),
    ]
    
    for src, tgt, label in edges:
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        
        if src == tgt:  # Self-loop for DDI
            circle = plt.Circle((x1, y1 + 0.15), 0.08, fill=False, 
                               color='gray', linewidth=2)
            ax.add_patch(circle)
            ax.annotate(label, xy=(x1, y1 + 0.25), ha='center', fontsize=8)
        else:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw nodes
    for node, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.08, color=node_colors[node], alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y + 0.01, node, ha='center', va='center', fontsize=10, 
               fontweight='bold', color='white')
        ax.text(x, y - 0.03, f'n={node_counts[node]}', ha='center', va='center', 
               fontsize=8, color='white')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('DDI Knowledge Graph Schema', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_kg_schema.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_kg_schema.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Created fig5_kg_schema")


def fig6_4panel_summary():
    """Create 4-panel summary figure."""
    stats = load_statistics()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: Node distribution
    ax1 = axes[0, 0]
    nodes = {k: v for k, v in stats['nodes'].items() if k != 'total'}
    labels1 = [k.replace('_', ' ').title() for k in nodes.keys()]
    sizes1 = list(nodes.values())
    colors1 = [COLORS.get(k, '#888888') for k in nodes.keys()]
    ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%', startangle=90)
    ax1.set_title('A) Node Type Distribution', fontsize=12, fontweight='bold')
    
    # Panel B: Edge distribution
    ax2 = axes[0, 1]
    edge_data = {
        'DDI': stats['edges']['ddi'],
        'Drug-SE': stats['edges']['drug_side_effect'],
        'Drug-Cat': stats['edges']['drug_category'],
        'Drug-Dis': stats['edges']['drug_disease'],
        'Drug-Path': stats['edges']['drug_pathway'],
        'Drug-Prot': stats['edges']['drug_protein'],
    }
    bars = ax2.barh(list(edge_data.keys()), list(edge_data.values()), 
                   color=['#1f77b4', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#ff7f0e'])
    ax2.set_xlabel('Number of Edges')
    ax2.set_title('B) Edge Type Distribution', fontsize=12, fontweight='bold')
    
    # Panel C: Severity distribution (recalibrated)
    ax3 = axes[1, 0]
    severity_data = {
        'Moderate': 608742,
        'Minor': 70682,
        'Contraindicated': 44306,
        'Major': 36044,
    }
    colors3 = [SEVERITY_COLORS[k] for k in severity_data.keys()]
    ax3.pie(list(severity_data.values()), labels=list(severity_data.keys()), 
           colors=colors3, autopct='%1.1f%%', startangle=90)
    ax3.set_title('C) DDI Severity Distribution (Recalibrated)', fontsize=12, fontweight='bold')
    
    # Panel D: Data sources
    ax4 = axes[1, 1]
    source_groups = {
        'DrugBank': sum(v for k, v in stats['sources'].items() if 'DrugBank' in k),
        'SIDER': sum(v for k, v in stats['sources'].items() if 'SIDER' in k),
        'CTD': sum(v for k, v in stats['sources'].items() if 'CTD' in k),
    }
    bars4 = ax4.bar(list(source_groups.keys()), list(source_groups.values()),
                   color=['#1f77b4', '#2ca02c', '#d62728'])
    ax4.set_ylabel('Number of Associations')
    ax4.set_title('D) Data Source Contributions', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('DDI Knowledge Graph Summary\n(45,655 nodes, 1,211,674 edges)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_4panel_summary.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_4panel_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Created fig6_4panel_summary")


def main():
    """Generate all figures."""
    print("Generating Knowledge Graph publication figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    fig1_node_distribution()
    fig2_edge_distribution()
    fig3_severity_distribution()
    fig4_data_sources()
    fig5_kg_schema()
    fig6_4panel_summary()
    
    print("\nAll figures generated successfully!")
    print(f"Files saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
