#!/usr/bin/env python3
"""
Generate Circos-style Hierarchical Edge Bundling Visualization for DDI Knowledge Graph.

Creates a circular layout with:
- Nodes arranged by type around the perimeter
- Hierarchical edge bundling to reduce clutter
- Color-coded node types and edge types
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import pickle
import json
import sys
from pathlib import Path as FilePath
from collections import defaultdict
import random

# Add parent directory to path for imports
sys.path.insert(0, str(FilePath(__file__).parent.parent))
from build_fact_based_kg import (
    DrugNode, ProteinNode, SideEffectNode, DiseaseNode, 
    PathwayNode, CategoryNode, DDIEdge, DrugProteinEdge,
    DrugSideEffectEdge, DrugDiseaseEdge, DrugPathwayEdge,
    DrugCategoryEdge, SNPEffectData, Provenance
)

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# Output directory
OUTPUT_DIR = FilePath(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Color scheme (colorblind-friendly)
NODE_COLORS = {
    'drug': '#1f77b4',       # Blue
    'protein': '#ff7f0e',    # Orange
    'side_effect': '#2ca02c', # Green
    'disease': '#d62728',    # Red
    'pathway': '#9467bd',    # Purple
    'category': '#8c564b',   # Brown
}

EDGE_COLORS = {
    'ddi': '#4a4a4a',
    'drug_protein': '#ff7f0e',
    'drug_side_effect': '#2ca02c',
    'drug_disease': '#d62728',
    'drug_pathway': '#9467bd',
    'drug_category': '#8c564b',
}

SEVERITY_COLORS = {
    'Contraindicated interaction': '#d62728',
    'Major interaction': '#ff7f0e',
    'Moderate interaction': '#2ca02c',
    'Minor interaction': '#1f77b4',
}


def load_kg_data():
    """Load knowledge graph data."""
    kg_path = FilePath(__file__).parent.parent / 'knowledge_graph_fact_based' / 'knowledge_graph.pkl'
    
    with open(kg_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def bezier_curve(p0, p1, p2, n_points=50):
    """Generate quadratic Bezier curve points."""
    t = np.linspace(0, 1, n_points)
    curve = np.outer((1-t)**2, p0) + np.outer(2*(1-t)*t, p1) + np.outer(t**2, p2)
    return curve


def cubic_bezier(p0, p1, p2, p3, n_points=50):
    """Generate cubic Bezier curve points."""
    t = np.linspace(0, 1, n_points)
    curve = (np.outer((1-t)**3, p0) + 
             np.outer(3*(1-t)**2*t, p1) + 
             np.outer(3*(1-t)*t**2, p2) + 
             np.outer(t**3, p3))
    return curve


def hierarchical_edge_bundle(p0, p1, center=(0, 0), beta=0.85, n_points=50):
    """
    Create hierarchical edge bundling curve between two points.
    
    Args:
        p0: Start point (x, y)
        p1: End point (x, y)
        center: Center point for bundling
        beta: Bundling strength (0=straight, 1=fully bundled)
        n_points: Number of points in curve
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    center = np.array(center)
    
    # Control points pulled toward center
    ctrl1 = p0 + beta * (center - p0) * 0.5
    ctrl2 = p1 + beta * (center - p1) * 0.5
    
    return cubic_bezier(p0, ctrl1, ctrl2, p1, n_points)


def create_circos_plot(data, max_drugs=100, max_edges_per_type=500, output_name='circos_heb'):
    """
    Create Circos-style hierarchical edge bundling visualization.
    
    Args:
        data: Knowledge graph data
        max_drugs: Maximum drugs to display
        max_edges_per_type: Maximum edges per type to display
        output_name: Output filename
    """
    print("Creating Circos Hierarchical Edge Bundling visualization...")
    
    # Sample drugs if too many
    all_drugs = list(data['drugs'].keys())
    if len(all_drugs) > max_drugs:
        drugs = random.sample(all_drugs, max_drugs)
    else:
        drugs = all_drugs
    drug_set = set(drugs)
    
    # Build node list by type (hierarchical grouping)
    nodes = []
    node_types = []
    node_labels = []
    
    # Add drugs first (main nodes)
    for drug_id in drugs:
        drug = data['drugs'].get(drug_id)
        if drug:
            nodes.append(drug_id)
            node_types.append('drug')
            node_labels.append(drug.name[:15] if hasattr(drug, 'name') else drug_id[:15])
    
    n_drugs = len(nodes)
    
    # Sample other node types
    # Proteins connected to our drugs
    drug_proteins = set()
    for edge in data['drug_protein_edges'][:max_edges_per_type*2]:
        if edge.drug_id in drug_set:
            drug_proteins.add(edge.protein_id)
    
    sampled_proteins = list(drug_proteins)[:50]
    for prot_id in sampled_proteins:
        prot = data['proteins'].get(prot_id)
        if prot:
            nodes.append(prot_id)
            node_types.append('protein')
            node_labels.append(prot.name[:12] if hasattr(prot, 'name') else prot_id[:12])
    
    # Side effects
    drug_ses = set()
    for edge in data['drug_se_edges'][:max_edges_per_type*2]:
        if edge.drug_id in drug_set:
            drug_ses.add(edge.side_effect_id)
    
    sampled_ses = list(drug_ses)[:50]
    for se_id in sampled_ses:
        se = data['side_effects'].get(se_id)
        if se:
            nodes.append(se_id)
            node_types.append('side_effect')
            node_labels.append(se.name[:12] if hasattr(se, 'name') else se_id[:12])
    
    # Diseases
    drug_diseases = set()
    for edge in data['drug_disease_edges'][:max_edges_per_type*2]:
        if edge.drug_id in drug_set:
            drug_diseases.add(edge.disease_id)
    
    sampled_diseases = list(drug_diseases)[:30]
    for dis_id in sampled_diseases:
        dis = data['diseases'].get(dis_id)
        if dis:
            nodes.append(dis_id)
            node_types.append('disease')
            node_labels.append(dis.name[:12] if hasattr(dis, 'name') else dis_id[:12])
    
    n_nodes = len(nodes)
    print(f"  Total nodes: {n_nodes} ({n_drugs} drugs)")
    
    # Create node ID to index mapping
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # Calculate node positions on circle
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 1.0
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    # Collect edges
    edges = []
    edge_types = []
    edge_alphas = []
    
    # DDI edges (only between displayed drugs)
    ddi_count = 0
    for edge in data['ddi_edges']:
        if ddi_count >= max_edges_per_type:
            break
        if edge.drug1_id in node_idx and edge.drug2_id in node_idx:
            edges.append((node_idx[edge.drug1_id], node_idx[edge.drug2_id]))
            edge_types.append('ddi')
            # Color by severity
            severity = getattr(edge, 'severity', 'Moderate interaction')
            edge_alphas.append(0.3 if 'Moderate' in severity else 0.5)
            ddi_count += 1
    
    # Drug-protein edges
    dp_count = 0
    for edge in data['drug_protein_edges']:
        if dp_count >= max_edges_per_type:
            break
        if edge.drug_id in node_idx and edge.protein_id in node_idx:
            edges.append((node_idx[edge.drug_id], node_idx[edge.protein_id]))
            edge_types.append('drug_protein')
            edge_alphas.append(0.4)
            dp_count += 1
    
    # Drug-side effect edges
    dse_count = 0
    for edge in data['drug_se_edges']:
        if dse_count >= max_edges_per_type:
            break
        if edge.drug_id in node_idx and edge.side_effect_id in node_idx:
            edges.append((node_idx[edge.drug_id], node_idx[edge.side_effect_id]))
            edge_types.append('drug_side_effect')
            edge_alphas.append(0.3)
            dse_count += 1
    
    # Drug-disease edges
    dd_count = 0
    for edge in data['drug_disease_edges']:
        if dd_count >= max_edges_per_type:
            break
        if edge.drug_id in node_idx and edge.disease_id in node_idx:
            edges.append((node_idx[edge.drug_id], node_idx[edge.disease_id]))
            edge_types.append('drug_disease')
            edge_alphas.append(0.4)
            dd_count += 1
    
    print(f"  Total edges: {len(edges)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect('equal')
    
    # Draw hierarchical edge bundles
    print("  Drawing edge bundles...")
    for (i, j), etype, alpha in zip(edges, edge_types, edge_alphas):
        p0 = positions[i]
        p1 = positions[j]
        
        # Create bundled curve
        curve = hierarchical_edge_bundle(p0, p1, center=(0, 0), beta=0.85)
        
        color = EDGE_COLORS.get(etype, '#888888')
        ax.plot(curve[:, 0], curve[:, 1], color=color, alpha=alpha, 
               linewidth=0.3, solid_capstyle='round')
    
    # Draw nodes
    print("  Drawing nodes...")
    for i, (node, ntype, label) in enumerate(zip(nodes, node_types, node_labels)):
        x, y = positions[i]
        color = NODE_COLORS.get(ntype, '#888888')
        
        # Node size based on type
        size = 80 if ntype == 'drug' else 40
        
        ax.scatter(x, y, s=size, c=color, zorder=10, edgecolors='white', linewidths=0.5)
    
    # Draw labels for drugs only (outer ring)
    label_radius = 1.12
    for i in range(n_drugs):
        angle = angles[i]
        x = label_radius * np.cos(angle)
        y = label_radius * np.sin(angle)
        
        # Rotate label to follow circle
        rotation = np.degrees(angle)
        if 90 < rotation < 270:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        
        ax.text(x, y, node_labels[i], fontsize=5, rotation=rotation,
               ha=ha, va='center', rotation_mode='anchor')
    
    # Add arc segments for node type groups
    type_counts = defaultdict(int)
    for ntype in node_types:
        type_counts[ntype] += 1
    
    # Draw type indicator arcs
    arc_radius = 1.05
    arc_width = 0.02
    start_idx = 0
    
    for ntype in ['drug', 'protein', 'side_effect', 'disease']:
        count = type_counts.get(ntype, 0)
        if count == 0:
            continue
        
        start_angle = angles[start_idx] if start_idx < len(angles) else 0
        end_idx = min(start_idx + count - 1, len(angles) - 1)
        end_angle = angles[end_idx] if end_idx >= 0 else 0
        
        # Draw arc
        theta = np.linspace(start_angle, end_angle, 50)
        inner = arc_radius - arc_width/2
        outer = arc_radius + arc_width/2
        
        x_inner = inner * np.cos(theta)
        y_inner = inner * np.sin(theta)
        x_outer = outer * np.cos(theta)
        y_outer = outer * np.sin(theta)
        
        color = NODE_COLORS.get(ntype, '#888888')
        ax.fill(np.concatenate([x_inner, x_outer[::-1]]),
               np.concatenate([y_inner, y_outer[::-1]]),
               color=color, alpha=0.7)
        
        start_idx += count
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color=NODE_COLORS['drug'], label=f'Drugs ({type_counts["drug"]})'),
        mpatches.Patch(color=NODE_COLORS['protein'], label=f'Proteins ({type_counts["protein"]})'),
        mpatches.Patch(color=NODE_COLORS['side_effect'], label=f'Side Effects ({type_counts["side_effect"]})'),
        mpatches.Patch(color=NODE_COLORS['disease'], label=f'Diseases ({type_counts["disease"]})'),
        plt.Line2D([0], [0], color=EDGE_COLORS['ddi'], label='DDI'),
        plt.Line2D([0], [0], color=EDGE_COLORS['drug_protein'], label='Drug-Protein'),
        plt.Line2D([0], [0], color=EDGE_COLORS['drug_side_effect'], label='Drug-Side Effect'),
        plt.Line2D([0], [0], color=EDGE_COLORS['drug_disease'], label='Drug-Disease'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             framealpha=0.9, title='Legend')
    
    # Style
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title('DDI Knowledge Graph\nHierarchical Edge Bundling Visualization',
                fontsize=16, fontweight='bold', y=1.02)
    
    # Add statistics text
    stats_text = f"Nodes: {n_nodes} | Edges: {len(edges)}"
    ax.text(0, -1.35, stats_text, ha='center', fontsize=10, style='italic')
    
    # Save
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / f'{output_name}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Saved: {OUTPUT_DIR / output_name}.pdf/png")


def create_severity_circos(data, max_drugs=150, max_ddis=1000, output_name='circos_severity'):
    """
    Create Circos visualization focused on DDI severity.
    
    Shows only drug nodes with DDI edges colored by severity.
    """
    print("\nCreating DDI Severity Circos visualization...")
    
    # Get drugs involved in DDIs
    ddi_drugs = set()
    severity_counts = defaultdict(int)
    
    for edge in data['ddi_edges'][:max_ddis * 10]:
        ddi_drugs.add(edge.drug1_id)
        ddi_drugs.add(edge.drug2_id)
        severity = getattr(edge, 'severity', 'Moderate interaction')
        severity_counts[severity] += 1
    
    # Sample drugs
    drugs = list(ddi_drugs)[:max_drugs]
    drug_set = set(drugs)
    n_drugs = len(drugs)
    
    # Create node mapping
    node_idx = {drug: i for i, drug in enumerate(drugs)}
    
    # Get drug names
    drug_names = []
    for drug_id in drugs:
        drug = data['drugs'].get(drug_id)
        name = drug.name[:12] if drug and hasattr(drug, 'name') else drug_id[:12]
        drug_names.append(name)
    
    # Calculate positions
    angles = np.linspace(0, 2 * np.pi, n_drugs, endpoint=False)
    radius = 1.0
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    # Collect DDI edges with severity
    edges_by_severity = defaultdict(list)
    
    for edge in data['ddi_edges'][:max_ddis * 5]:
        if edge.drug1_id in drug_set and edge.drug2_id in drug_set:
            severity = getattr(edge, 'severity', 'Moderate interaction')
            edges_by_severity[severity].append(
                (node_idx[edge.drug1_id], node_idx[edge.drug2_id])
            )
    
    # Limit edges per severity
    for sev in edges_by_severity:
        if len(edges_by_severity[sev]) > max_ddis // 4:
            edges_by_severity[sev] = random.sample(edges_by_severity[sev], max_ddis // 4)
    
    total_edges = sum(len(e) for e in edges_by_severity.values())
    print(f"  Drugs: {n_drugs}, DDI edges: {total_edges}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect('equal')
    
    # Draw edges by severity (draw less severe first, more severe on top)
    severity_order = ['Minor interaction', 'Moderate interaction', 
                      'Major interaction', 'Contraindicated interaction']
    
    for severity in severity_order:
        edges = edges_by_severity.get(severity, [])
        if not edges:
            continue
        
        color = SEVERITY_COLORS.get(severity, '#888888')
        alpha = 0.2 if 'Moderate' in severity else 0.4 if 'Minor' in severity else 0.6
        lw = 0.3 if 'Moderate' in severity else 0.5 if 'Minor' in severity else 0.8
        
        for i, j in edges:
            p0 = positions[i]
            p1 = positions[j]
            curve = hierarchical_edge_bundle(p0, p1, center=(0, 0), beta=0.85)
            ax.plot(curve[:, 0], curve[:, 1], color=color, alpha=alpha,
                   linewidth=lw, solid_capstyle='round')
    
    # Draw drug nodes
    for i, (drug_id, name) in enumerate(zip(drugs, drug_names)):
        x, y = positions[i]
        ax.scatter(x, y, s=60, c=NODE_COLORS['drug'], zorder=10,
                  edgecolors='white', linewidths=0.5)
    
    # Draw labels
    label_radius = 1.08
    for i, name in enumerate(drug_names):
        angle = angles[i]
        x = label_radius * np.cos(angle)
        y = label_radius * np.sin(angle)
        
        rotation = np.degrees(angle)
        if 90 < rotation < 270:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        
        ax.text(x, y, name, fontsize=5, rotation=rotation,
               ha=ha, va='center', rotation_mode='anchor')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=SEVERITY_COLORS['Contraindicated interaction'],
                  linewidth=3, label=f'Contraindicated ({len(edges_by_severity.get("Contraindicated interaction", []))})'),
        plt.Line2D([0], [0], color=SEVERITY_COLORS['Major interaction'],
                  linewidth=3, label=f'Major ({len(edges_by_severity.get("Major interaction", []))})'),
        plt.Line2D([0], [0], color=SEVERITY_COLORS['Moderate interaction'],
                  linewidth=2, label=f'Moderate ({len(edges_by_severity.get("Moderate interaction", []))})'),
        plt.Line2D([0], [0], color=SEVERITY_COLORS['Minor interaction'],
                  linewidth=2, label=f'Minor ({len(edges_by_severity.get("Minor interaction", []))})'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             framealpha=0.9, title='DDI Severity')
    
    # Style
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis('off')
    ax.set_title('Drug-Drug Interaction Network\nSeverity-Based Hierarchical Edge Bundling',
                fontsize=16, fontweight='bold', y=1.02)
    
    # Statistics
    stats_text = f"Drugs: {n_drugs} | DDIs: {total_edges}"
    ax.text(0, -1.25, stats_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / f'{output_name}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Saved: {OUTPUT_DIR / output_name}.pdf/png")


def create_drug_class_circos(data, max_drugs=200, output_name='circos_drug_classes'):
    """
    Create Circos visualization with drugs grouped by category/class.
    """
    print("\nCreating Drug Class Circos visualization...")
    
    # Group drugs by category
    drug_categories = defaultdict(list)
    category_colors = {}
    color_palette = plt.cm.tab20.colors
    
    # Get drug-category mappings
    for edge in data['drug_category_edges']:
        drug_id = edge.drug_id
        cat_name = edge.category_name
        if drug_id in data['drugs']:
            drug_categories[cat_name].append(drug_id)
    
    # Select top categories with most drugs
    top_categories = sorted(drug_categories.keys(), 
                           key=lambda x: len(drug_categories[x]), 
                           reverse=True)[:10]
    
    # Assign colors
    for i, cat in enumerate(top_categories):
        category_colors[cat] = color_palette[i % len(color_palette)]
    
    # Build ordered node list (grouped by category)
    nodes = []
    node_categories = []
    node_names = []
    
    for cat in top_categories:
        cat_drugs = drug_categories[cat][:max_drugs // len(top_categories)]
        for drug_id in cat_drugs:
            if drug_id not in nodes:
                nodes.append(drug_id)
                node_categories.append(cat)
                drug = data['drugs'].get(drug_id)
                name = drug.name[:10] if drug and hasattr(drug, 'name') else drug_id[:10]
                node_names.append(name)
    
    n_nodes = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    drug_set = set(nodes)
    
    print(f"  Nodes: {n_nodes} across {len(top_categories)} categories")
    
    # Positions
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 1.0
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    # DDI edges between displayed drugs
    edges = []
    for edge in data['ddi_edges']:
        if edge.drug1_id in drug_set and edge.drug2_id in drug_set:
            edges.append((node_idx[edge.drug1_id], node_idx[edge.drug2_id]))
            if len(edges) >= 2000:
                break
    
    print(f"  Edges: {len(edges)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.set_aspect('equal')
    
    # Draw edges
    for i, j in edges:
        p0 = positions[i]
        p1 = positions[j]
        curve = hierarchical_edge_bundle(p0, p1, center=(0, 0), beta=0.85)
        ax.plot(curve[:, 0], curve[:, 1], color='#888888', alpha=0.15,
               linewidth=0.3, solid_capstyle='round')
    
    # Draw category arcs
    arc_radius = 1.06
    arc_width = 0.04
    
    start_idx = 0
    for cat in top_categories:
        count = sum(1 for c in node_categories if c == cat)
        if count == 0:
            continue
        
        end_idx = start_idx + count - 1
        if end_idx >= len(angles):
            end_idx = len(angles) - 1
        
        start_angle = angles[start_idx]
        end_angle = angles[end_idx]
        
        theta = np.linspace(start_angle, end_angle, 50)
        inner = arc_radius - arc_width/2
        outer = arc_radius + arc_width/2
        
        x_inner = inner * np.cos(theta)
        y_inner = inner * np.sin(theta)
        x_outer = outer * np.cos(theta)
        y_outer = outer * np.sin(theta)
        
        color = category_colors.get(cat, '#888888')
        ax.fill(np.concatenate([x_inner, x_outer[::-1]]),
               np.concatenate([y_inner, y_outer[::-1]]),
               color=color, alpha=0.8)
        
        # Category label
        mid_angle = (start_angle + end_angle) / 2
        label_r = 1.15
        lx = label_r * np.cos(mid_angle)
        ly = label_r * np.sin(mid_angle)
        
        rotation = np.degrees(mid_angle)
        if 90 < rotation < 270:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        
        # Truncate category name
        short_cat = cat[:20] + '...' if len(cat) > 20 else cat
        ax.text(lx, ly, short_cat, fontsize=7, rotation=rotation,
               ha=ha, va='center', rotation_mode='anchor', fontweight='bold')
        
        start_idx = end_idx + 1
    
    # Draw nodes
    for i, (node, cat) in enumerate(zip(nodes, node_categories)):
        x, y = positions[i]
        color = category_colors.get(cat, NODE_COLORS['drug'])
        ax.scatter(x, y, s=30, c=[color], zorder=10, edgecolors='white', linewidths=0.3)
    
    # Legend
    legend_elements = [mpatches.Patch(color=category_colors[cat], label=cat[:25])
                      for cat in top_categories]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
             framealpha=0.9, title='Drug Categories', ncol=1)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title('DDI Knowledge Graph by Drug Category\nHierarchical Edge Bundling',
                fontsize=16, fontweight='bold', y=1.02)
    
    stats_text = f"Drugs: {n_nodes} | Categories: {len(top_categories)} | DDIs: {len(edges)}"
    ax.text(0, -1.35, stats_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / f'{output_name}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Saved: {OUTPUT_DIR / output_name}.pdf/png")


def main():
    """Generate all Circos visualizations."""
    print("=" * 60)
    print("Generating Circos Hierarchical Edge Bundling Visualizations")
    print("=" * 60)
    
    # Load data
    print("\nLoading knowledge graph data...")
    data = load_kg_data()
    print(f"  Drugs: {len(data['drugs'])}")
    print(f"  DDI edges: {len(data['ddi_edges'])}")
    
    # Generate visualizations
    create_circos_plot(data, max_drugs=100, max_edges_per_type=400)
    create_severity_circos(data, max_drugs=150, max_ddis=800)
    create_drug_class_circos(data, max_drugs=200)
    
    print("\n" + "=" * 60)
    print("All Circos visualizations generated!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
