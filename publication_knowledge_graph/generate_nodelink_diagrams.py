#!/usr/bin/env python3
"""
Generate Node-Link Diagrams for DDI Knowledge Graph
- Full network overview with community detection
- Drug-centric ego network
- Multi-entity type subgraph
- Severity-based filtered network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
from collections import defaultdict
import ast
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = '/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/publication_knowledge_graph/figures'

def load_ddi_data():
    """Load the recalibrated DDI data"""
    csv_path = '/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/data/ddi_recalibrated.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} DDI pairs")
    return df

def get_drug_classes(df):
    """Extract drug classes from ATC codes"""
    atc_classes = {
        'A': 'Alimentary',
        'B': 'Blood',
        'C': 'Cardiovascular',
        'D': 'Dermatologicals',
        'G': 'Genitourinary',
        'H': 'Hormones',
        'J': 'Anti-infectives',
        'L': 'Antineoplastic',
        'M': 'Musculoskeletal',
        'N': 'Nervous System',
        'P': 'Antiparasitic',
        'R': 'Respiratory',
        'S': 'Sensory Organs',
        'V': 'Various'
    }
    
    drug_to_class = {}
    drug_to_name = {}
    
    for suffix in ['1', '2']:
        atc_col = f'atc_{suffix}'
        drug_col = f'drugbank_id_{suffix}'
        name_col = f'drug_name_{suffix}'
        
        if atc_col in df.columns and drug_col in df.columns:
            for _, row in df.iterrows():
                atc_str = row.get(atc_col, '')
                drug_id = row.get(drug_col, '')
                drug_name = row.get(name_col, drug_id)
                
                drug_to_name[drug_id] = drug_name
                
                if pd.notna(atc_str) and isinstance(atc_str, str) and len(atc_str) > 2:
                    try:
                        atc_list = ast.literal_eval(atc_str)
                        if isinstance(atc_list, list) and len(atc_list) > 0:
                            atc = atc_list[0]
                            if len(atc) > 0:
                                first_letter = atc[0].upper()
                                if first_letter in atc_classes:
                                    drug_to_class[drug_id] = atc_classes[first_letter]
                    except (ValueError, SyntaxError):
                        pass
    
    return drug_to_class, drug_to_name, atc_classes


def generate_network_overview(df, output_path):
    """Generate a network overview showing top interacting drugs"""
    print("\n=== Generating Network Overview Diagram ===")
    
    drug_to_class, drug_to_name, atc_classes = get_drug_classes(df)
    
    # Count interactions per drug
    drug_interactions = defaultdict(int)
    for _, row in df.iterrows():
        drug_interactions[row['drugbank_id_1']] += 1
        drug_interactions[row['drugbank_id_2']] += 1
    
    # Get top 50 most connected drugs
    top_drugs = sorted(drug_interactions.keys(), 
                       key=lambda x: drug_interactions[x], 
                       reverse=True)[:50]
    top_drugs_set = set(top_drugs)
    
    # Build subgraph with these drugs
    G = nx.Graph()
    
    # Add nodes
    for drug in top_drugs:
        G.add_node(drug, 
                   name=drug_to_name.get(drug, drug),
                   drug_class=drug_to_class.get(drug, 'Unknown'),
                   degree=drug_interactions[drug])
    
    # Add edges (only between top drugs)
    edge_count = 0
    severity_col = 'severity_recalibrated'
    
    for _, row in df.iterrows():
        d1, d2 = row['drugbank_id_1'], row['drugbank_id_2']
        if d1 in top_drugs_set and d2 in top_drugs_set:
            severity = row.get(severity_col, 'Moderate')
            G.add_edge(d1, d2, severity=severity)
            edge_count += 1
    
    print(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    # Color by drug class
    class_colors = {
        'Alimentary': '#1f77b4',
        'Blood': '#d62728',
        'Cardiovascular': '#ff7f0e',
        'Dermatologicals': '#9467bd',
        'Genitourinary': '#8c564b',
        'Hormones': '#e377c2',
        'Anti-infectives': '#7f7f7f',
        'Antineoplastic': '#bcbd22',
        'Musculoskeletal': '#17becf',
        'Nervous System': '#2ca02c',
        'Antiparasitic': '#aec7e8',
        'Respiratory': '#ffbb78',
        'Sensory Organs': '#98df8a',
        'Various': '#ff9896',
        'Unknown': '#c7c7c7'
    }
    
    node_colors = [class_colors.get(G.nodes[n].get('drug_class', 'Unknown'), '#c7c7c7') 
                   for n in G.nodes()]
    
    # Node sizes based on degree
    node_sizes = [300 + G.degree(n) * 50 for n in G.nodes()]
    
    # Edge colors based on severity
    severity_colors = {
        'Contraindicated interaction': '#d62728',
        'Major interaction': '#ff7f0e',
        'Moderate interaction': '#2ca02c',
        'Minor interaction': '#1f77b4'
    }
    
    edge_colors = [severity_colors.get(G.edges[e].get('severity', 'Moderate interaction'), '#cccccc') 
                   for e in G.edges()]
    edge_alphas = [0.8 if 'Contraindicated' in str(G.edges[e].get('severity', '')) or 
                   'Major' in str(G.edges[e].get('severity', '')) else 0.3 
                   for e in G.edges()]
    
    # Draw edges
    for i, (u, v) in enumerate(G.edges()):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=edge_colors[i], alpha=edge_alphas[i], linewidth=1, zorder=1)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.9, edgecolors='black', 
                           linewidths=1)
    
    # Add labels for top nodes
    top_10 = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:15]
    labels = {n: drug_to_name.get(n, n)[:12] for n in top_10}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Legend for drug classes
    legend_elements = []
    classes_in_graph = set(G.nodes[n].get('drug_class', 'Unknown') for n in G.nodes())
    for cls in sorted(classes_in_graph):
        if cls in class_colors:
            legend_elements.append(mpatches.Patch(facecolor=class_colors[cls], 
                                                   label=cls, edgecolor='black'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
              title='Drug Classes', title_fontsize=10, framealpha=0.9)
    
    # Severity legend
    sev_legend = [Line2D([0], [0], color=severity_colors[s], linewidth=3, label=s.replace(' interaction', ''))
                  for s in ['Contraindicated interaction', 'Major interaction', 
                           'Moderate interaction', 'Minor interaction']]
    ax.legend(handles=sev_legend, loc='upper right', fontsize=9,
              title='DDI Severity', title_fontsize=10, framealpha=0.9)
    
    ax.set_title('DDI Knowledge Graph: Top 50 Most Connected Drugs\n(Node-Link Diagram)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved network overview to {output_path}")


def generate_ego_network(df, output_path, center_drug='DB00945'):
    """Generate ego network centered on a specific drug (default: Aspirin)"""
    print("\n=== Generating Ego Network Diagram ===")
    
    drug_to_class, drug_to_name, atc_classes = get_drug_classes(df)
    
    # Find the drug name
    center_name = drug_to_name.get(center_drug, center_drug)
    print(f"Center drug: {center_name} ({center_drug})")
    
    # Get all drugs that interact with center drug
    neighbors = set()
    edges_data = []
    severity_col = 'severity_recalibrated'
    
    for _, row in df.iterrows():
        d1, d2 = row['drugbank_id_1'], row['drugbank_id_2']
        if d1 == center_drug:
            neighbors.add(d2)
            edges_data.append((d1, d2, row.get(severity_col, 'Moderate')))
        elif d2 == center_drug:
            neighbors.add(d1)
            edges_data.append((d1, d2, row.get(severity_col, 'Moderate')))
    
    print(f"Found {len(neighbors)} interacting drugs")
    
    # Limit to top 60 neighbors by severity (prioritize severe)
    severity_order = {'Contraindicated interaction': 4, 'Major interaction': 3, 
                      'Moderate interaction': 2, 'Minor interaction': 1}
    
    neighbor_severity = {}
    for d1, d2, sev in edges_data:
        other = d2 if d1 == center_drug else d1
        score = severity_order.get(sev, 2)
        if other not in neighbor_severity or score > neighbor_severity[other]:
            neighbor_severity[other] = score
    
    top_neighbors = sorted(neighbors, key=lambda x: neighbor_severity.get(x, 0), reverse=True)[:60]
    top_neighbors_set = set(top_neighbors)
    
    # Build ego graph
    G = nx.Graph()
    
    # Add center node
    G.add_node(center_drug, name=center_name, drug_class=drug_to_class.get(center_drug, 'Unknown'), 
               is_center=True)
    
    # Add neighbor nodes
    for drug in top_neighbors:
        G.add_node(drug, name=drug_to_name.get(drug, drug),
                   drug_class=drug_to_class.get(drug, 'Unknown'), is_center=False)
    
    # Add edges
    for d1, d2, sev in edges_data:
        other = d2 if d1 == center_drug else d1
        if other in top_neighbors_set:
            G.add_edge(center_drug, other, severity=sev)
    
    print(f"Ego network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Radial layout - center drug in middle
    pos = {}
    pos[center_drug] = (0, 0)
    
    # Group neighbors by severity for radial placement
    severity_groups = defaultdict(list)
    for n in G.neighbors(center_drug):
        sev = G.edges[center_drug, n].get('severity', 'Moderate')
        severity_groups[sev].append(n)
    
    # Place nodes in concentric rings by severity
    ring_radii = {
        'Contraindicated interaction': 1.0,
        'Major interaction': 1.0,
        'Moderate interaction': 1.5,
        'Minor interaction': 2.0
    }
    
    current_angles = {sev: 0 for sev in ring_radii}
    nodes_per_severity = {sev: len(severity_groups[sev]) for sev in ring_radii}
    
    for sev, nodes in severity_groups.items():
        n_nodes = len(nodes)
        if n_nodes == 0:
            continue
        radius = ring_radii.get(sev, 1.5)
        angle_step = 2 * np.pi / max(n_nodes, 1)
        start_angle = current_angles[sev]
        
        for i, node in enumerate(nodes):
            angle = start_angle + i * angle_step
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
    
    # Colors
    class_colors = {
        'Blood': '#d62728',
        'Cardiovascular': '#ff7f0e',
        'Nervous System': '#2ca02c',
        'Anti-infectives': '#7f7f7f',
        'Antineoplastic': '#bcbd22',
        'Alimentary': '#1f77b4',
        'Hormones': '#e377c2',
        'Musculoskeletal': '#17becf',
        'Unknown': '#c7c7c7'
    }
    
    severity_colors = {
        'Contraindicated interaction': '#d62728',
        'Major interaction': '#ff7f0e',
        'Moderate interaction': '#2ca02c',
        'Minor interaction': '#1f77b4'
    }
    
    # Draw edges
    for u, v in G.edges():
        sev = G.edges[u, v].get('severity', 'Moderate')
        color = severity_colors.get(sev, '#cccccc')
        width = 3 if 'Contraindicated' in sev or 'Major' in sev else 1
        alpha = 0.8 if 'Contraindicated' in sev or 'Major' in sev else 0.4
        
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=color, alpha=alpha, linewidth=width, zorder=1)
    
    # Draw neighbor nodes
    for node in G.nodes():
        if node == center_drug:
            continue
        drug_class = G.nodes[node].get('drug_class', 'Unknown')
        color = class_colors.get(drug_class, '#c7c7c7')
        ax.scatter(pos[node][0], pos[node][1], s=400, c=color, 
                   edgecolors='black', linewidths=1, zorder=2, alpha=0.9)
    
    # Draw center node (larger, highlighted)
    ax.scatter(pos[center_drug][0], pos[center_drug][1], s=2000, c='gold',
               edgecolors='black', linewidths=3, zorder=3, marker='*')
    ax.annotate(center_name, pos[center_drug], fontsize=12, fontweight='bold',
                ha='center', va='bottom', xytext=(0, 30), textcoords='offset points')
    
    # Add some labels for severe interactions
    severe_neighbors = [n for n in G.neighbors(center_drug) 
                        if 'Contraindicated' in str(G.edges[center_drug, n].get('severity', '')) or
                        'Major' in str(G.edges[center_drug, n].get('severity', ''))][:15]
    
    for node in severe_neighbors:
        name = G.nodes[node].get('name', node)[:15]
        ax.annotate(name, pos[node], fontsize=7, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')
    
    # Legend
    legend_elements = []
    for sev in ['Contraindicated interaction', 'Major interaction', 'Moderate interaction', 'Minor interaction']:
        legend_elements.append(Line2D([0], [0], color=severity_colors[sev], linewidth=3, 
                                      label=sev.replace(' interaction', '')))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              title='DDI Severity', title_fontsize=11, framealpha=0.9)
    
    # Ring labels
    for sev, radius in ring_radii.items():
        if sev in severity_groups and len(severity_groups[sev]) > 0:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, linestyle='--', 
                                    color='gray', alpha=0.3))
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title(f'Ego Network: Drug Interactions with {center_name}\n({len(G.edges())} interactions shown)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved ego network to {output_path}")


def generate_severity_network(df, output_path):
    """Generate network showing only severe (Contraindicated + Major) interactions"""
    print("\n=== Generating Severe DDI Network ===")
    
    drug_to_class, drug_to_name, atc_classes = get_drug_classes(df)
    
    # Filter for severe interactions
    severity_col = 'severity_recalibrated'
    severe_df = df[df[severity_col].str.contains('Contraindicated|Major', na=False)]
    print(f"Found {len(severe_df):,} severe interactions")
    
    # Count involvement in severe DDIs
    drug_severe_count = defaultdict(int)
    for _, row in severe_df.iterrows():
        drug_severe_count[row['drugbank_id_1']] += 1
        drug_severe_count[row['drugbank_id_2']] += 1
    
    # Get top 40 drugs with most severe interactions
    top_drugs = sorted(drug_severe_count.keys(), 
                       key=lambda x: drug_severe_count[x], 
                       reverse=True)[:40]
    top_drugs_set = set(top_drugs)
    
    # Build graph
    G = nx.Graph()
    
    for drug in top_drugs:
        G.add_node(drug, name=drug_to_name.get(drug, drug),
                   drug_class=drug_to_class.get(drug, 'Unknown'),
                   severe_count=drug_severe_count[drug])
    
    # Add edges
    for _, row in severe_df.iterrows():
        d1, d2 = row['drugbank_id_1'], row['drugbank_id_2']
        if d1 in top_drugs_set and d2 in top_drugs_set:
            sev = row.get(severity_col, 'Major')
            G.add_edge(d1, d2, severity=sev)
    
    print(f"Severe network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Layout with more spread
    pos = nx.kamada_kawai_layout(G)
    
    # Colors
    class_colors = {
        'Blood': '#d62728',
        'Cardiovascular': '#ff7f0e',
        'Nervous System': '#2ca02c',
        'Anti-infectives': '#7f7f7f',
        'Antineoplastic': '#bcbd22',
        'Alimentary': '#1f77b4',
        'Hormones': '#e377c2',
        'Musculoskeletal': '#17becf',
        'Respiratory': '#ffbb78',
        'Unknown': '#c7c7c7'
    }
    
    node_colors = [class_colors.get(G.nodes[n].get('drug_class', 'Unknown'), '#c7c7c7') 
                   for n in G.nodes()]
    
    # Node sizes based on severe interaction count
    node_sizes = [200 + G.nodes[n].get('severe_count', 1) * 5 for n in G.nodes()]
    
    # Edge colors
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        sev = G.edges[u, v].get('severity', '')
        if 'Contraindicated' in sev:
            edge_colors.append('#d62728')
            edge_widths.append(2.5)
        else:
            edge_colors.append('#ff7f0e')
            edge_widths.append(1.5)
    
    # Draw edges
    for i, (u, v) in enumerate(G.edges()):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=edge_colors[i], alpha=0.6, linewidth=edge_widths[i], zorder=1)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, edgecolors='black',
                           linewidths=1.5)
    
    # Labels for all nodes
    labels = {n: drug_to_name.get(n, n)[:10] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)
    
    # Legend
    legend_elements = []
    classes_in_graph = set(G.nodes[n].get('drug_class', 'Unknown') for n in G.nodes())
    for cls in sorted(classes_in_graph):
        if cls in class_colors:
            legend_elements.append(mpatches.Patch(facecolor=class_colors[cls],
                                                   label=cls, edgecolor='black'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              title='Drug Classes', title_fontsize=10, framealpha=0.9)
    
    # Edge legend
    edge_legend = [
        Line2D([0], [0], color='#d62728', linewidth=3, label='Contraindicated'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='Major')
    ]
    ax2 = ax.twinx()
    ax2.legend(handles=edge_legend, loc='upper right', fontsize=10,
               title='Severity', title_fontsize=11, framealpha=0.9)
    ax2.axis('off')
    
    ax.set_title('Severe DDI Network\n(Contraindicated and Major Interactions Only)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved severe DDI network to {output_path}")


def generate_class_network(df, output_path):
    """Generate aggregated network showing interactions between drug classes"""
    print("\n=== Generating Drug Class Network ===")
    
    drug_to_class, drug_to_name, atc_classes = get_drug_classes(df)
    
    # Count interactions and severities between classes
    class_interactions = defaultdict(int)
    class_severe = defaultdict(int)
    severity_col = 'severity_recalibrated'
    
    for _, row in df.iterrows():
        d1, d2 = row['drugbank_id_1'], row['drugbank_id_2']
        c1 = drug_to_class.get(d1, 'Unknown')
        c2 = drug_to_class.get(d2, 'Unknown')
        
        if c1 != 'Unknown' and c2 != 'Unknown':
            key = tuple(sorted([c1, c2]))
            class_interactions[key] += 1
            
            sev = row.get(severity_col, '')
            if 'Contraindicated' in str(sev) or 'Major' in str(sev):
                class_severe[key] += 1
    
    # Build class-level graph
    G = nx.Graph()
    
    # Add nodes for each class
    class_drug_counts = defaultdict(int)
    for drug, cls in drug_to_class.items():
        class_drug_counts[cls] += 1
    
    for cls, count in class_drug_counts.items():
        if count >= 10:  # Only classes with enough drugs
            G.add_node(cls, drug_count=count)
    
    # Add edges
    for (c1, c2), count in class_interactions.items():
        if c1 in G.nodes() and c2 in G.nodes() and count >= 100:
            severe_ratio = class_severe[(c1, c2)] / count if count > 0 else 0
            G.add_edge(c1, c2, weight=count, severe_ratio=severe_ratio)
    
    print(f"Class network: {len(G.nodes())} classes, {len(G.edges())} connections")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Circular layout
    pos = nx.circular_layout(G)
    
    # Node sizes based on drug count
    node_sizes = [500 + G.nodes[n].get('drug_count', 1) * 3 for n in G.nodes()]
    
    # Class colors
    class_colors = {
        'Alimentary': '#1f77b4',
        'Blood': '#d62728',
        'Cardiovascular': '#ff7f0e',
        'Dermatologicals': '#9467bd',
        'Genitourinary': '#8c564b',
        'Hormones': '#e377c2',
        'Anti-infectives': '#7f7f7f',
        'Antineoplastic': '#bcbd22',
        'Musculoskeletal': '#17becf',
        'Nervous System': '#2ca02c',
        'Antiparasitic': '#aec7e8',
        'Respiratory': '#ffbb78',
        'Sensory Organs': '#98df8a',
        'Various': '#ff9896'
    }
    
    node_colors = [class_colors.get(n, '#c7c7c7') for n in G.nodes()]
    
    # Draw edges with width based on interaction count
    max_weight = max(G.edges[e].get('weight', 1) for e in G.edges()) if G.edges() else 1
    
    for u, v in G.edges():
        weight = G.edges[u, v].get('weight', 1)
        severe_ratio = G.edges[u, v].get('severe_ratio', 0)
        
        width = 1 + (weight / max_weight) * 10
        # Color based on severe ratio
        if severe_ratio > 0.15:
            color = '#d62728'
            alpha = 0.8
        elif severe_ratio > 0.10:
            color = '#ff7f0e'
            alpha = 0.7
        else:
            color = '#2ca02c'
            alpha = 0.5
        
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=color, alpha=alpha, linewidth=width, zorder=1)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, edgecolors='black',
                           linewidths=2)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Legend for edge severity
    edge_legend = [
        Line2D([0], [0], color='#d62728', linewidth=4, label='>15% Severe'),
        Line2D([0], [0], color='#ff7f0e', linewidth=3, label='10-15% Severe'),
        Line2D([0], [0], color='#2ca02c', linewidth=2, label='<10% Severe')
    ]
    ax.legend(handles=edge_legend, loc='upper right', fontsize=10,
              title='Severe DDI Rate', title_fontsize=11, framealpha=0.9)
    
    ax.set_title('Drug Class Interaction Network\n(Edge width = interaction count, color = severity rate)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved class network to {output_path}")


def main():
    """Generate all node-link diagrams"""
    print("=" * 60)
    print("Generating Node-Link Diagrams for DDI Knowledge Graph")
    print("=" * 60)
    
    # Load data
    df = load_ddi_data()
    
    # Generate visualizations
    generate_network_overview(df, f'{OUTPUT_DIR}/nodelink_network_overview.png')
    generate_ego_network(df, f'{OUTPUT_DIR}/nodelink_ego_network.png')
    generate_severity_network(df, f'{OUTPUT_DIR}/nodelink_severe_network.png')
    generate_class_network(df, f'{OUTPUT_DIR}/nodelink_class_network.png')
    
    print("\n" + "=" * 60)
    print("All node-link diagrams generated successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR}/nodelink_network_overview.png/pdf")
    print(f"  - {OUTPUT_DIR}/nodelink_ego_network.png/pdf")
    print(f"  - {OUTPUT_DIR}/nodelink_severe_network.png/pdf")
    print(f"  - {OUTPUT_DIR}/nodelink_class_network.png/pdf")


if __name__ == '__main__':
    main()
