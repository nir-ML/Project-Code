#!/usr/bin/env python3
"""
Generate only:
1. Chord diagram with severity encoding (compact, 1200 DPI)
2. Severe DDI network (compact, 1200 DPI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict
import ast
import os

OUTPUT_DIR = '/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/publication_knowledge_graph/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading DDI data...")
ddi_df = pd.read_csv('/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/data/ddi_semantic_recalibrated_final.csv')

ddi_df = ddi_df.rename(columns={
    'drug_name_1': 'drug_1',
    'drug_name_2': 'drug_2',
    'severity_final': 'severity'
})
ddi_df['severity'] = ddi_df['severity'].str.replace(' interaction', '', regex=False)

# ATC class mapping
ATC_CLASSES = {
    'A': 'Alimentary',
    'B': 'Blood',
    'C': 'Cardiovascular',
    'D': 'Dermatological',
    'G': 'Genitourinary',
    'H': 'Hormones',
    'J': 'Anti-infectives',
    'L': 'Antineoplastic',
    'M': 'Musculoskeletal',
    'N': 'Nervous System',
    'P': 'Antiparasitic',
    'R': 'Respiratory',
    'S': 'Sensory',
    'V': 'Various'
}

ATC_COLORS = {
    'A': '#e41a1c', 'B': '#377eb8', 'C': '#4daf4a', 'D': '#984ea3',
    'G': '#ff7f00', 'H': '#ffff33', 'J': '#a65628', 'L': '#f781bf',
    'M': '#999999', 'N': '#66c2a5', 'P': '#fc8d62', 'R': '#8da0cb',
    'S': '#e78ac3', 'V': '#a6d854'
}

def get_atc_class(atc_code):
    if pd.isna(atc_code) or atc_code == '' or atc_code == '[]':
        return None
    if isinstance(atc_code, str):
        if atc_code.startswith('['):
            try:
                codes = ast.literal_eval(atc_code)
                if codes and len(codes) > 0:
                    return codes[0][0]
            except:
                pass
        else:
            return atc_code[0] if len(atc_code) > 0 else None
    return None

# Map drugs to classes
ddi_df['class_1'] = ddi_df['atc_1'].apply(get_atc_class)
ddi_df['class_2'] = ddi_df['atc_2'].apply(get_atc_class)

drug_to_class = {}
for _, row in ddi_df.iterrows():
    if pd.notna(row['class_1']):
        drug_to_class[row['drug_1']] = row['class_1']
    if pd.notna(row['class_2']):
        drug_to_class[row['drug_2']] = row['class_2']

# Build severe pairs
severe_pairs = set()
severity_map = {}
for _, row in ddi_df.iterrows():
    d1, d2 = row['drug_1'], row['drug_2']
    pair = tuple(sorted([d1, d2]))
    severity_map[pair] = row['severity']
    if row['severity'] in ['Major', 'Contraindicated']:
        severe_pairs.add(pair)

print(f"Severe DDI pairs: {len(severe_pairs)}")


def generate_chord_severity_compact():
    """Generate compact chord diagram with severity encoding"""
    print("\n=== Generating Chord Diagram (Compact, 1200 DPI) ===")
    
    # Filter to classified DDIs
    classified_df = ddi_df.dropna(subset=['class_1', 'class_2'])
    
    # Build matrices
    classes = sorted(set(classified_df['class_1'].unique()) | set(classified_df['class_2'].unique()))
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    interaction_matrix = np.zeros((n, n))
    severe_matrix = np.zeros((n, n))
    
    for _, row in classified_df.iterrows():
        i, j = class_to_idx[row['class_1']], class_to_idx[row['class_2']]
        interaction_matrix[i, j] += 1
        interaction_matrix[j, i] += 1
        if row['severity'] in ['Major', 'Contraindicated']:
            severe_matrix[i, j] += 1
            severe_matrix[j, i] += 1
    
    severity_rate = np.divide(severe_matrix, interaction_matrix, 
                               out=np.zeros_like(severe_matrix), 
                               where=interaction_matrix > 0)
    
    # Compact figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Calculate arc sizes
    totals = interaction_matrix.sum(axis=1)
    grand_total = totals.sum()
    
    gap = 0.03
    total_gap = gap * n
    available = 2 * np.pi - total_gap
    
    arc_sizes = (totals / grand_total) * available
    
    starts = np.zeros(n)
    ends = np.zeros(n)
    current = 0
    for i in range(n):
        starts[i] = current
        ends[i] = current + arc_sizes[i]
        current = ends[i] + gap
    
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    
    # Draw outer arcs
    for i in range(n):
        theta = np.linspace(starts[i], ends[i], 50)
        r_inner = 0.88
        r_outer = 0.98
        
        ax.fill_between(theta, r_inner, r_outer, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Compact label
        mid_theta = (starts[i] + ends[i]) / 2
        label = classes[i]  # Just letter
        rotation = np.degrees(mid_theta)
        if np.pi/2 < mid_theta < 3*np.pi/2:
            rotation += 180
        ax.text(mid_theta, 1.08, label, ha='center', va='center',
                fontsize=10, fontweight='bold', rotation=rotation - 90)
    
    # Draw chords with severity colors
    for i in range(n):
        for j in range(i, n):
            if interaction_matrix[i, j] > 0:
                weight = interaction_matrix[i, j] / grand_total
                if weight < 0.002:
                    continue
                
                sev_rate = severity_rate[i, j]
                if sev_rate > 0.5:
                    chord_color = '#d62728'
                    alpha = 0.8
                elif sev_rate > 0.3:
                    chord_color = '#ff7f0e'
                    alpha = 0.7
                elif sev_rate > 0.15:
                    chord_color = '#fdae61'
                    alpha = 0.6
                elif sev_rate > 0.08:
                    chord_color = '#91cf60'
                    alpha = 0.5
                else:
                    chord_color = '#2ca02c'
                    alpha = 0.4
                
                arc_i = (starts[i] + ends[i]) / 2
                arc_j = (starts[j] + ends[j]) / 2
                
                t = np.linspace(0, 1, 50)
                theta = arc_i * (1-t)**2 + ((arc_i + arc_j)/2) * 2*t*(1-t) + arc_j * t**2
                r = 0.82 * np.sin(np.pi * t)
                
                linewidth = max(0.8, weight * 80)
                ax.plot(theta, r, color=chord_color, alpha=alpha, linewidth=linewidth)
    
    # Compact legend
    legend_elements = [
        mpatches.Patch(color='#d62728', label='>50%'),
        mpatches.Patch(color='#ff7f0e', label='30-50%'),
        mpatches.Patch(color='#fdae61', label='15-30%'),
        mpatches.Patch(color='#91cf60', label='8-15%'),
        mpatches.Patch(color='#2ca02c', label='<8%'),
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=8, 
              title='Severity', title_fontsize=9, framealpha=0.95,
              handlelength=1, handleheight=1)
    
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    
    plt.title('Drug Class DDI Severity', fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/chord_severity_compact.{fmt}', dpi=1200, 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: chord_severity_compact.png/pdf")


def generate_severe_network_compact():
    """Generate compact severe DDI network"""
    print("\n=== Generating Severe DDI Network (Compact, 1200 DPI) ===")
    
    # Count severe interactions per drug
    drug_severe_count = defaultdict(int)
    for d1, d2 in severe_pairs:
        drug_severe_count[d1] += 1
        drug_severe_count[d2] += 1
    
    # Top 25 drugs
    top_drugs = sorted(drug_severe_count.items(), key=lambda x: -x[1])[:25]
    top_drug_names = [d[0] for d in top_drugs]
    
    print("Top 10 drugs by severe DDI:")
    for d, c in top_drugs[:10]:
        cls = drug_to_class.get(d, '?')
        print(f"  {d}: {c} ({cls})")
    
    # Build network
    G = nx.Graph()
    
    for drug in top_drug_names:
        cls = drug_to_class.get(drug, 'V')
        G.add_node(drug, drug_class=cls, severe_count=drug_severe_count[drug])
    
    for d1, d2 in severe_pairs:
        if d1 in top_drug_names and d2 in top_drug_names:
            pair = tuple(sorted([d1, d2]))
            sev = severity_map.get(pair, 'Major')
            G.add_edge(d1, d2, severity=sev)
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Compact figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Layout
    pos = nx.kamada_kawai_layout(G)
    
    # Node properties
    node_colors = [ATC_COLORS.get(G.nodes[n]['drug_class'], '#808080') for n in G.nodes()]
    node_sizes = [200 + G.nodes[n]['severe_count'] * 3 for n in G.nodes()]
    
    # Edge colors
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        sev = G[u][v]['severity']
        if sev == 'Contraindicated':
            edge_colors.append('#d62728')
            edge_widths.append(1.5)
        else:
            edge_colors.append('#ff7f0e')
            edge_widths.append(1.0)
    
    # Draw
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6, 
                           width=edge_widths, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                           edgecolors='black', linewidths=0.8, ax=ax)
    
    # Compact labels
    labels = {n: n[:8] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight='bold', ax=ax)
    
    # Compact legend
    classes_in_graph = set(G.nodes[n]['drug_class'] for n in G.nodes())
    legend_elements = []
    for cls in sorted(classes_in_graph):
        legend_elements.append(mpatches.Patch(color=ATC_COLORS.get(cls, '#808080'), 
                                               label=f'{cls}'))
    legend_elements.append(plt.Line2D([0], [0], color='#d62728', linewidth=2, label='Contra'))
    legend_elements.append(plt.Line2D([0], [0], color='#ff7f0e', linewidth=2, label='Major'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=2,
              handlelength=1, handleheight=0.8, columnspacing=0.5)
    
    ax.set_title('Severe DDI Network', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/network_severe_ddi_compact.{fmt}', dpi=1200, 
                    bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: network_severe_ddi_compact.png/pdf")


if __name__ == '__main__':
    generate_chord_severity_compact()
    generate_severe_network_compact()
    print("\n✓ Both figures generated at 1200 DPI!")
