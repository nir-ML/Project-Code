#!/usr/bin/env python3
"""
Network visualization for drug alternatives:
- Shows only severe (Major/Contraindicated) DDIs
- Network propagation to find safe alternatives within same class
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

# Rename columns
ddi_df = ddi_df.rename(columns={
    'drug_name_1': 'drug_1',
    'drug_name_2': 'drug_2',
    'severity_final': 'severity'
})
ddi_df['severity'] = ddi_df['severity'].str.replace(' interaction', '', regex=False)

print(f"Total DDIs: {len(ddi_df)}")
print(f"Severity distribution:\n{ddi_df['severity'].value_counts()}")

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
    # Handle string representation of list like "['B01AE02']"
    if isinstance(atc_code, str):
        if atc_code.startswith('['):
            # Extract first ATC code from list representation
            try:
                codes = ast.literal_eval(atc_code)
                if codes and len(codes) > 0:
                    return codes[0][0]  # First char of first code
            except:
                pass
        else:
            return atc_code[0] if len(atc_code) > 0 else None
    return None

# Map drugs to classes
ddi_df['class_1'] = ddi_df['atc_1'].apply(get_atc_class)
ddi_df['class_2'] = ddi_df['atc_2'].apply(get_atc_class)

# Build drug-class mapping
drug_to_class = {}
for _, row in ddi_df.iterrows():
    if pd.notna(row['class_1']):
        drug_to_class[row['drug_1']] = row['class_1']
    if pd.notna(row['class_2']):
        drug_to_class[row['drug_2']] = row['class_2']

# Build class-drugs mapping
class_to_drugs = defaultdict(set)
for drug, cls in drug_to_class.items():
    class_to_drugs[cls].add(drug)

print(f"\nDrugs with ATC class: {len(drug_to_class)}")
for cls in sorted(class_to_drugs.keys()):
    print(f"  {cls} ({ATC_CLASSES.get(cls, cls)}): {len(class_to_drugs[cls])} drugs")

# Build interaction indices
severe_pairs = set()
all_pairs = set()
severity_map = {}

for _, row in ddi_df.iterrows():
    d1, d2 = row['drug_1'], row['drug_2']
    sev = row['severity']
    pair = tuple(sorted([d1, d2]))
    all_pairs.add(pair)
    severity_map[pair] = sev
    if sev in ['Major', 'Contraindicated']:
        severe_pairs.add(pair)

print(f"\nSevere DDI pairs: {len(severe_pairs)}")
print(f"Total unique pairs: {len(all_pairs)}")


def generate_severe_network():
    """Generate network of severe DDIs only"""
    print("\n=== Generating Severe DDI Network ===")
    
    # Count severe interactions per drug
    drug_severe_count = defaultdict(int)
    for d1, d2 in severe_pairs:
        drug_severe_count[d1] += 1
        drug_severe_count[d2] += 1
    
    # Get top drugs by severe interaction count
    top_drugs = sorted(drug_severe_count.items(), key=lambda x: -x[1])[:30]
    top_drug_names = [d[0] for d in top_drugs]
    
    print(f"Top 10 drugs by severe DDI count:")
    for d, c in top_drugs[:10]:
        cls = drug_to_class.get(d, '?')
        print(f"  {d}: {c} severe DDIs (class {cls})")
    
    # Build network
    G = nx.Graph()
    
    for drug in top_drug_names:
        cls = drug_to_class.get(drug, 'V')
        G.add_node(drug, drug_class=cls, severe_count=drug_severe_count[drug])
    
    # Add edges (only severe)
    edge_colors = []
    for d1, d2 in severe_pairs:
        if d1 in top_drug_names and d2 in top_drug_names:
            pair = tuple(sorted([d1, d2]))
            sev = severity_map.get(pair, 'Major')
            G.add_edge(d1, d2, severity=sev)
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Layout
    pos = nx.kamada_kawai_layout(G)
    
    # Node colors by class
    node_colors = [ATC_COLORS.get(G.nodes[n]['drug_class'], '#808080') for n in G.nodes()]
    node_sizes = [300 + G.nodes[n]['severe_count'] * 5 for n in G.nodes()]
    
    # Edge colors by severity
    edge_colors = []
    for u, v in G.edges():
        sev = G[u][v]['severity']
        if sev == 'Contraindicated':
            edge_colors.append('#d62728')  # Red
        else:
            edge_colors.append('#ff7f0e')  # Orange
    
    # Draw
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6, width=1.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                           edgecolors='black', linewidths=1, ax=ax)
    
    # Labels
    labels = {n: n[:10] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)
    
    # Legend - classes
    class_legend = []
    classes_in_graph = set(G.nodes[n]['drug_class'] for n in G.nodes())
    for cls in sorted(classes_in_graph):
        class_legend.append(mpatches.Patch(color=ATC_COLORS.get(cls, '#808080'), 
                                            label=f'{cls}: {ATC_CLASSES.get(cls, cls)}'))
    
    # Legend - severity
    class_legend.append(plt.Line2D([0], [0], color='#d62728', linewidth=3, label='Contraindicated'))
    class_legend.append(plt.Line2D([0], [0], color='#ff7f0e', linewidth=3, label='Major'))
    
    ax.legend(handles=class_legend, loc='upper left', fontsize=9, ncol=2)
    
    ax.set_title('Severe DDI Network (Major & Contraindicated Only)\nNode color = Drug class, Edge color = Severity',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/network_severe_ddi.{fmt}', dpi=300, 
                    bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: network_severe_ddi.png/pdf")


def generate_alternative_propagation_network():
    """Generate network showing safe alternative propagation within class"""
    print("\n=== Generating Safe Alternative Propagation Network ===")
    
    # Focus on a specific high-risk scenario:
    # Patient on Blood class drug needs Musculoskeletal drug
    # Show which Blood drugs have severe DDIs and which are safe alternatives
    
    # Get Blood and Musculoskeletal drugs
    blood_drugs = list(class_to_drugs.get('B', set()))
    musc_drugs = list(class_to_drugs.get('M', set()))
    
    print(f"Blood drugs: {len(blood_drugs)}")
    print(f"Musculoskeletal drugs: {len(musc_drugs)}")
    
    # For each blood drug, count severe DDIs with musculoskeletal
    blood_musc_severe = defaultdict(list)
    blood_musc_safe = defaultdict(list)
    
    for bd in blood_drugs:
        for md in musc_drugs:
            pair = tuple(sorted([bd, md]))
            if pair in severe_pairs:
                blood_musc_severe[bd].append(md)
            elif pair in all_pairs:
                blood_musc_safe[bd].append(md)
    
    # Get blood drugs with most severe interactions
    blood_by_severe = sorted(blood_musc_severe.keys(), key=lambda x: -len(blood_musc_severe[x]))
    
    print(f"\nBlood drugs with severe M interactions:")
    for bd in blood_by_severe[:10]:
        print(f"  {bd}: {len(blood_musc_severe[bd])} severe, {len(blood_musc_safe[bd])} safe")
    
    # Select top problem drugs and some safe alternatives
    problem_blood = blood_by_severe[:5]
    
    # Find blood drugs with NO severe M interactions (safe alternatives)
    safe_blood = [bd for bd in blood_drugs if len(blood_musc_severe[bd]) == 0 and len(blood_musc_safe[bd]) > 0]
    print(f"\nBlood drugs with NO severe M interactions: {len(safe_blood)}")
    safe_blood = safe_blood[:5]
    
    # Select some musculoskeletal drugs
    musc_by_severe = sorted(musc_drugs, key=lambda x: sum(1 for bd in blood_drugs 
                                                          if tuple(sorted([bd, x])) in severe_pairs), reverse=True)
    target_musc = musc_by_severe[:5]
    
    # Build visualization network
    G = nx.Graph()
    
    # Add nodes
    for bd in problem_blood:
        G.add_node(bd, node_type='problem', drug_class='B')
    for bd in safe_blood:
        G.add_node(bd, node_type='safe_alt', drug_class='B')
    for md in target_musc:
        G.add_node(md, node_type='target', drug_class='M')
    
    # Add edges
    for bd in problem_blood + safe_blood:
        for md in target_musc:
            pair = tuple(sorted([bd, md]))
            if pair in severe_pairs:
                sev = severity_map.get(pair, 'Major')
                G.add_edge(bd, md, edge_type='severe', severity=sev)
            elif pair in all_pairs:
                G.add_edge(bd, md, edge_type='safe')
    
    # Add substitution edges between blood drugs
    for pb in problem_blood:
        for sb in safe_blood:
            G.add_edge(pb, sb, edge_type='substitute')
    
    print(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Custom layout: 3 columns
    pos = {}
    
    # Problem blood drugs on left
    for i, d in enumerate(problem_blood):
        pos[d] = (-1, 1 - i * 0.4)
    
    # Safe alternatives in middle-left
    for i, d in enumerate(safe_blood):
        pos[d] = (-0.3, 0.8 - i * 0.4)
    
    # Musculoskeletal drugs on right
    for i, d in enumerate(target_musc):
        pos[d] = (1, 1 - i * 0.4)
    
    # Draw edges by type
    severe_edges = [(u, v) for u, v in G.edges() if G[u][v].get('edge_type') == 'severe']
    safe_edges = [(u, v) for u, v in G.edges() if G[u][v].get('edge_type') == 'safe']
    subst_edges = [(u, v) for u, v in G.edges() if G[u][v].get('edge_type') == 'substitute']
    
    # Severe edges (red, thick)
    if severe_edges:
        edge_colors_severe = ['#d62728' if G[u][v].get('severity') == 'Contraindicated' else '#ff7f0e' 
                              for u, v in severe_edges]
        nx.draw_networkx_edges(G, pos, edgelist=severe_edges, edge_color=edge_colors_severe,
                               width=3, alpha=0.8, style='solid', ax=ax)
    
    # Safe edges (green)
    if safe_edges:
        nx.draw_networkx_edges(G, pos, edgelist=safe_edges, edge_color='#2ca02c',
                               width=2, alpha=0.7, style='solid', ax=ax)
    
    # Substitution edges (blue, dashed)
    if subst_edges:
        nx.draw_networkx_edges(G, pos, edgelist=subst_edges, edge_color='#1f77b4',
                               width=2, alpha=0.6, style='dashed', ax=ax)
    
    # Draw nodes
    problem_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'problem']
    safe_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'safe_alt']
    target_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'target']
    
    nx.draw_networkx_nodes(G, pos, nodelist=problem_nodes, node_color='#d62728',
                           node_size=2000, edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=safe_nodes, node_color='#2ca02c',
                           node_size=2000, edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='#ff7f0e',
                           node_size=2000, edgecolors='black', linewidths=2, ax=ax)
    
    # Labels
    labels = {n: n[:12] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Column headers
    ax.text(-1, 1.3, 'Problem Drugs\n(Blood class with\nsevere M interactions)', 
            ha='center', fontsize=11, fontweight='bold', color='#d62728')
    ax.text(-0.3, 1.3, 'Safe Alternatives\n(Blood class without\nsevere M interactions)', 
            ha='center', fontsize=11, fontweight='bold', color='#2ca02c')
    ax.text(1, 1.3, 'Target Drugs\n(Musculoskeletal)', 
            ha='center', fontsize=11, fontweight='bold', color='#ff7f0e')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#d62728', label='Problem Drug (severe DDIs)'),
        mpatches.Patch(color='#2ca02c', label='Safe Alternative (same class)'),
        mpatches.Patch(color='#ff7f0e', label='Target Drug (needed)'),
        plt.Line2D([0], [0], color='#d62728', linewidth=3, label='Contraindicated DDI'),
        plt.Line2D([0], [0], color='#ff7f0e', linewidth=3, label='Major DDI'),
        plt.Line2D([0], [0], color='#2ca02c', linewidth=3, label='Safe Combination'),
        plt.Line2D([0], [0], color='#1f77b4', linewidth=2, linestyle='--', label='Class Substitution'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=10, ncol=4,
              bbox_to_anchor=(0.5, -0.05))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1.6)
    ax.axis('off')
    
    ax.set_title('Safe Alternative Network Propagation\nBlood ↔ Musculoskeletal (83.3% severe rate)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/network_safe_alternatives.{fmt}', dpi=1200, 
                    bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: network_safe_alternatives.png/pdf")


def generate_class_severity_network():
    """Generate class-level network with severity encoding"""
    print("\n=== Generating Class-Level Severity Network ===")
    
    # Build class-pair statistics
    class_pairs = defaultdict(lambda: {'total': 0, 'severe': 0})
    
    for _, row in ddi_df.iterrows():
        c1, c2 = row['class_1'], row['class_2']
        if pd.isna(c1) or pd.isna(c2):
            continue
        pair = tuple(sorted([c1, c2]))
        class_pairs[pair]['total'] += 1
        if row['severity'] in ['Major', 'Contraindicated']:
            class_pairs[pair]['severe'] += 1
    
    # Build network
    G = nx.Graph()
    
    classes = sorted(set(drug_to_class.values()))
    for cls in classes:
        G.add_node(cls, name=ATC_CLASSES.get(cls, cls), drug_count=len(class_to_drugs[cls]))
    
    # Add edges with severity rate
    for (c1, c2), stats in class_pairs.items():
        if stats['total'] > 100:  # Minimum threshold
            sev_rate = stats['severe'] / stats['total']
            G.add_edge(c1, c2, total=stats['total'], severe=stats['severe'], 
                       sev_rate=sev_rate)
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Print high-severity pairs
    print("\nHigh-severity class pairs:")
    for c1, c2, data in sorted(G.edges(data=True), key=lambda x: -x[2]['sev_rate'])[:10]:
        print(f"  {ATC_CLASSES.get(c1, c1)} ↔ {ATC_CLASSES.get(c2, c2)}: "
              f"{data['sev_rate']*100:.1f}% severe ({data['severe']}/{data['total']})")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Circular layout
    pos = nx.circular_layout(G)
    
    # Node sizes by drug count
    node_sizes = [500 + G.nodes[n]['drug_count'] * 3 for n in G.nodes()]
    
    # Node colors by class
    node_colors = [ATC_COLORS.get(n, '#808080') for n in G.nodes()]
    
    # Edge colors by severity rate
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        sev_rate = G[u][v]['sev_rate']
        if sev_rate > 0.5:
            edge_colors.append('#d62728')  # Red
        elif sev_rate > 0.3:
            edge_colors.append('#ff7f0e')  # Orange
        elif sev_rate > 0.15:
            edge_colors.append('#fdae61')  # Light orange
        elif sev_rate > 0.08:
            edge_colors.append('#91cf60')  # Light green
        else:
            edge_colors.append('#2ca02c')  # Green
        
        # Width by total count
        edge_widths.append(1 + np.log10(G[u][v]['total']) * 2)
    
    # Draw
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                           alpha=0.7, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors='black', linewidths=2, ax=ax)
    
    # Labels
    labels = {n: f"{n}\n{ATC_CLASSES.get(n, n)[:8]}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='#d62728', linewidth=4, label='>50% severe'),
        plt.Line2D([0], [0], color='#ff7f0e', linewidth=4, label='30-50% severe'),
        plt.Line2D([0], [0], color='#fdae61', linewidth=4, label='15-30% severe'),
        plt.Line2D([0], [0], color='#91cf60', linewidth=4, label='8-15% severe'),
        plt.Line2D([0], [0], color='#2ca02c', linewidth=4, label='<8% severe'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, title='Severity Rate')
    
    ax.set_title('Drug Class Interaction Network\nEdge color = Severity rate, Node size = Drug count',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/network_class_severity.{fmt}', dpi=1200, 
                    bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: network_class_severity.png/pdf")


if __name__ == '__main__':
    # generate_severe_network()  # Removed
    generate_alternative_propagation_network()
    generate_class_severity_network()
    print("\n✓ All network visualizations generated!")
