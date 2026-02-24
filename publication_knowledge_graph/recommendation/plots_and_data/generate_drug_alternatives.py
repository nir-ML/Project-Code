#!/usr/bin/env python3
"""
Generate drug alternative visualization:
Shows within-class drug substitutions to avoid severe DDIs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
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

def get_atc_class(atc_code):
    if pd.isna(atc_code) or atc_code == '':
        return None
    return atc_code[0] if len(atc_code) > 0 else None

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

print(f"Total drugs with ATC class: {len(drug_to_class)}")

# Build severe interaction index
severe_pairs = set()
for _, row in ddi_df[ddi_df['severity'].isin(['Major', 'Contraindicated'])].iterrows():
    severe_pairs.add((row['drug_1'], row['drug_2']))
    severe_pairs.add((row['drug_2'], row['drug_1']))

print(f"Severe DDI pairs: {len(severe_pairs)//2}")


def find_alternatives(target_drug, interacting_drug):
    """Find drugs in same class as target that don't have severe DDI with interacting_drug"""
    if target_drug not in drug_to_class:
        return []
    
    target_class = drug_to_class[target_drug]
    alternatives = []
    
    for alt_drug in class_to_drugs[target_class]:
        if alt_drug == target_drug:
            continue
        if (alt_drug, interacting_drug) not in severe_pairs:
            alternatives.append(alt_drug)
    
    return alternatives


def generate_alternatives_heatmap():
    """Generate heatmap showing drug alternatives within classes"""
    print("\nGenerating Drug Alternatives Heatmap...")
    
    # Focus on high-risk class pairs
    high_risk_pairs = [
        ('B', 'M'),  # Blood - Musculoskeletal (83.3% severe)
        ('L', 'B'),  # Antineoplastic - Blood (65.2% severe)
        ('B', 'B'),  # Blood - Blood (51.1% severe)
        ('B', 'N'),  # Blood - Nervous System (33.6% severe)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (cls1, cls2) in enumerate(high_risk_pairs):
        ax = axes[idx]
        
        # Get drugs in each class (top 8 by interaction count)
        drugs1 = list(class_to_drugs[cls1])[:15]
        drugs2 = list(class_to_drugs[cls2])[:15] if cls1 != cls2 else drugs1
        
        # Count severe DDIs for each drug to sort
        drug_severe_count = defaultdict(int)
        for d in drugs1 + drugs2:
            for pair in severe_pairs:
                if d in pair:
                    drug_severe_count[d] += 1
        
        drugs1 = sorted(drugs1, key=lambda x: -drug_severe_count[x])[:8]
        drugs2 = sorted(drugs2, key=lambda x: -drug_severe_count[x])[:8]
        
        n1, n2 = len(drugs1), len(drugs2)
        matrix = np.zeros((n1, n2))
        
        for i, d1 in enumerate(drugs1):
            for j, d2 in enumerate(drugs2):
                if d1 == d2:
                    matrix[i, j] = -1  # Same drug
                elif (d1, d2) in severe_pairs:
                    matrix[i, j] = 2  # Severe
                else:
                    matrix[i, j] = 0  # Safe
        
        # Custom colormap
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#808080', '#2ca02c', '#d62728'])  # Gray (same), Green (safe), Red (severe)
        
        im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=2, aspect='auto')
        
        # Labels
        short_drugs1 = [d[:10] + '..' if len(d) > 12 else d for d in drugs1]
        short_drugs2 = [d[:10] + '..' if len(d) > 12 else d for d in drugs2]
        
        ax.set_xticks(range(n2))
        ax.set_yticks(range(n1))
        ax.set_xticklabels(short_drugs2, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(short_drugs1, fontsize=8)
        
        # Grid
        ax.set_xticks(np.arange(-0.5, n2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n1, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
        
        cls1_name = ATC_CLASSES.get(cls1, cls1)
        cls2_name = ATC_CLASSES.get(cls2, cls2)
        ax.set_xlabel(f'{cls2_name} Class', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{cls1_name} Class', fontsize=10, fontweight='bold')
        ax.set_title(f'{cls1_name} ↔ {cls2_name}', fontsize=11, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#d62728', label='Severe DDI (avoid)'),
        mpatches.Patch(color='#2ca02c', label='Safe Alternative'),
        mpatches.Patch(color='#808080', label='Same Drug'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, 
               bbox_to_anchor=(0.5, 0.98))
    
    plt.suptitle('Within-Class Drug Alternatives for High-Risk Pairs\n(Green = Safe substitution to avoid severe DDI)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/drug_alternatives_heatmap.{fmt}', dpi=300, 
                    bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: drug_alternatives_heatmap.png/pdf")


def generate_substitution_network():
    """Generate network showing drug substitutions within a class"""
    print("\nGenerating Drug Substitution Network...")
    
    import networkx as nx
    
    # Focus on Blood class (highest severity)
    target_class = 'B'
    interacting_class = 'M'  # Musculoskeletal (83.3% severe with Blood)
    
    blood_drugs = list(class_to_drugs[target_class])
    musc_drugs = list(class_to_drugs[interacting_class])
    
    # Find blood drugs with severe interactions with musculoskeletal
    blood_severe = defaultdict(list)
    for bd in blood_drugs:
        for md in musc_drugs:
            if (bd, md) in severe_pairs:
                blood_severe[bd].append(md)
    
    # Get top blood drugs by severe DDI count
    top_blood = sorted(blood_severe.keys(), key=lambda x: -len(blood_severe[x]))[:6]
    
    # For each, find alternatives
    fig, ax = plt.subplots(figsize=(14, 10))
    
    G = nx.Graph()
    
    # Add nodes
    node_colors = []
    node_sizes = []
    labels = {}
    
    # Central node for each problem drug
    y_positions = np.linspace(0.8, 0.2, len(top_blood))
    pos = {}
    
    for i, drug in enumerate(top_blood):
        # Problem drug (red)
        G.add_node(drug, node_type='problem')
        pos[drug] = (0.2, y_positions[i])
        node_colors.append('#d62728')
        node_sizes.append(2000)
        labels[drug] = drug[:12]
        
        # Alternatives (green) - up to 3
        alts = find_alternatives(drug, musc_drugs[0])[:3]  # Check against first musc drug
        for j, alt in enumerate(alts):
            if alt not in G:
                G.add_node(alt, node_type='alternative')
                pos[alt] = (0.5, y_positions[i] + (j-1)*0.05)
                node_colors.append('#2ca02c')
                node_sizes.append(1500)
                labels[alt] = alt[:12]
            G.add_edge(drug, alt, edge_type='substitute')
        
        # Add one interacting drug example
        if blood_severe[drug]:
            inter_drug = blood_severe[drug][0]
            if inter_drug not in G:
                G.add_node(inter_drug, node_type='interacting')
                pos[inter_drug] = (0.8, y_positions[i])
                node_colors.append('#ff7f0e')
                node_sizes.append(1200)
                labels[inter_drug] = inter_drug[:12]
            G.add_edge(drug, inter_drug, edge_type='severe')
    
    # Draw
    edge_colors = ['#2ca02c' if G[u][v].get('edge_type') == 'substitute' else '#d62728' 
                   for u, v in G.edges()]
    edge_styles = ['solid' if G[u][v].get('edge_type') == 'substitute' else 'dashed' 
                   for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    
    # Draw edges separately for different styles
    substitute_edges = [(u, v) for u, v in G.edges() if G[u][v].get('edge_type') == 'substitute']
    severe_edges = [(u, v) for u, v in G.edges() if G[u][v].get('edge_type') == 'severe']
    
    nx.draw_networkx_edges(G, pos, edgelist=substitute_edges, edge_color='#2ca02c', 
                           width=2, style='solid', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=severe_edges, edge_color='#d62728', 
                           width=2, style='dashed', ax=ax)
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#d62728', label='Problem Drug (has severe DDI)'),
        mpatches.Patch(color='#2ca02c', label='Safe Alternative (same class)'),
        mpatches.Patch(color='#ff7f0e', label='Interacting Drug'),
        plt.Line2D([0], [0], color='#2ca02c', linewidth=2, label='Substitution'),
        plt.Line2D([0], [0], color='#d62728', linewidth=2, linestyle='--', label='Severe DDI'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Column labels
    ax.text(0.2, 0.95, 'Blood Class\n(Problem Drugs)', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.95, 'Blood Class\n(Alternatives)', ha='center', fontsize=11, fontweight='bold', color='#2ca02c')
    ax.text(0.8, 0.95, 'Musculoskeletal\n(Interacting)', ha='center', fontsize=11, fontweight='bold', color='#ff7f0e')
    
    plt.title('Drug Substitution to Avoid Severe DDIs\nBlood ↔ Musculoskeletal (83.3% severe rate)', 
              fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/drug_substitution_network.{fmt}', dpi=300, 
                    bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: drug_substitution_network.png/pdf")


def generate_class_alternatives_summary():
    """Generate summary showing alternative availability per class pair"""
    print("\nGenerating Class Alternatives Summary...")
    
    # Calculate for each class pair: % of drugs with alternatives
    classes = sorted(set(drug_to_class.values()))
    n = len(classes)
    
    # Matrix: alt_rate[i][j] = % of class i drugs that have safe alternative when interacting with class j
    alt_rate = np.zeros((n, n))
    total_safe = np.zeros((n, n))
    total_severe = np.zeros((n, n))
    
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            drugs1 = list(class_to_drugs[c1])
            drugs2 = list(class_to_drugs[c2])
            
            # Count severe and safe pairs
            severe_count = 0
            safe_count = 0
            
            for d1 in drugs1:
                for d2 in drugs2:
                    if d1 != d2:
                        if (d1, d2) in severe_pairs:
                            severe_count += 1
                        else:
                            safe_count += 1
            
            total = severe_count + safe_count
            if total > 0:
                alt_rate[i, j] = safe_count / total * 100
                total_safe[i, j] = safe_count
                total_severe[i, j] = severe_count
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Custom colormap: Red (low alternatives) to Green (high alternatives)
    cmap = plt.cm.RdYlGn
    
    im = ax.imshow(alt_rate, cmap=cmap, vmin=0, vmax=100, aspect='equal')
    
    # Labels
    class_labels = [ATC_CLASSES.get(c, c) for c in classes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)
    
    # Add percentage text
    for i in range(n):
        for j in range(n):
            rate = alt_rate[i, j]
            if total_safe[i, j] + total_severe[i, j] > 100:  # Only label significant pairs
                text_color = 'white' if rate < 30 or rate > 70 else 'black'
                ax.text(j, i, f'{rate:.0f}%', ha='center', va='center', 
                        fontsize=7, color=text_color, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Safe Drug Combinations (%)', fontsize=11)
    
    # Grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Drug Class 2', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drug Class 1', fontsize=12, fontweight='bold')
    
    plt.title('Safe Drug Alternative Availability by Class Pair\n(Higher % = More replacement options to avoid severe DDI)', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/class_alternatives_summary.{fmt}', dpi=300, 
                    bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: class_alternatives_summary.png/pdf")


if __name__ == '__main__':
    generate_alternatives_heatmap()
    generate_substitution_network()
    generate_class_alternatives_summary()
    print("\n✓ All drug alternative visualizations generated!")
