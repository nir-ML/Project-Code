#!/usr/bin/env python3
"""
Visualize Centrality Measures for DDI Knowledge Graph
Generates publication-quality figures for network analysis
"""

import sys
sys.path.insert(0, '/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph')

import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import os

# Import all KG classes for unpickling
from ddi_knowledge_graph_enriched import (
    EnrichedDDIKnowledgeGraph, DrugNode, ProteinNode, SideEffectNode,
    DiseaseNode, PathwayNode, DDIEdge, DrugTargetEdge, DrugSideEffectEdge,
    DrugDiseaseEdge, SimilarityEdge, SIDERIntegrator, KEGGIntegrator,
    PubChemIntegrator, DrugTargetIntegrator
)

# Set publication style - 1200 DPI for publication quality
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight'
})

def load_knowledge_graph():
    """Load the enriched knowledge graph"""
    kg_path = "/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/knowledge_graph_enriched/enriched_kg.pkl"
    print(f"Loading knowledge graph from {kg_path}...")
    with open(kg_path, 'rb') as f:
        kg_obj = pickle.load(f)
    # Get the internal networkx graph
    kg = kg_obj.graph
    print(f"Loaded graph with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges")
    return kg, kg_obj

def extract_ddi_subgraph(kg):
    """Extract DDI subgraph (drug-drug interactions only)"""
    print("Extracting DDI subgraph...")
    
    # Get drug nodes - check both node_type attribute and direct type
    drug_nodes = []
    for n, d in kg.nodes(data=True):
        if d.get('node_type') == 'Drug' or d.get('type') == 'Drug':
            drug_nodes.append(n)
    
    print(f"  Found {len(drug_nodes)} drug nodes")
    
    # Get DDI edges
    ddi_edges = set()
    for u, v, d in kg.edges(data=True):
        edge_type = d.get('edge_type', d.get('type', ''))
        if 'INTERACTS' in str(edge_type).upper() or edge_type == 'DDI':
            ddi_edges.add((u, v))
    
    print(f"  Found {len(ddi_edges)} DDI edges")
    
    # Create undirected graph for centrality analysis
    ddi_graph = nx.Graph()
    ddi_graph.add_nodes_from(drug_nodes)
    ddi_graph.add_edges_from(ddi_edges)
    
    # Copy node attributes
    for n in ddi_graph.nodes():
        if kg.has_node(n):
            ddi_graph.nodes[n].update(kg.nodes[n])
    
    # Remove isolated nodes
    ddi_graph.remove_nodes_from(list(nx.isolates(ddi_graph)))
    
    print(f"DDI subgraph: {ddi_graph.number_of_nodes()} drugs, {ddi_graph.number_of_edges()} interactions")
    return ddi_graph

def compute_centralities(G, sample_size=500):
    """Compute various centrality measures"""
    print("Computing centrality measures...")
    
    # For large graphs, sample for betweenness
    if G.number_of_nodes() > sample_size:
        print(f"  Using sampling for betweenness (k={sample_size})...")
        betweenness = nx.betweenness_centrality(G, k=sample_size, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    
    print("  Computing degree centrality...")
    degree = nx.degree_centrality(G)
    
    print("  Computing eigenvector centrality...")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        print("  Eigenvector failed, using degree as fallback")
        eigenvector = degree
    
    print("  Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)
    
    print("  Computing closeness centrality (sampling)...")
    # Sample for closeness due to computational cost
    nodes = list(G.nodes())
    if len(nodes) > 200:
        sample_nodes = np.random.choice(nodes, 200, replace=False)
        closeness = {}
        for n in sample_nodes:
            closeness[n] = nx.closeness_centrality(G, u=n)
        # Fill missing with average
        avg_closeness = np.mean(list(closeness.values()))
        for n in nodes:
            if n not in closeness:
                closeness[n] = avg_closeness
    else:
        closeness = nx.closeness_centrality(G)
    
    return {
        'degree': degree,
        'betweenness': betweenness,
        'eigenvector': eigenvector,
        'pagerank': pagerank,
        'closeness': closeness
    }

def plot_centrality_distributions(centralities, output_dir):
    """Plot distribution of each centrality measure"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    titles = ['Degree Centrality', 'Betweenness Centrality', 'Eigenvector Centrality', 
              'PageRank', 'Closeness Centrality']
    keys = ['degree', 'betweenness', 'eigenvector', 'pagerank', 'closeness']
    
    for i, (key, title, color) in enumerate(zip(keys, titles, colors)):
        values = list(centralities[key].values())
        axes[i].hist(values, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[i].set_xlabel(title)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(title)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
        axes[i].legend(loc='upper right')
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centrality_distributions.png'), dpi=1200)
    plt.savefig(os.path.join(output_dir, 'centrality_distributions.pdf'))
    print(f"Saved: centrality_distributions.png/pdf")
    plt.close()

def plot_centrality_correlations(centralities, output_dir):
    """Plot correlation matrix and scatter plots between centralities"""
    import pandas as pd
    
    # Create DataFrame
    nodes = list(centralities['degree'].keys())
    df = pd.DataFrame({
        'Degree': [centralities['degree'][n] for n in nodes],
        'Betweenness': [centralities['betweenness'][n] for n in nodes],
        'Eigenvector': [centralities['eigenvector'][n] for n in nodes],
        'PageRank': [centralities['pagerank'][n] for n in nodes],
        'Closeness': [centralities['closeness'][n] for n in nodes]
    }, index=nodes)
    
    # Correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation')
    
    # Add labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)
    
    ax.set_title('Centrality Measure Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centrality_correlations.png'), dpi=1200)
    plt.savefig(os.path.join(output_dir, 'centrality_correlations.pdf'))
    print(f"Saved: centrality_correlations.png/pdf")
    plt.close()
    
    return df

def plot_top_drugs_centrality(centralities, kg, output_dir, top_n=15):
    """Bar chart of top drugs by each centrality measure"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    measures = [
        ('degree', 'Degree Centrality', '#3498db'),
        ('betweenness', 'Betweenness Centrality', '#e74c3c'),
        ('eigenvector', 'Eigenvector Centrality', '#2ecc71'),
        ('pagerank', 'PageRank', '#9b59b6')
    ]
    
    for i, (key, title, color) in enumerate(measures):
        # Get top N drugs
        sorted_nodes = sorted(centralities[key].items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get drug names from KG if available
        names = []
        values = []
        for node, val in sorted_nodes:
            if kg.has_node(node):
                name = kg.nodes[node].get('name', node)
                # Truncate long names
                if len(name) > 20:
                    name = name[:17] + '...'
            else:
                name = node[:20] if len(node) > 20 else node
            names.append(name)
            values.append(val)
        
        # Horizontal bar chart
        y_pos = np.arange(len(names))
        axes[i].barh(y_pos, values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(names)
        axes[i].invert_yaxis()
        axes[i].set_xlabel(title)
        axes[i].set_title(f'Top {top_n} Drugs by {title}')
        
        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(v + max(values)*0.01, j, f'{v:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_drugs_centrality.png'), dpi=1200)
    plt.savefig(os.path.join(output_dir, 'top_drugs_centrality.pdf'))
    print(f"Saved: top_drugs_centrality.png/pdf")
    plt.close()

def plot_centrality_network(G, centralities, output_dir, top_n=50):
    """Network visualization with node size by centrality"""
    print(f"Creating network visualization with top {top_n} drugs...")
    
    # Get top N drugs by degree
    sorted_by_degree = sorted(centralities['degree'].items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes = [n for n, _ in sorted_by_degree]
    
    # Create subgraph
    subG = G.subgraph(top_nodes).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Layout
    pos = nx.spring_layout(subG, k=2, iterations=100, seed=42)
    
    # Plot 1: Node size by Degree, color by Betweenness
    ax = axes[0]
    node_sizes = [centralities['degree'][n] * 3000 + 100 for n in subG.nodes()]
    node_colors = [centralities['betweenness'][n] for n in subG.nodes()]
    
    nodes = nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=node_sizes, 
                                    node_color=node_colors, cmap='YlOrRd',
                                    alpha=0.8, linewidths=1, edgecolors='black')
    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.2, width=0.5)
    
    # Add labels for top 10
    top10 = [n for n, _ in sorted_by_degree[:10]]
    labels = {n: n[:15] for n in top10}
    nx.draw_networkx_labels(subG, pos, labels, ax=ax, font_size=7)
    
    plt.colorbar(nodes, ax=ax, label='Betweenness Centrality')
    ax.set_title('Node Size: Degree | Color: Betweenness')
    ax.axis('off')
    
    # Plot 2: Node size by PageRank, color by Eigenvector
    ax = axes[1]
    node_sizes = [centralities['pagerank'][n] * 50000 + 100 for n in subG.nodes()]
    node_colors = [centralities['eigenvector'][n] for n in subG.nodes()]
    
    nodes = nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=node_sizes,
                                    node_color=node_colors, cmap='YlGnBu',
                                    alpha=0.8, linewidths=1, edgecolors='black')
    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.2, width=0.5)
    nx.draw_networkx_labels(subG, pos, labels, ax=ax, font_size=7)
    
    plt.colorbar(nodes, ax=ax, label='Eigenvector Centrality')
    ax.set_title('Node Size: PageRank | Color: Eigenvector')
    ax.axis('off')
    
    plt.suptitle(f'DDI Network Centrality Visualization (Top {top_n} Drugs)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centrality_network.png'), dpi=1200)
    plt.savefig(os.path.join(output_dir, 'centrality_network.pdf'))
    print(f"Saved: centrality_network.png/pdf")
    plt.close()

def plot_combined_centrality_heatmap(centralities, kg, output_dir, top_n=30):
    """Heatmap showing multiple centrality measures for top drugs"""
    import pandas as pd
    
    # Get top drugs by average rank across centralities
    all_ranks = {}
    for key in ['degree', 'betweenness', 'eigenvector', 'pagerank']:
        sorted_nodes = sorted(centralities[key].items(), key=lambda x: x[1], reverse=True)
        for rank, (node, _) in enumerate(sorted_nodes):
            if node not in all_ranks:
                all_ranks[node] = []
            all_ranks[node].append(rank)
    
    avg_ranks = {n: np.mean(ranks) for n, ranks in all_ranks.items() if len(ranks) == 4}
    top_nodes = sorted(avg_ranks.items(), key=lambda x: x[1])[:top_n]
    top_node_ids = [n for n, _ in top_nodes]
    
    # Build data matrix
    data = []
    labels = []
    for node in top_node_ids:
        if kg.has_node(node):
            name = kg.nodes[node].get('name', node)
            if len(name) > 25:
                name = name[:22] + '...'
        else:
            name = node[:25]
        labels.append(name)
        data.append([
            centralities['degree'][node],
            centralities['betweenness'][node],
            centralities['eigenvector'][node],
            centralities['pagerank'][node]
        ])
    
    # Normalize columns for visualization
    data = np.array(data)
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(data_norm, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Degree', 'Betweenness', 'Eigenvector', 'PageRank'])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    
    plt.colorbar(im, ax=ax, label='Normalized Centrality')
    ax.set_title(f'Top {top_n} Drugs: Multi-Centrality Profile')
    ax.set_xlabel('Centrality Measure')
    ax.set_ylabel('Drug')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centrality_heatmap.png'), dpi=1200)
    plt.savefig(os.path.join(output_dir, 'centrality_heatmap.pdf'))
    print(f"Saved: centrality_heatmap.png/pdf")
    plt.close()

def main():
    output_dir = "/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/publication_knowledge_graph/knowledge_graph/method_brief/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process
    kg, kg_obj = load_knowledge_graph()
    ddi_graph = extract_ddi_subgraph(kg)
    
    # Compute centralities
    centralities = compute_centralities(ddi_graph)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_centrality_distributions(centralities, output_dir)
    df = plot_centrality_correlations(centralities, output_dir)
    plot_top_drugs_centrality(centralities, kg, output_dir)
    plot_centrality_network(ddi_graph, centralities, output_dir)
    plot_combined_centrality_heatmap(centralities, kg, output_dir)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CENTRALITY SUMMARY STATISTICS")
    print("="*60)
    for key in ['degree', 'betweenness', 'eigenvector', 'pagerank']:
        values = list(centralities[key].values())
        print(f"\n{key.upper()}:")
        print(f"  Mean: {np.mean(values):.6f}")
        print(f"  Std:  {np.std(values):.6f}")
        print(f"  Max:  {np.max(values):.6f}")
        print(f"  Min:  {np.min(values):.6f}")
    
    print("\n" + "="*60)
    print("Visualizations saved to:", output_dir)
    print("="*60)

if __name__ == "__main__":
    main()
