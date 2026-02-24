#!/usr/bin/env python3
"""
Generate Publication-Grade Statistics Figure for DDI Knowledge Graph
Clean, professional style suitable for academic journals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Use publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

OUTPUT_DIR = '/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/publication_knowledge_graph/figures'

def create_publication_stats_figure():
    """Create publication-grade statistics figure"""
    
    # Data
    nodes = {
        'Pathways': 25958,
        'Side Effects': 5548,
        'Drugs': 4313,
        'Categories': 3619,
        'Proteins': 3176,
        'Diseases': 3041,
    }
    
    edges = {
        'Drug-Drug Interaction': 759774,
        'Drug-Side Effect': 265238,
        'Drug-Category': 70618,
        'Drug-Disease': 63278,
        'Drug-Pathway': 31207,
        'Drug-Protein': 21559,
    }
    
    severity = {
        'Moderate': 608742,
        'Minor': 70682,
        'Contraindicated': 44306,
        'Major': 36044,
    }
    
    sources = {
        'DrugBank': 882158,
        'SIDER': 265238,
        'CTD': 63278,
    }
    
    class_severity = [
        ('Blood ↔ Musculoskeletal', 83.3, 8976),
        ('Antineoplastic ↔ Blood', 65.2, 18748),
        ('Blood ↔ Blood', 51.1, 7346),
        ('Blood ↔ Nervous System', 33.6, 15448),
        ('Anti-infectives ↔ Blood', 20.3, 13218),
        ('Alimentary ↔ Blood', 17.2, 5798),
    ]
    
    total_nodes = sum(nodes.values())
    total_edges = sum(edges.values())
    total_ddis = sum(severity.values())
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.3)
    
    # Color palette (muted, publication-friendly)
    colors_nodes = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027']
    colors_edges = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']
    colors_severity = {'Contraindicated': '#d73027', 'Major': '#fc8d59', 
                       'Moderate': '#91cf60', 'Minor': '#4575b4'}
    colors_sources = ['#4575b4', '#91bfdb', '#fee090']
    
    # =========================================================================
    # Panel A: Node Types - Treemap
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('A. Node Type Distribution', fontsize=12, fontweight='bold', loc='left')
    
    # Simple treemap using squarify algorithm
    node_items = sorted(nodes.items(), key=lambda x: -x[1])
    
    def squarify(values, x, y, width, height):
        """Simple squarify algorithm for treemap"""
        if len(values) == 0:
            return []
        
        total = sum(v for _, v in values)
        rects = []
        
        if width >= height:
            # Horizontal split
            curr_x = x
            for name, val in values:
                w = (val / total) * width
                rects.append((name, val, curr_x, y, w, height))
                curr_x += w
        else:
            # Vertical split
            curr_y = y
            for name, val in values:
                h = (val / total) * height
                rects.append((name, val, x, curr_y, width, h))
                curr_y += h
        
        return rects
    
    rects = squarify(node_items, 0, 0, 1, 1)
    
    for i, (name, val, rx, ry, rw, rh) in enumerate(rects):
        rect = Rectangle((rx, ry), rw, rh, facecolor=colors_nodes[i],
                         edgecolor='white', linewidth=2)
        ax1.add_patch(rect)
        
        # Label
        pct = (val / total_nodes) * 100
        if rw > 0.15 and rh > 0.15:
            ax1.text(rx + rw/2, ry + rh/2 + 0.05, name, ha='center', va='center',
                    fontsize=9, fontweight='bold')
            ax1.text(rx + rw/2, ry + rh/2 - 0.05, f'n={val:,}\n({pct:.1f}%)', 
                    ha='center', va='center', fontsize=8)
        elif rw > 0.08 or rh > 0.08:
            ax1.text(rx + rw/2, ry + rh/2, f'{name}\n{val:,}', ha='center', va='center',
                    fontsize=7)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(0.5, -0.08, f'Total: {total_nodes:,} nodes', ha='center', fontsize=10,
            transform=ax1.transAxes)
    
    # =========================================================================
    # Panel B: Edge Types - Proportional Area
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('B. Edge Type Distribution', fontsize=12, fontweight='bold', loc='left')
    
    edge_items = sorted(edges.items(), key=lambda x: -x[1])
    
    # Stacked horizontal segments
    y_positions = np.arange(len(edge_items))
    
    for i, (name, val) in enumerate(edge_items):
        width = val / total_edges
        rect = Rectangle((0, i), width, 0.7, facecolor=colors_edges[i],
                         edgecolor='black', linewidth=0.5)
        ax2.add_patch(rect)
        
        pct = (val / total_edges) * 100
        # Label
        short_name = name.replace('Drug-', '').replace(' Interaction', '')
        ax2.text(width + 0.02, i + 0.35, f'{short_name}: {val:,} ({pct:.1f}%)', 
                va='center', fontsize=9)
    
    ax2.set_xlim(0, 1.6)
    ax2.set_ylim(-0.3, len(edge_items))
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax2.set_yticks([])
    ax2.set_xlabel('Proportion of Total Edges')
    ax2.text(0.5, -0.15, f'Total: {total_edges:,} edges', ha='center', fontsize=10,
            transform=ax2.transAxes)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # =========================================================================
    # Panel C: Data Sources - Stacked Area
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('C. Data Source Contribution', fontsize=12, fontweight='bold', loc='left')
    
    source_items = sorted(sources.items(), key=lambda x: -x[1])
    
    # Nested rectangles
    max_val = source_items[0][1]
    for i, (name, val) in enumerate(source_items):
        size = np.sqrt(val / max_val)
        offset = (1 - size) / 2
        rect = Rectangle((offset, offset), size, size, facecolor=colors_sources[i],
                         edgecolor='black', linewidth=1, alpha=0.8)
        ax3.add_patch(rect)
        
        pct = (val / total_edges) * 100
        y_pos = offset + size/2 + (0.1 if i == 0 else 0)
        ax3.text(0.5, y_pos, f'{name}\n{val:,} ({pct:.0f}%)', ha='center', va='center',
                fontsize=9 if i == 0 else 8, fontweight='bold' if i == 0 else 'normal')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect('equal')
    
    # =========================================================================
    # Panel D: Severity Distribution - Waffle-style
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('D. DDI Severity Distribution', fontsize=12, fontweight='bold', loc='left')
    
    # Create waffle chart (10x10 grid = 100 cells)
    grid_size = 10
    total_cells = grid_size * grid_size
    
    # Calculate cells per category
    severity_order = ['Moderate', 'Minor', 'Contraindicated', 'Major']
    cell_counts = {}
    remaining = total_cells
    
    for i, sev in enumerate(severity_order[:-1]):
        count = round((severity[sev] / total_ddis) * total_cells)
        cell_counts[sev] = min(count, remaining)
        remaining -= cell_counts[sev]
    cell_counts[severity_order[-1]] = remaining
    
    # Draw waffle
    cell_idx = 0
    for sev in severity_order:
        color = colors_severity[sev]
        for _ in range(cell_counts[sev]):
            row = cell_idx // grid_size
            col = cell_idx % grid_size
            rect = Rectangle((col * 0.1, row * 0.1), 0.09, 0.09,
                            facecolor=color, edgecolor='white', linewidth=0.5)
            ax4.add_patch(rect)
            cell_idx += 1
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_aspect('equal')
    
    # Legend
    legend_y = -0.15
    for i, sev in enumerate(severity_order):
        pct = (severity[sev] / total_ddis) * 100
        rect = Rectangle((i * 0.25, legend_y), 0.05, 0.05,
                         facecolor=colors_severity[sev], edgecolor='black',
                         linewidth=0.5, transform=ax4.transAxes, clip_on=False)
        ax4.add_patch(rect)
        ax4.text(i * 0.25 + 0.07, legend_y + 0.025, f'{sev}\n({pct:.1f}%)',
                fontsize=7, va='center', transform=ax4.transAxes)
    
    ax4.text(0.5, -0.28, f'Total: {total_ddis:,} DDI pairs', ha='center', fontsize=10,
            transform=ax4.transAxes)
    
    # =========================================================================
    # Panel E: Summary Statistics Table
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('E. Summary Statistics', fontsize=12, fontweight='bold', loc='left')
    ax5.axis('off')
    
    stats_data = [
        ('Total Nodes', f'{total_nodes:,}'),
        ('Total Edges', f'{total_edges:,}'),
        ('Unique Drugs', '4,313'),
        ('DDI Pairs', '759,774'),
        ('Side Effects', '5,548'),
        ('Diseases', '3,041'),
        ('Protein Targets', '3,176'),
        ('Pathways', '25,958'),
    ]
    
    # Draw table
    row_height = 0.11
    for i, (label, value) in enumerate(stats_data):
        y = 0.88 - i * row_height
        # Alternating background
        if i % 2 == 0:
            rect = Rectangle((0, y - 0.02), 1, row_height, facecolor='#f0f0f0',
                            edgecolor='none', transform=ax5.transAxes, clip_on=False)
            ax5.add_patch(rect)
        
        ax5.text(0.05, y + 0.03, label, fontsize=10, va='center', transform=ax5.transAxes)
        ax5.text(0.95, y + 0.03, value, fontsize=10, va='center', ha='right',
                fontweight='bold', transform=ax5.transAxes)
    
    # Border
    rect = Rectangle((0, 0.88 - len(stats_data) * row_height + 0.09), 1, 
                     len(stats_data) * row_height, facecolor='none',
                     edgecolor='black', linewidth=1, transform=ax5.transAxes, clip_on=False)
    ax5.add_patch(rect)
    
    # =========================================================================
    # Panel F: High-Risk Drug Class Pairs
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('F. Severe DDI Rates by Drug Class', fontsize=12, fontweight='bold', loc='left')
    
    y_pos = np.arange(len(class_severity))
    
    for i, (name, pct, count) in enumerate(class_severity):
        # Background bar (100%)
        ax6.barh(i, 100, height=0.6, color='#e0e0e0', edgecolor='none')
        
        # Filled bar (severity %)
        color = '#d73027' if pct > 50 else '#fc8d59' if pct > 30 else '#fdae61' if pct > 15 else '#91cf60'
        ax6.barh(i, pct, height=0.6, color=color, edgecolor='none')
        
        # Label
        ax6.text(pct + 2, i, f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([x[0] for x in class_severity], fontsize=8)
    ax6.set_xlim(0, 100)
    ax6.set_xlabel('Percentage of Severe DDIs\n(Major + Contraindicated)')
    ax6.invert_yaxis()
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    # =========================================================================
    # Save
    # =========================================================================
    plt.tight_layout()
    
    output_path = f'{OUTPUT_DIR}/kg_stats_publication.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved publication-grade figure to {output_path}")
    return output_path


if __name__ == '__main__':
    print("Generating Publication-Grade Statistics Figure...")
    create_publication_stats_figure()
    print("Done!")
