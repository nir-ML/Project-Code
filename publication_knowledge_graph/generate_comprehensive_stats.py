#!/usr/bin/env python3
"""
Generate a beautiful comprehensive statistics figure for DDI Knowledge Graph
Uses creative visualizations: treemap, donut, bubble, and infographic elements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/publication_knowledge_graph/figures'

def create_comprehensive_stats_figure():
    """Create a beautiful single figure with all KG statistics"""
    
    # Knowledge Graph Statistics
    stats = {
        'nodes': {
            'Drugs': 4313,
            'Pathways': 25958,
            'Side Effects': 5548,
            'Categories': 3619,
            'Proteins': 3176,
            'Diseases': 3041,
        },
        'edges': {
            'Drug-Drug': 759774,
            'Drug-SE': 265238,
            'Drug-Category': 70618,
            'Drug-Disease': 63278,
            'Drug-Pathway': 31207,
            'Drug-Protein': 21559,
        },
        'severity': {
            'Moderate': 608742,
            'Minor': 70682,
            'Contraindicated': 44306,
            'Major': 36044,
        },
        'sources': {
            'DrugBank': 882158,
            'SIDER': 265238,
            'CTD': 63278,
        }
    }
    
    total_nodes = sum(stats['nodes'].values())
    total_edges = sum(stats['edges'].values())
    total_ddis = sum(stats['severity'].values())
    
    # Create figure with dark theme
    fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    
    # Main title
    fig.suptitle('DDI Knowledge Graph Statistics', 
                 fontsize=28, fontweight='bold', color='white', y=0.96)
    fig.text(0.5, 0.92, f'{total_nodes:,} Nodes  •  {total_edges:,} Edges  •  {total_ddis:,} DDI Pairs',
             ha='center', fontsize=16, color='#888888')
    
    # =========================================================================
    # PANEL 1: Node Types - Bubble Chart (Top Left)
    # =========================================================================
    ax1 = fig.add_axes([0.02, 0.48, 0.32, 0.40])
    ax1.set_facecolor('#1a1a2e')
    
    node_colors = ['#e94560', '#0f3460', '#16213e', '#533483', '#4ecca3', '#ff6b6b']
    node_items = list(stats['nodes'].items())
    
    # Calculate bubble sizes (sqrt for area perception)
    max_val = max(stats['nodes'].values())
    
    # Position bubbles in a cluster
    positions = [
        (0.5, 0.7),   # Pathways (largest)
        (0.25, 0.4),  # Side Effects
        (0.7, 0.35),  # Drugs
        (0.4, 0.25),  # Categories
        (0.75, 0.65), # Proteins
        (0.15, 0.7),  # Diseases
    ]
    
    for i, ((name, count), (x, y)) in enumerate(zip(node_items, positions)):
        size = np.sqrt(count / max_val) * 0.25
        circle = Circle((x, y), size, facecolor=node_colors[i], 
                        edgecolor='white', linewidth=2, alpha=0.85)
        ax1.add_patch(circle)
        
        # Label
        fontsize = 10 if count < 5000 else 12
        ax1.text(x, y + 0.02, name, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', color='white')
        ax1.text(x, y - 0.05, f'{count:,}', ha='center', va='center',
                fontsize=9, color='white', alpha=0.9)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Node Types', fontsize=16, fontweight='bold', color='white', pad=10)
    ax1.axis('off')
    
    # =========================================================================
    # PANEL 2: Edge Types - Proportional Stacked Rectangles (Top Middle)
    # =========================================================================
    ax2 = fig.add_axes([0.36, 0.48, 0.28, 0.40])
    ax2.set_facecolor('#1a1a2e')
    
    edge_colors = ['#e94560', '#4ecca3', '#ff6b6b', '#45b7d1', '#96ceb4', '#ffeaa7']
    edge_items = sorted(stats['edges'].items(), key=lambda x: -x[1])
    
    # Create horizontal stacked proportional bars
    y_pos = 0.85
    bar_height = 0.1
    
    for i, (name, count) in enumerate(edge_items):
        width = (count / total_edges) * 0.9
        rect = FancyBboxPatch((0.05, y_pos - i * 0.14), width, bar_height,
                              boxstyle="round,pad=0.01,rounding_size=0.02",
                              facecolor=edge_colors[i], edgecolor='white',
                              linewidth=1.5, alpha=0.9)
        ax2.add_patch(rect)
        
        # Label on bar
        pct = (count / total_edges) * 100
        label_x = 0.05 + width / 2
        ax2.text(label_x, y_pos - i * 0.14 + bar_height/2, 
                f'{name}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        
        # Count on right
        ax2.text(0.05 + width + 0.02, y_pos - i * 0.14 + bar_height/2,
                f'{count:,} ({pct:.1f}%)', ha='left', va='center',
                fontsize=8, color='white', alpha=0.8)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Edge Types', fontsize=16, fontweight='bold', color='white', pad=10)
    ax2.axis('off')
    
    # =========================================================================
    # PANEL 3: Data Sources - Concentric Rings (Top Right)
    # =========================================================================
    ax3 = fig.add_axes([0.66, 0.48, 0.32, 0.40])
    ax3.set_facecolor('#1a1a2e')
    ax3.set_aspect('equal')
    
    source_colors = ['#e94560', '#4ecca3', '#45b7d1']
    source_items = sorted(stats['sources'].items(), key=lambda x: -x[1])
    
    center = (0.5, 0.5)
    max_radius = 0.35
    
    for i, (name, count) in enumerate(source_items):
        radius = max_radius * (1 - i * 0.25)
        pct = (count / total_edges) * 100
        
        # Draw filled circle
        circle = Circle(center, radius, facecolor=source_colors[i],
                       edgecolor='white', linewidth=2, alpha=0.7 - i*0.15)
        ax3.add_patch(circle)
    
    # Add labels
    for i, (name, count) in enumerate(source_items):
        radius = max_radius * (1 - i * 0.25)
        pct = (count / total_edges) * 100
        
        if i == 0:
            y_offset = radius + 0.08
        else:
            y_offset = max_radius * (1 - (i-0.5) * 0.25)
        
        ax3.text(0.5, 0.5 + y_offset, f'{name}',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white')
        ax3.text(0.5, 0.5 + y_offset - 0.06, f'{count:,} ({pct:.0f}%)',
                ha='center', va='center', fontsize=9, color='white', alpha=0.8)
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Data Sources', fontsize=16, fontweight='bold', color='white', pad=10)
    ax3.axis('off')
    
    # =========================================================================
    # PANEL 4: Severity Distribution - Radial Gauge (Bottom Left)
    # =========================================================================
    ax4 = fig.add_axes([0.02, 0.05, 0.32, 0.38])
    ax4.set_facecolor('#1a1a2e')
    ax4.set_aspect('equal')
    
    severity_colors = {
        'Contraindicated': '#d62728',
        'Major': '#ff7f0e',
        'Moderate': '#2ca02c',
        'Minor': '#1f77b4'
    }
    
    # Create donut chart
    center = (0.5, 0.45)
    outer_r = 0.35
    inner_r = 0.20
    
    angles = []
    start_angle = 90  # Start from top
    
    for name in ['Moderate', 'Minor', 'Contraindicated', 'Major']:
        count = stats['severity'][name]
        angle = (count / total_ddis) * 360
        angles.append((name, count, start_angle, angle))
        start_angle -= angle
    
    for name, count, start, extent in angles:
        wedge = Wedge(center, outer_r, start - extent, start,
                     width=outer_r - inner_r,
                     facecolor=severity_colors[name],
                     edgecolor='#1a1a2e', linewidth=2)
        ax4.add_patch(wedge)
    
    # Center text
    ax4.text(0.5, 0.45, f'{total_ddis:,}', ha='center', va='center',
            fontsize=18, fontweight='bold', color='white')
    ax4.text(0.5, 0.38, 'DDI Pairs', ha='center', va='center',
            fontsize=10, color='#888888')
    
    # Legend below
    legend_y = 0.02
    legend_x = 0.1
    for i, name in enumerate(['Contraindicated', 'Major', 'Moderate', 'Minor']):
        count = stats['severity'][name]
        pct = (count / total_ddis) * 100
        
        rect = FancyBboxPatch((legend_x + i * 0.22, legend_y), 0.03, 0.03,
                              boxstyle="round,pad=0.005",
                              facecolor=severity_colors[name], edgecolor='white')
        ax4.add_patch(rect)
        ax4.text(legend_x + i * 0.22 + 0.04, legend_y + 0.015,
                f'{name}\n{pct:.1f}%', ha='left', va='center',
                fontsize=7, color='white')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 0.9)
    ax4.set_title('DDI Severity Distribution', fontsize=16, fontweight='bold', 
                  color='white', pad=10)
    ax4.axis('off')
    
    # =========================================================================
    # PANEL 5: Key Metrics - Infographic Cards (Bottom Middle)
    # =========================================================================
    ax5 = fig.add_axes([0.36, 0.05, 0.28, 0.38])
    ax5.set_facecolor('#1a1a2e')
    
    metrics = [
        ('4,313', 'Unique Drugs', '#e94560'),
        ('759,774', 'DDI Pairs', '#4ecca3'),
        ('5,548', 'Side Effects', '#45b7d1'),
        ('3,041', 'Diseases', '#ff6b6b'),
        ('3,176', 'Proteins', '#96ceb4'),
        ('25,958', 'Pathways', '#ffeaa7'),
    ]
    
    for i, (value, label, color) in enumerate(metrics):
        row = i // 2
        col = i % 2
        x = 0.1 + col * 0.45
        y = 0.72 - row * 0.28
        
        # Card background
        card = FancyBboxPatch((x - 0.08, y - 0.08), 0.38, 0.22,
                              boxstyle="round,pad=0.02,rounding_size=0.03",
                              facecolor='#16213e', edgecolor=color,
                              linewidth=2, alpha=0.9)
        ax5.add_patch(card)
        
        # Icon circle
        icon = Circle((x, y + 0.02), 0.04, facecolor=color, alpha=0.3)
        ax5.add_patch(icon)
        
        # Value and label
        ax5.text(x + 0.12, y + 0.04, value, ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)
        ax5.text(x + 0.12, y - 0.04, label, ha='center', va='center',
                fontsize=9, color='white', alpha=0.8)
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('Key Metrics', fontsize=16, fontweight='bold', color='white', pad=10)
    ax5.axis('off')
    
    # =========================================================================
    # PANEL 6: Class Interaction Summary - Matrix Dots (Bottom Right)
    # =========================================================================
    ax6 = fig.add_axes([0.66, 0.05, 0.32, 0.38])
    ax6.set_facecolor('#1a1a2e')
    
    # Top severity class pairs
    class_data = [
        ('Blood↔Musc.', 83.3, 8976),
        ('Antineo↔Blood', 65.2, 18748),
        ('Blood↔Blood', 51.1, 7346),
        ('Blood↔CNS', 33.6, 15448),
        ('Anti-inf↔Blood', 20.3, 13218),
    ]
    
    ax6.text(0.5, 0.92, 'High-Risk Class Pairs', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax6.text(0.5, 0.85, '(% Severe DDIs)', ha='center', va='center',
            fontsize=9, color='#888888')
    
    for i, (name, pct, count) in enumerate(class_data):
        y = 0.72 - i * 0.15
        
        # Progress bar background
        bar_bg = FancyBboxPatch((0.35, y - 0.03), 0.55, 0.08,
                                boxstyle="round,pad=0.005,rounding_size=0.01",
                                facecolor='#16213e', edgecolor='none')
        ax6.add_patch(bar_bg)
        
        # Progress bar fill
        fill_width = (pct / 100) * 0.55
        color = '#d62728' if pct > 50 else '#ff7f0e' if pct > 20 else '#2ca02c'
        bar_fill = FancyBboxPatch((0.35, y - 0.03), fill_width, 0.08,
                                  boxstyle="round,pad=0.005,rounding_size=0.01",
                                  facecolor=color, edgecolor='none', alpha=0.8)
        ax6.add_patch(bar_fill)
        
        # Labels
        ax6.text(0.05, y, name, ha='left', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax6.text(0.92, y, f'{pct:.1f}%', ha='right', va='center',
                fontsize=10, fontweight='bold', color=color)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('Severity by Drug Class', fontsize=16, fontweight='bold', 
                  color='white', pad=10)
    ax6.axis('off')
    
    # =========================================================================
    # Save
    # =========================================================================
    output_path = f'{OUTPUT_DIR}/kg_comprehensive_stats.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    
    print(f"Saved comprehensive stats figure to {output_path}")
    return output_path


if __name__ == '__main__':
    print("=" * 60)
    print("Generating Comprehensive Knowledge Graph Statistics Figure")
    print("=" * 60)
    create_comprehensive_stats_figure()
    print("\nDone!")
