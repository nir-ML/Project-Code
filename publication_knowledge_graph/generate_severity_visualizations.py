#!/usr/bin/env python3
"""
Generate Publication-Grade Visualizations for DDI Knowledge Graph
- Chord Diagram with Severity Coloring
- Class-Based Node-Link Diagram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import ast
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = '/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/publication_knowledge_graph/figures'

def load_and_process_data():
    """Load DDI data and compute class-level statistics"""
    df = pd.read_csv('/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/data/ddi_recalibrated.csv')
    
    atc_classes = {
        'A': 'Alimentary', 'B': 'Blood', 'C': 'Cardiovascular', 'D': 'Dermatologicals',
        'G': 'Genitourinary', 'H': 'Hormones', 'J': 'Anti-infectives', 'L': 'Antineoplastic',
        'M': 'Musculoskeletal', 'N': 'Nervous System', 'P': 'Antiparasitic', 'R': 'Respiratory',
        'S': 'Sensory Organs', 'V': 'Various'
    }
    
    # Map drugs to classes
    drug_to_class = {}
    for suffix in ['1', '2']:
        for _, row in df.iterrows():
            atc_str = row.get(f'atc_{suffix}', '')
            drug_id = row.get(f'drugbank_id_{suffix}', '')
            if pd.notna(atc_str) and isinstance(atc_str, str) and len(atc_str) > 2:
                try:
                    atc_list = ast.literal_eval(atc_str)
                    if isinstance(atc_list, list) and len(atc_list) > 0:
                        first_letter = atc_list[0][0].upper()
                        if first_letter in atc_classes:
                            drug_to_class[drug_id] = atc_classes[first_letter]
                except:
                    pass
    
    # Compute class interactions with severity
    class_total = defaultdict(int)
    class_severe = defaultdict(int)
    
    for _, row in df.iterrows():
        c1 = drug_to_class.get(row['drugbank_id_1'], 'Unknown')
        c2 = drug_to_class.get(row['drugbank_id_2'], 'Unknown')
        if c1 != 'Unknown' and c2 != 'Unknown':
            key = tuple(sorted([c1, c2]))
            class_total[key] += 1
            sev = row.get('severity_recalibrated', '')
            if 'Contraindicated' in str(sev) or 'Major' in str(sev):
                class_severe[key] += 1
    
    # Compute severity rates
    class_severity_rate = {}
    for key, total in class_total.items():
        class_severity_rate[key] = (class_severe[key] / total * 100) if total > 0 else 0
    
    return class_total, class_severe, class_severity_rate, atc_classes


def generate_chord_with_severity(class_total, class_severe, class_severity_rate, atc_classes, output_path):
    """Generate chord diagram with severity-based coloring"""
    print("Generating Chord Diagram with Severity...")
    
    # Get classes with significant interactions
    class_totals = defaultdict(int)
    for (c1, c2), count in class_total.items():
        if count > 500:
            class_totals[c1] += count
            class_totals[c2] += count
    
    classes = sorted([c for c, t in class_totals.items() if t > 5000])
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Build matrix
    matrix = np.zeros((n_classes, n_classes))
    severity_matrix = np.zeros((n_classes, n_classes))
    
    for (c1, c2), count in class_total.items():
        if c1 in class_to_idx and c2 in class_to_idx:
            i, j = class_to_idx[c1], class_to_idx[c2]
            matrix[i, j] = count
            matrix[j, i] = count
            sev_rate = class_severity_rate.get((c1, c2), class_severity_rate.get((c2, c1), 0))
            severity_matrix[i, j] = sev_rate
            severity_matrix[j, i] = sev_rate
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    
    # Class colors (muted palette)
    class_colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    # Calculate arc sizes
    totals = matrix.sum(axis=1)
    total_sum = totals.sum()
    
    gap = 0.03
    available = 2 * np.pi - gap * n_classes
    arc_sizes = (totals / total_sum) * available
    
    arc_starts = np.zeros(n_classes)
    current = 0
    for i in range(n_classes):
        arc_starts[i] = current
        current += arc_sizes[i] + gap
    
    # Draw outer arcs
    for i, cls in enumerate(classes):
        theta1 = arc_starts[i]
        theta2 = theta1 + arc_sizes[i]
        angles = np.linspace(theta1, theta2, 50)
        
        ax.fill_between(angles, 0.88, 1.0, color=class_colors[i], alpha=0.9)
        ax.plot(angles, [0.88] * 50, color='black', linewidth=0.5)
        ax.plot(angles, [1.0] * 50, color='black', linewidth=0.5)
        
        # Label
        mid_angle = (theta1 + theta2) / 2
        rotation = np.degrees(mid_angle) - 90
        ha = 'left'
        if mid_angle > np.pi/2 and mid_angle < 3*np.pi/2:
            rotation += 180
            ha = 'right'
        
        ax.text(mid_angle, 1.08, cls, ha=ha, va='center', rotation=rotation,
               fontsize=9, fontweight='bold')
    
    # Draw chords with severity coloring
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            if matrix[i, j] > 1000:
                # Severity-based color
                sev_rate = severity_matrix[i, j]
                if sev_rate > 50:
                    color = '#d62728'  # Red - very high severity
                    alpha = 0.8
                elif sev_rate > 30:
                    color = '#ff7f0e'  # Orange - high severity
                    alpha = 0.7
                elif sev_rate > 15:
                    color = '#ffbb78'  # Light orange - moderate severity
                    alpha = 0.6
                elif sev_rate > 8:
                    color = '#98df8a'  # Light green - low severity
                    alpha = 0.5
                else:
                    color = '#2ca02c'  # Green - very low severity
                    alpha = 0.4
                
                # Width based on interaction count
                strength = matrix[i, j] / matrix.max()
                
                # Chord positions
                src_mid = arc_starts[i] + arc_sizes[i] / 2
                tgt_mid = arc_starts[j] + arc_sizes[j] / 2
                
                # Draw curved chord
                angles = np.linspace(src_mid, tgt_mid, 50)
                radii = 0.85 - 0.45 * np.sin(np.linspace(0, np.pi, 50))
                
                linewidth = 1 + strength * 8
                ax.plot(angles, radii, color=color, alpha=alpha, linewidth=linewidth)
    
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    
    # Title
    ax.set_title('Drug Class Interaction Chord Diagram\n(Chord Color = Severity Rate)', 
                 fontsize=14, fontweight='bold', pad=20, y=1.05)
    
    # Severity legend
    legend_elements = [
        Line2D([0], [0], color='#d62728', linewidth=4, label='>50% Severe'),
        Line2D([0], [0], color='#ff7f0e', linewidth=4, label='30-50% Severe'),
        Line2D([0], [0], color='#ffbb78', linewidth=4, label='15-30% Severe'),
        Line2D([0], [0], color='#98df8a', linewidth=4, label='8-15% Severe'),
        Line2D([0], [0], color='#2ca02c', linewidth=4, label='<8% Severe'),
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=9, 
              title='Severity Rate\n(Major + Contraindicated)', title_fontsize=10,
              frameon=True, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved to {output_path}")


def generate_class_nodelink(class_total, class_severe, class_severity_rate, atc_classes, output_path):
    """Generate class-based node-link diagram"""
    print("Generating Class-Based Node-Link Diagram...")
    
    # Get class totals
    class_totals = defaultdict(int)
    class_severe_totals = defaultdict(int)
    
    for (c1, c2), count in class_total.items():
        class_totals[c1] += count
        class_totals[c2] += count
        class_severe_totals[c1] += class_severe[(c1, c2)]
        class_severe_totals[c2] += class_severe[(c1, c2)]
    
    # Filter to classes with enough interactions
    classes = [c for c, t in class_totals.items() if t > 5000]
    n_classes = len(classes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='white')
    
    # Circular layout
    angles = np.linspace(0, 2*np.pi, n_classes, endpoint=False)
    radius = 0.35
    positions = {cls: (0.5 + radius * np.cos(a - np.pi/2), 
                       0.5 + radius * np.sin(a - np.pi/2)) 
                 for cls, a in zip(classes, angles)}
    
    # Draw edges first (behind nodes)
    max_count = max(class_total.values())
    
    for (c1, c2), count in class_total.items():
        if c1 in positions and c2 in positions and count > 2000:
            x1, y1 = positions[c1]
            x2, y2 = positions[c2]
            
            # Severity-based color
            sev_rate = class_severity_rate.get((c1, c2), 0)
            if sev_rate > 50:
                color = '#d62728'
                zorder = 3
            elif sev_rate > 30:
                color = '#ff7f0e'
                zorder = 2
            elif sev_rate > 15:
                color = '#fdae61'
                zorder = 1
            else:
                color = '#91cf60'
                zorder = 0
            
            # Width based on count
            width = 0.5 + (count / max_count) * 6
            alpha = 0.3 + (sev_rate / 100) * 0.5
            
            # Curved edge through center
            mid_x = 0.5 + (x1 + x2 - 1) * 0.3
            mid_y = 0.5 + (y1 + y2 - 1) * 0.3
            
            # Bezier-like curve
            t = np.linspace(0, 1, 50)
            curve_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
            curve_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
            
            ax.plot(curve_x, curve_y, color=color, linewidth=width, alpha=alpha, 
                   solid_capstyle='round', zorder=zorder)
    
    # Draw nodes
    max_total = max(class_totals.values())
    
    # Color by overall severity rate of class
    for cls in classes:
        x, y = positions[cls]
        total = class_totals[cls]
        severe = class_severe_totals[cls]
        sev_rate = (severe / total * 100) if total > 0 else 0
        
        # Node size based on total interactions
        size = 1500 + (total / max_total) * 4000
        
        # Color based on severity rate
        if sev_rate > 25:
            color = '#d62728'
        elif sev_rate > 15:
            color = '#ff7f0e'
        elif sev_rate > 10:
            color = '#ffbb78'
        else:
            color = '#2ca02c'
        
        ax.scatter(x, y, s=size, c=color, edgecolors='black', linewidths=2, zorder=10)
        
        # Label
        ax.text(x, y + 0.01, cls, ha='center', va='center', fontsize=9, 
               fontweight='bold', zorder=11)
        ax.text(x, y - 0.035, f'{total//1000}K DDIs\n({sev_rate:.0f}% sev)', 
               ha='center', va='center', fontsize=7, zorder=11)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', 
               markersize=15, label='>25% Severe', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
               markersize=15, label='15-25% Severe', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffbb78', 
               markersize=15, label='10-15% Severe', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', 
               markersize=15, label='<10% Severe', markeredgecolor='black'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             title='Node Color: Class Severity', title_fontsize=11, framealpha=0.9)
    
    # Edge legend
    edge_legend = [
        Line2D([0], [0], color='#d62728', linewidth=4, label='>50% Severe'),
        Line2D([0], [0], color='#ff7f0e', linewidth=3, label='30-50%'),
        Line2D([0], [0], color='#fdae61', linewidth=2, label='15-30%'),
        Line2D([0], [0], color='#91cf60', linewidth=1.5, label='<15%'),
    ]
    
    leg2 = ax.legend(handles=edge_legend, loc='upper right', fontsize=10,
                    title='Edge Color: Pair Severity', title_fontsize=11, framealpha=0.9)
    ax.add_artist(ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
                           title='Node Color: Class Severity', title_fontsize=11, framealpha=0.9))
    
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('Drug Class Interaction Network\n(Node/Edge Color = Severity Rate)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved to {output_path}")


def main():
    print("=" * 60)
    print("Generating Publication-Grade Visualizations")
    print("=" * 60)
    
    class_total, class_severe, class_severity_rate, atc_classes = load_and_process_data()
    
    generate_chord_with_severity(class_total, class_severe, class_severity_rate, atc_classes,
                                  f'{OUTPUT_DIR}/chord_severity.png')
    
    generate_class_nodelink(class_total, class_severe, class_severity_rate, atc_classes,
                            f'{OUTPUT_DIR}/nodelink_class_severity.png')
    
    print("\nDone!")


if __name__ == '__main__':
    main()
