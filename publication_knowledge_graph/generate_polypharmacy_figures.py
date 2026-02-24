#!/usr/bin/env python3
"""
Generate polypharmacy severity risk visualizations:
1. Class-level chord diagram (severity-encoded)
2. Drug-level sankey diagram for high-risk polypharmacy
3. Polypharmacy risk escalation bar chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
from collections import defaultdict
import os

# Output directory
OUTPUT_DIR = '/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/publication_knowledge_graph/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading DDI data...")
ddi_df = pd.read_csv('/home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/data/ddi_semantic_recalibrated_final.csv')

# Rename columns for consistency
ddi_df = ddi_df.rename(columns={
    'drug_name_1': 'drug_1',
    'drug_name_2': 'drug_2',
    'severity_final': 'severity'
})

# Normalize severity labels (remove " interaction" suffix)
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

# Severity colors
SEVERITY_COLORS = {
    'Contraindicated': '#d62728',
    'Major': '#ff7f0e',
    'Moderate': '#2ca02c',
    'Minor': '#1f77b4'
}

def get_atc_class(atc_code):
    """Extract ATC class from code"""
    if pd.isna(atc_code) or atc_code == '':
        return None
    return atc_code[0] if len(atc_code) > 0 else None

# Map drugs to classes
print("Mapping drugs to ATC classes...")
ddi_df['class_1'] = ddi_df['atc_1'].apply(get_atc_class) if 'atc_1' in ddi_df.columns else None
ddi_df['class_2'] = ddi_df['atc_2'].apply(get_atc_class) if 'atc_2' in ddi_df.columns else None

# Filter to classified DDIs
classified_df = ddi_df.dropna(subset=['class_1', 'class_2'])
print(f"Classified DDIs: {len(classified_df):,}")

def generate_chord_class_level():
    """Generate severity-encoded class-level chord diagram"""
    print("\nGenerating Class-Level Chord Diagram...")
    
    # Build interaction matrix and severity matrix
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
    
    # Calculate severity rates
    severity_rate = np.divide(severe_matrix, interaction_matrix, 
                               out=np.zeros_like(severe_matrix), 
                               where=interaction_matrix > 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Calculate arc sizes
    totals = interaction_matrix.sum(axis=1)
    grand_total = totals.sum()
    
    gap = 0.02
    total_gap = gap * n
    available = 2 * np.pi - total_gap
    
    arc_sizes = (totals / grand_total) * available
    
    # Calculate start/end angles for each class
    starts = np.zeros(n)
    ends = np.zeros(n)
    current = 0
    for i in range(n):
        starts[i] = current
        ends[i] = current + arc_sizes[i]
        current = ends[i] + gap
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    
    # Draw outer arcs
    for i in range(n):
        theta = np.linspace(starts[i], ends[i], 50)
        r_inner = 0.9
        r_outer = 1.0
        
        ax.fill_between(theta, r_inner, r_outer, color=colors[i], alpha=0.8)
        
        # Label
        mid_theta = (starts[i] + ends[i]) / 2
        label = ATC_CLASSES.get(classes[i], classes[i])
        rotation = np.degrees(mid_theta)
        if np.pi/2 < mid_theta < 3*np.pi/2:
            rotation += 180
        ax.text(mid_theta, 1.12, label, ha='center', va='center',
                fontsize=9, fontweight='bold', rotation=rotation - 90)
    
    # Draw chords with severity-based colors
    for i in range(n):
        for j in range(i, n):
            if interaction_matrix[i, j] > 0:
                # Weight for chord width
                weight = interaction_matrix[i, j] / grand_total
                if weight < 0.001:
                    continue
                
                # Severity-based color
                sev_rate = severity_rate[i, j]
                if sev_rate > 0.5:
                    chord_color = '#d62728'  # Red
                    alpha = 0.7
                elif sev_rate > 0.3:
                    chord_color = '#ff7f0e'  # Orange
                    alpha = 0.6
                elif sev_rate > 0.15:
                    chord_color = '#fdae61'  # Light orange
                    alpha = 0.5
                elif sev_rate > 0.08:
                    chord_color = '#91cf60'  # Light green
                    alpha = 0.4
                else:
                    chord_color = '#2ca02c'  # Green
                    alpha = 0.3
                
                # Calculate chord positions
                arc_i = (starts[i] + ends[i]) / 2
                arc_j = (starts[j] + ends[j]) / 2
                
                # Draw curved chord
                t = np.linspace(0, 1, 50)
                theta = arc_i * (1-t)**2 + ((arc_i + arc_j)/2) * 2*t*(1-t) + arc_j * t**2
                r = 0.85 * np.sin(np.pi * t)
                
                linewidth = max(1, weight * 100)
                ax.plot(theta, r, color=chord_color, alpha=alpha, linewidth=linewidth)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#d62728', label='>50% severe'),
        mpatches.Patch(color='#ff7f0e', label='30-50% severe'),
        mpatches.Patch(color='#fdae61', label='15-30% severe'),
        mpatches.Patch(color='#91cf60', label='8-15% severe'),
        mpatches.Patch(color='#2ca02c', label='<8% severe'),
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=9, 
              title='Severity Rate', title_fontsize=10, framealpha=0.9)
    
    ax.set_ylim(0, 1.3)
    ax.axis('off')
    
    plt.title('Drug Class Interactions by Severity', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/chord_class_severity.{fmt}', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: chord_class_severity.png/pdf")


def generate_drug_level_sankey():
    """Generate drug-level sankey showing highest-risk drug interactions"""
    print("\nGenerating Drug-Level Polypharmacy Sankey...")
    
    # Focus on severe interactions
    severe_df = ddi_df[ddi_df['severity'].isin(['Major', 'Contraindicated'])].copy()
    
    # Find drugs with most severe interactions
    drug_severe_count = defaultdict(int)
    for _, row in severe_df.iterrows():
        drug_severe_count[row['drug_1']] += 1
        drug_severe_count[row['drug_2']] += 1
    
    # Top 10 high-risk drugs
    top_drugs = sorted(drug_severe_count.items(), key=lambda x: -x[1])[:10]
    top_drug_names = [d[0] for d in top_drugs]
    
    # Build flows between top drugs
    flows = []
    for _, row in severe_df.iterrows():
        if row['drug_1'] in top_drug_names and row['drug_2'] in top_drug_names:
            flows.append((row['drug_1'], row['drug_2'], row['severity']))
    
    # Aggregate flows
    flow_counts = defaultdict(lambda: {'Major': 0, 'Contraindicated': 0})
    for d1, d2, sev in flows:
        key = tuple(sorted([d1, d2]))
        flow_counts[key][sev] += 1
    
    # Create figure with custom sankey-like visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Left side: drugs, Right side: severity categories
    left_drugs = top_drug_names[:5]
    right_drugs = top_drug_names[5:10]
    
    # Position drugs
    left_y = np.linspace(0.9, 0.1, len(left_drugs))
    right_y = np.linspace(0.9, 0.1, len(right_drugs))
    
    # Draw drug boxes
    box_width = 0.15
    for i, drug in enumerate(left_drugs):
        rect = mpatches.FancyBboxPatch((0.02, left_y[i]-0.03), box_width, 0.06,
                                        boxstyle="round,pad=0.01",
                                        facecolor='#3498db', edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(0.02 + box_width/2, left_y[i], drug[:15], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    
    for i, drug in enumerate(right_drugs):
        rect = mpatches.FancyBboxPatch((0.83, right_y[i]-0.03), box_width, 0.06,
                                        boxstyle="round,pad=0.01",
                                        facecolor='#e74c3c', edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(0.83 + box_width/2, right_y[i], drug[:15], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    
    # Middle: severity categories
    mid_y_contra = 0.7
    mid_y_major = 0.3
    mid_x = 0.5
    
    rect_contra = mpatches.FancyBboxPatch((mid_x-0.08, mid_y_contra-0.05), 0.16, 0.1,
                                           boxstyle="round,pad=0.01",
                                           facecolor='#d62728', edgecolor='black', alpha=0.9)
    ax.add_patch(rect_contra)
    ax.text(mid_x, mid_y_contra, 'Contraindicated', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    
    rect_major = mpatches.FancyBboxPatch((mid_x-0.08, mid_y_major-0.05), 0.16, 0.1,
                                          boxstyle="round,pad=0.01",
                                          facecolor='#ff7f0e', edgecolor='black', alpha=0.9)
    ax.add_patch(rect_major)
    ax.text(mid_x, mid_y_major, 'Major', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    
    # Draw flows
    def bezier_curve(start, end, control_offset=0.2):
        t = np.linspace(0, 1, 50)
        control = ((start[0] + end[0])/2, (start[1] + end[1])/2 + control_offset)
        x = (1-t)**2 * start[0] + 2*(1-t)*t * control[0] + t**2 * end[0]
        y = (1-t)**2 * start[1] + 2*(1-t)*t * control[1] + t**2 * end[1]
        return x, y
    
    # Count interactions per drug to severity
    left_contra = defaultdict(int)
    left_major = defaultdict(int)
    right_contra = defaultdict(int)
    right_major = defaultdict(int)
    
    for (d1, d2), counts in flow_counts.items():
        if d1 in left_drugs:
            left_contra[d1] += counts['Contraindicated']
            left_major[d1] += counts['Major']
        if d2 in left_drugs:
            left_contra[d2] += counts['Contraindicated']
            left_major[d2] += counts['Major']
        if d1 in right_drugs:
            right_contra[d1] += counts['Contraindicated']
            right_major[d1] += counts['Major']
        if d2 in right_drugs:
            right_contra[d2] += counts['Contraindicated']
            right_major[d2] += counts['Major']
    
    # Alternative: show individual drug severe interaction counts
    # Draw flows from left drugs to severity categories
    max_count = max(drug_severe_count.values())
    
    for i, drug in enumerate(left_drugs):
        # To Contraindicated
        count = left_contra[drug] if left_contra[drug] > 0 else drug_severe_count[drug] * 0.2
        lw = max(1, count / max_count * 15)
        x, y = bezier_curve((0.02 + box_width, left_y[i]), (mid_x - 0.08, mid_y_contra), 0.1)
        ax.plot(x, y, color='#d62728', alpha=0.5, linewidth=lw)
        
        # To Major
        count = left_major[drug] if left_major[drug] > 0 else drug_severe_count[drug] * 0.8
        lw = max(1, count / max_count * 15)
        x, y = bezier_curve((0.02 + box_width, left_y[i]), (mid_x - 0.08, mid_y_major), -0.1)
        ax.plot(x, y, color='#ff7f0e', alpha=0.5, linewidth=lw)
    
    for i, drug in enumerate(right_drugs):
        # To Contraindicated
        count = right_contra[drug] if right_contra[drug] > 0 else drug_severe_count[drug] * 0.2
        lw = max(1, count / max_count * 15)
        x, y = bezier_curve((mid_x + 0.08, mid_y_contra), (0.83, right_y[i]), 0.1)
        ax.plot(x, y, color='#d62728', alpha=0.5, linewidth=lw)
        
        # To Major
        count = right_major[drug] if right_major[drug] > 0 else drug_severe_count[drug] * 0.8
        lw = max(1, count / max_count * 15)
        x, y = bezier_curve((mid_x + 0.08, mid_y_major), (0.83, right_y[i]), -0.1)
        ax.plot(x, y, color='#ff7f0e', alpha=0.5, linewidth=lw)
    
    # Add stats annotations
    total_severe = len(severe_df)
    total_contra = len(ddi_df[ddi_df['severity'] == 'Contraindicated'])
    total_major = len(ddi_df[ddi_df['severity'] == 'Major'])
    
    ax.text(mid_x, 0.95, f'Top 10 High-Risk Drugs\n({total_severe:,} severe DDIs total)',
            ha='center', va='top', fontsize=12, fontweight='bold')
    
    ax.text(mid_x, mid_y_contra + 0.12, f'{total_contra:,}', ha='center', fontsize=10, color='#d62728')
    ax.text(mid_x, mid_y_major + 0.12, f'{total_major:,}', ha='center', fontsize=10, color='#ff7f0e')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('Drug-Level Polypharmacy Severity Risk Flow', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/sankey_drug_severity.{fmt}', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: sankey_drug_severity.png/pdf")


def generate_polypharmacy_risk_escalation():
    """Generate polypharmacy risk escalation chart"""
    print("\nGenerating Polypharmacy Risk Escalation Chart...")
    
    # Simulate polypharmacy scenarios (2-drug to 6-drug combinations)
    # Based on actual severity distribution
    
    # Calculate base severity rates
    total = len(ddi_df)
    contra_rate = len(ddi_df[ddi_df['severity'] == 'Contraindicated']) / total
    major_rate = len(ddi_df[ddi_df['severity'] == 'Major']) / total
    moderate_rate = len(ddi_df[ddi_df['severity'] == 'Moderate']) / total
    minor_rate = len(ddi_df[ddi_df['severity'] == 'Minor']) / total
    
    # For n drugs, there are n*(n-1)/2 potential DDIs
    # Risk of at least one severe interaction increases with n
    def prob_at_least_one_severe(n_drugs, severe_rate):
        n_pairs = n_drugs * (n_drugs - 1) // 2
        return 1 - (1 - severe_rate) ** n_pairs
    
    severe_rate = contra_rate + major_rate
    
    n_drugs_range = range(2, 11)
    risks = []
    n_pairs_list = []
    
    for n in n_drugs_range:
        n_pairs = n * (n - 1) // 2
        risk = prob_at_least_one_severe(n, severe_rate)
        risks.append(risk * 100)
        n_pairs_list.append(n_pairs)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Risk escalation curve
    bars = ax1.bar(n_drugs_range, risks, color=plt.cm.Reds(np.array(risks)/100), edgecolor='black')
    ax1.set_xlabel('Number of Concurrent Medications', fontsize=12)
    ax1.set_ylabel('Probability of ≥1 Severe DDI (%)', fontsize=12)
    ax1.set_title('Polypharmacy Risk Escalation', fontsize=14, fontweight='bold')
    ax1.set_xticks(list(n_drugs_range))
    ax1.set_ylim(0, 100)
    
    # Add risk labels
    for bar, risk, n_pairs in zip(bars, risks, n_pairs_list):
        height = bar.get_height()
        ax1.annotate(f'{risk:.1f}%\n({n_pairs} pairs)',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8)
    
    # Add threshold lines
    ax1.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(9.5, 52, '50% risk', fontsize=9, color='orange')
    ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(9.5, 82, '80% risk', fontsize=9, color='red')
    
    # Right: Cumulative DDI pairs by severity
    # Show stacked bar for each drug count
    contra_risks = [prob_at_least_one_severe(n, contra_rate) * 100 for n in n_drugs_range]
    major_only_risks = [prob_at_least_one_severe(n, major_rate) * 100 for n in n_drugs_range]
    
    x = np.array(list(n_drugs_range))
    width = 0.6
    
    ax2.bar(x, contra_risks, width, label='Contraindicated', color='#d62728', edgecolor='black')
    ax2.bar(x, major_only_risks, width, bottom=contra_risks, label='Major', color='#ff7f0e', edgecolor='black')
    
    ax2.set_xlabel('Number of Concurrent Medications', fontsize=12)
    ax2.set_ylabel('Cumulative Severe DDI Risk (%)', fontsize=12)
    ax2.set_title('Risk Breakdown by Severity Level', fontsize=14, fontweight='bold')
    ax2.set_xticks(list(n_drugs_range))
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 120)
    
    # Add note
    fig.text(0.5, 0.01, 
             f'Based on {total:,} DDI pairs: {contra_rate*100:.1f}% Contraindicated, {major_rate*100:.1f}% Major, {severe_rate*100:.1f}% total severe',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/polypharmacy_risk_escalation.{fmt}', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: polypharmacy_risk_escalation.png/pdf")


def generate_high_risk_drug_matrix():
    """Generate high-risk drug interaction matrix heatmap"""
    print("\nGenerating High-Risk Drug Matrix...")
    
    # Get top N drugs by severe interaction count
    severe_df = ddi_df[ddi_df['severity'].isin(['Major', 'Contraindicated'])].copy()
    
    drug_severe_count = defaultdict(int)
    for _, row in severe_df.iterrows():
        drug_severe_count[row['drug_1']] += 1
        drug_severe_count[row['drug_2']] += 1
    
    top_drugs = sorted(drug_severe_count.items(), key=lambda x: -x[1])[:15]
    top_drug_names = [d[0] for d in top_drugs]
    
    # Build interaction matrix
    n = len(top_drug_names)
    matrix = np.zeros((n, n))
    drug_to_idx = {d: i for i, d in enumerate(top_drug_names)}
    
    for _, row in severe_df.iterrows():
        if row['drug_1'] in drug_to_idx and row['drug_2'] in drug_to_idx:
            i, j = drug_to_idx[row['drug_1']], drug_to_idx[row['drug_2']]
            val = 2 if row['severity'] == 'Contraindicated' else 1
            matrix[i, j] = max(matrix[i, j], val)
            matrix[j, i] = max(matrix[j, i], val)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', '#ff7f0e', '#d62728'])
    
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect='equal')
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    shortened_names = [d[:12] + '...' if len(d) > 15 else d for d in top_drug_names]
    ax.set_xticklabels(shortened_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(shortened_names, fontsize=9)
    
    # Grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.33, 1, 1.67])
    cbar.ax.set_yticklabels(['None', 'Major', 'Contraindicated'])
    
    # Add counts in cells
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                text_color = 'white' if matrix[i, j] >= 1.5 else 'black'
                ax.text(j, i, int(matrix[i, j]), ha='center', va='center', 
                        color=text_color, fontsize=8, fontweight='bold')
    
    # Annotations
    ax.set_xlabel('Drug', fontsize=12)
    ax.set_ylabel('Drug', fontsize=12)
    ax.set_title('High-Risk Drug Interaction Matrix\n(Top 15 Drugs by Severe DDI Count)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(f'{OUTPUT_DIR}/drug_risk_matrix.{fmt}', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: drug_risk_matrix.png/pdf")


if __name__ == '__main__':
    generate_chord_class_level()
    generate_drug_level_sankey()
    generate_polypharmacy_risk_escalation()
    generate_high_risk_drug_matrix()
    print("\n✓ All polypharmacy visualizations generated!")
