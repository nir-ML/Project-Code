"""
FAERS Validation Runner

Executes complete validation of network-based risk scores against FDA FAERS data.
Validates that network topology metrics correlate with real-world adverse event patterns.

Usage:
    python modules/run_faers_validation.py [--sample-size N] [--output DIR]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.faers_integration import FAERSValidator, FAERSClient


def load_drug_risk_data(data_path: str = None) -> pd.DataFrame:
    """Load drug risk data with PRI scores"""
    if data_path is None:
        # Try multiple locations
        possible_paths = [
            "publication/data/all_drugs_with_metrics.csv",
            "publication/data/pri_scores_detailed.csv",
            "data/ddi_cardio_or_antithrombotic_labeled (1).csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
    
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError("No drug risk data file found")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    return df


def prepare_validation_sample(df: pd.DataFrame, sample_size: int = 100) -> List[Tuple[str, float]]:
    """
    Prepare a representative sample of drugs for validation
    
    Strategy: Sample across PRI score distribution to capture full range
    """
    # Check for required columns
    if 'drug_name' in df.columns:
        name_col = 'drug_name'
    elif 'Drug' in df.columns:
        name_col = 'Drug'
    elif 'drug1_name' in df.columns:
        name_col = 'drug1_name'
    else:
        raise ValueError("Could not find drug name column")
    
    # Check for PRI column
    if 'pri_score' in df.columns:
        pri_col = 'pri_score'
    elif 'PRI' in df.columns:
        pri_col = 'PRI'
    elif 'degree_centrality' in df.columns:
        pri_col = 'degree_centrality'  # Use centrality as proxy
    else:
        # Calculate simple risk proxy
        if 'Severity' in df.columns:
            severity_map = {'Major': 3, 'Moderate': 2, 'Minor': 1}
            df['_pri_proxy'] = df['Severity'].map(severity_map).fillna(1)
            pri_col = '_pri_proxy'
        else:
            df['_pri_proxy'] = 1  # Uniform
            pri_col = '_pri_proxy'
            print("Warning: No risk score column found, using uniform weights")
    
    # Get unique drugs with their risk scores
    unique_drugs = df.groupby(name_col)[pri_col].mean().reset_index()
    unique_drugs.columns = ['drug_name', 'pri_score']
    
    # Stratified sampling across PRI distribution
    unique_drugs['PRI_quintile'] = pd.qcut(unique_drugs['pri_score'], 
                                           q=5, labels=False, duplicates='drop')
    
    sample_per_quintile = max(1, sample_size // 5)
    sampled = []
    
    for q in range(5):
        quintile_drugs = unique_drugs[unique_drugs['PRI_quintile'] == q]
        n_sample = min(sample_per_quintile, len(quintile_drugs))
        sampled.append(quintile_drugs.sample(n=n_sample, random_state=42))
    
    sample_df = pd.concat(sampled, ignore_index=True)
    
    # Fill remaining if needed
    if len(sample_df) < sample_size:
        remaining = unique_drugs[~unique_drugs['drug_name'].isin(sample_df['drug_name'])]
        n_more = min(sample_size - len(sample_df), len(remaining))
        if n_more > 0:
            sample_df = pd.concat([
                sample_df,
                remaining.sample(n=n_more, random_state=42)
            ], ignore_index=True)
    
    return list(zip(sample_df['drug_name'], sample_df['pri_score']))


def run_validation(drugs_with_pri: List[Tuple[str, float]], 
                  output_dir: str = "validation_results") -> Dict[str, Any]:
    """Run complete FAERS validation"""
    
    os.makedirs(output_dir, exist_ok=True)
    validator = FAERSValidator()
    
    print(f"\nStarting FAERS validation for {len(drugs_with_pri)} drugs...")
    print("=" * 60)
    
    # Progress callback
    def progress(current, total, drug):
        pct = current / total * 100
        print(f"  [{current}/{total}] ({pct:.0f}%) Validating: {drug[:30]}...")
    
    # Run validation
    validation_results = validator.batch_validate_drugs(drugs_with_pri, progress)
    
    # Calculate correlations
    print("\nCalculating correlations...")
    correlations = validator.calculate_correlation(validation_results)
    
    # Summary statistics
    successful = [v for v in validation_results if v["validation_status"] == "success"]
    failed = [v for v in validation_results if v["validation_status"] == "failed"]
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_drugs": len(drugs_with_pri),
        "successful_validations": len(successful),
        "failed_validations": len(failed),
        "success_rate": len(successful) / len(drugs_with_pri) if drugs_with_pri else 0,
        "correlations": correlations,
        "high_risk_drugs": [
            {"drug": v["drug_name"], "network_pri": v["network_pri"], 
             "faers_score": v["faers_risk_score"], "serious_ratio": v["serious_event_ratio"]}
            for v in sorted(successful, key=lambda x: x["faers_risk_score"], reverse=True)[:10]
        ]
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total drugs validated: {summary['total_drugs']}")
    print(f"Successful: {summary['successful_validations']} ({summary['success_rate']:.1%})")
    print(f"Failed: {summary['failed_validations']}")
    
    if correlations.get("spearman_correlation") is not None:
        print(f"\nCorrelation Analysis:")
        print(f"  Spearman correlation: {correlations['spearman_correlation']:.4f}")
        print(f"  Spearman p-value: {correlations['spearman_p_value']:.4e}")
        print(f"  Pearson correlation: {correlations['pearson_correlation']:.4f}")
        print(f"  Sample size: {correlations['sample_size']}")
        
        # Interpret correlation
        rho = abs(correlations['spearman_correlation'])
        if rho >= 0.7:
            strength = "strong"
        elif rho >= 0.4:
            strength = "moderate" 
        elif rho >= 0.2:
            strength = "weak"
        else:
            strength = "negligible"
        
        print(f"\n  Interpretation: {strength} correlation between network and FAERS metrics")
        
        if correlations['spearman_p_value'] < 0.05:
            print("  Statistical significance: YES (p < 0.05)")
        else:
            print("  Statistical significance: NO (p >= 0.05)")
    
    print(f"\nTop 10 High-Risk Drugs (by FAERS):")
    for i, drug in enumerate(summary['high_risk_drugs'], 1):
        print(f"  {i:2}. {drug['drug'][:25]:25} | Network: {drug['network_pri']:.4f} | FAERS: {drug['faers_score']:.4f}")
    
    # Save results
    results_file = os.path.join(output_dir, "faers_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary": summary,
            "individual_results": validation_results
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Save detailed CSV
    results_df = pd.DataFrame(validation_results)
    csv_file = os.path.join(output_dir, "faers_validation_details.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"Detailed CSV saved to: {csv_file}")
    
    return summary


def generate_validation_figures(validation_file: str, output_dir: str):
    """Generate validation visualization figures"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    with open(validation_file, 'r') as f:
        data = json.load(f)
    
    results = pd.DataFrame(data['individual_results'])
    results = results[results['validation_status'] == 'success']
    results = results[results['total_reports'] > 100]  # Filter low-data drugs
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot: Network PRI vs FAERS Risk Score
    ax1 = axes[0, 0]
    ax1.scatter(results['network_pri'], results['faers_risk_score'], 
               alpha=0.6, c='steelblue', edgecolor='white')
    ax1.set_xlabel('Network PRI Score')
    ax1.set_ylabel('FAERS Risk Score')
    ax1.set_title('Network vs FAERS Risk Scores')
    
    # Add regression line
    z = np.polyfit(results['network_pri'], results['faers_risk_score'], 1)
    p = np.poly1d(z)
    ax1.plot(results['network_pri'].sort_values(), 
             p(results['network_pri'].sort_values()), 
             "r--", alpha=0.8, label='Trend')
    ax1.legend()
    
    # 2. Distribution of FAERS Risk Scores
    ax2 = axes[0, 1]
    ax2.hist(results['faers_risk_score'], bins=20, color='steelblue', 
             edgecolor='white', alpha=0.7)
    ax2.set_xlabel('FAERS Risk Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of FAERS Risk Scores')
    ax2.axvline(results['faers_risk_score'].mean(), color='red', 
                linestyle='--', label='Mean')
    ax2.legend()
    
    # 3. Serious Event Ratio vs Network PRI
    ax3 = axes[1, 0]
    ax3.scatter(results['network_pri'], results['serious_event_ratio'],
               alpha=0.6, c='coral', edgecolor='white')
    ax3.set_xlabel('Network PRI Score')
    ax3.set_ylabel('Serious Event Ratio')
    ax3.set_title('Network PRI vs Serious Event Ratio')
    
    # 4. Top drugs comparison
    ax4 = axes[1, 1]
    top_10 = results.nlargest(10, 'faers_risk_score')
    y_pos = np.arange(len(top_10))
    
    ax4.barh(y_pos - 0.2, top_10['network_pri'] / top_10['network_pri'].max(), 
             height=0.4, color='steelblue', label='Network (normalized)')
    ax4.barh(y_pos + 0.2, top_10['faers_risk_score'], 
             height=0.4, color='coral', label='FAERS')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([d[:15] for d in top_10['drug_name']])
    ax4.set_xlabel('Risk Score')
    ax4.set_title('Top 10 Risk Drugs: Network vs FAERS')
    ax4.legend()
    
    plt.tight_layout()
    
    fig_file = os.path.join(output_dir, "faers_validation_figures.png")
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"Figures saved to: {fig_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run FAERS validation')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of drugs to validate (default: 50)')
    parser.add_argument('--output', type=str, default='validation_results',
                       help='Output directory (default: validation_results)')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to drug risk data file')
    parser.add_argument('--generate-figures', action='store_true',
                       help='Generate validation figures')
    
    args = parser.parse_args()
    
    print("FAERS External Validation")
    print("=" * 60)
    
    # Load data
    try:
        df = load_drug_risk_data(args.data_file)
        print(f"Loaded {len(df)} records")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure drug risk data is available.")
        return
    
    # Prepare sample
    drugs_with_pri = prepare_validation_sample(df, args.sample_size)
    print(f"Prepared {len(drugs_with_pri)} drugs for validation")
    
    # Run validation
    summary = run_validation(drugs_with_pri, args.output)
    
    # Generate figures if requested
    if args.generate_figures:
        validation_file = os.path.join(args.output, "faers_validation_results.json")
        if os.path.exists(validation_file):
            generate_validation_figures(validation_file, args.output)
    
    print("\n[OK] Validation complete!")
    
    return summary


if __name__ == "__main__":
    main()
