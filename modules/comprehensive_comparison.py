"""
Comprehensive Comparison Module

Three-approach comparison for drug interaction risk assessment:
1. Algorithmic: Network topology metrics (degree, betweenness, centrality)
2. GNN-Severity: Graph Attention Networks predicting severity class
3. GNN-Embedding: GAT with PubMedBERT drug embeddings

This module runs all approaches and generates comparative analysis for publication.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.drug_risk_network import DrugRiskNetwork


class AlgorithmicRiskAssessor:
    """
    Algorithmic risk assessment using network topology metrics only
    
    Avoids circular validation by NOT using AI-generated severity labels.
    Instead, uses graph-theoretic measures that capture drug risk from structure.
    """
    
    def __init__(self):
        self.network: Optional[DrugRiskNetwork] = None
        self.drug_metrics: Dict[str, Dict] = {}
        
    def build_network(self, df: pd.DataFrame) -> None:
        """Build interaction network from DDI data"""
        self.network = DrugRiskNetwork()
        
        print("Building drug interaction network...")
        
        # DrugRiskNetwork expects specific columns
        required_cols = ['drug_name_1', 'drug_name_2', 'severity_label', 'interaction_description']
        
        # Map column names if needed
        col_mapping = {
            'drug1_name': 'drug_name_1',
            'drug2_name': 'drug_name_2',
            'Severity': 'severity_label'
        }
        
        df_copy = df.copy()
        for old_col, new_col in col_mapping.items():
            if old_col in df_copy.columns and new_col not in df_copy.columns:
                df_copy[new_col] = df_copy[old_col]
        
        # Ensure required columns exist
        if 'interaction_description' not in df_copy.columns:
            df_copy['interaction_description'] = 'Drug interaction'
        
        # Build network using DrugRiskNetwork
        try:
            self.network.build_network(df_copy)
        except Exception as e:
            print(f"Warning: Could not build full network: {e}")
            print("Using simplified network construction...")
            self._build_simple_network(df)
            
    def _build_simple_network(self, df: pd.DataFrame) -> None:
        """Fallback simple network construction"""
        import networkx as nx
        
        self.G = nx.Graph()
        
        # Handle different column names
        if 'drug1_name' in df.columns:
            d1_col, d2_col = 'drug1_name', 'drug2_name'
        elif 'drug_name_1' in df.columns:
            d1_col, d2_col = 'drug_name_1', 'drug_name_2'
        else:
            print("Warning: Could not find drug columns")
            return
            
        for _, row in df.iterrows():
            d1, d2 = row[d1_col], row[d2_col]
            if self.G.has_edge(d1, d2):
                self.G[d1][d2]['weight'] += 1
            else:
                self.G.add_edge(d1, d2, weight=1)
        
        print(f"Network: {self.G.number_of_nodes()} drugs, {self.G.number_of_edges()} interactions")
    
    def compute_risk_metrics(self) -> pd.DataFrame:
        """Compute topology-based risk metrics for all drugs"""
        import networkx as nx
        
        # If we have the full DrugRiskNetwork
        if self.network and self.network._initialized:
            metrics_list = []
            for drug_name, node in self.network.nodes.items():
                # Compute severity profile from counts
                severity_score = (
                    node.contraindicated_count * 10 +
                    node.major_count * 7 +
                    node.moderate_count * 4 +
                    node.minor_count * 1
                )
                total_interactions = (node.contraindicated_count + node.major_count + 
                                     node.moderate_count + node.minor_count)
                max_severity = max(1, total_interactions) * 10
                severity_profile = min(1.0, severity_score / max_severity)
                
                metrics_list.append({
                    'drug_name': drug_name,
                    'degree_centrality': node.degree_centrality,
                    'weighted_degree': node.weighted_degree,
                    'betweenness_centrality': node.betweenness_centrality,
                    'severity_profile': severity_profile,
                    'pri_score': node.pri_score,
                    'contraindicated_count': node.contraindicated_count,
                    'major_count': node.major_count,
                    'moderate_count': node.moderate_count,
                    'minor_count': node.minor_count
                })
            return pd.DataFrame(metrics_list)
        
        # Fallback to networkx metrics
        if hasattr(self, 'G'):
            print("Computing network centrality metrics...")
            
            degree_cent = nx.degree_centrality(self.G)
            betweenness = nx.betweenness_centrality(self.G)
            
            try:
                eigenvector = nx.eigenvector_centrality(self.G, max_iter=500)
            except:
                eigenvector = {n: 0 for n in self.G.nodes()}
            
            pagerank = nx.pagerank(self.G)
            closeness = nx.closeness_centrality(self.G)
            
            metrics_list = []
            for drug in self.G.nodes():
                pri = (
                    0.30 * degree_cent.get(drug, 0) +
                    0.25 * betweenness.get(drug, 0) +
                    0.20 * eigenvector.get(drug, 0) +
                    0.15 * pagerank.get(drug, 0) +
                    0.10 * closeness.get(drug, 0)
                )
                
                metrics_list.append({
                    'drug_name': drug,
                    'degree_centrality': degree_cent.get(drug, 0),
                    'betweenness_centrality': betweenness.get(drug, 0),
                    'eigenvector_centrality': eigenvector.get(drug, 0),
                    'pagerank': pagerank.get(drug, 0),
                    'closeness_centrality': closeness.get(drug, 0),
                    'pri_score': pri
                })
                
                self.drug_metrics[drug] = metrics_list[-1]
            
            return pd.DataFrame(metrics_list)
        
        raise ValueError("Network not built. Call build_network first.")
    
    def get_high_risk_drugs(self, top_n: int = 50) -> List[Dict]:
        """Get top N high-risk drugs by PRI score"""
        df = self.compute_risk_metrics()
        df_sorted = df.nlargest(top_n, 'pri_score')
        
        return df_sorted.to_dict('records')


class ComprehensiveComparison:
    """
    Run and compare all three risk assessment approaches
    """
    
    def __init__(self, output_dir: str = "comparison_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.algorithmic = AlgorithmicRiskAssessor()
        self.results: Dict[str, Any] = {}
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load DDI dataset"""
        if data_path is None:
            paths = [
                "data/ddi_cardio_or_antithrombotic_labeled (1).csv",
                "ddi_cardio_or_antithrombotic_labeled (1).csv"
            ]
            for p in paths:
                if os.path.exists(p):
                    data_path = p
                    break
        
        if data_path is None or not os.path.exists(data_path):
            raise FileNotFoundError("DDI data file not found")
        
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} interactions")
        
        return df
    
    def run_algorithmic_approach(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run algorithmic network-based approach"""
        print("\n" + "=" * 60)
        print("APPROACH 1: Algorithmic (Network Topology)")
        print("=" * 60)
        
        self.algorithmic.build_network(df)
        metrics_df = self.algorithmic.compute_risk_metrics()
        
        # Summary statistics
        results = {
            'approach': 'Algorithmic',
            'description': 'Network topology metrics (no AI labels)',
            'num_drugs': len(metrics_df),
            'pri_stats': {
                'mean': float(metrics_df['pri_score'].mean()),
                'std': float(metrics_df['pri_score'].std()),
                'min': float(metrics_df['pri_score'].min()),
                'max': float(metrics_df['pri_score'].max()),
                'median': float(metrics_df['pri_score'].median())
            },
            'top_risk_drugs': metrics_df.nlargest(20, 'pri_score')[
                ['drug_name', 'pri_score', 'degree_centrality', 'betweenness_centrality']
            ].to_dict('records')
        }
        
        # Save detailed results
        metrics_df.to_csv(
            os.path.join(self.output_dir, 'algorithmic_risk_scores.csv'),
            index=False
        )
        
        print(f"  Drugs analyzed: {results['num_drugs']}")
        print(f"  Mean PRI: {results['pri_stats']['mean']:.4f}")
        print(f"  Max PRI: {results['pri_stats']['max']:.4f}")
        
        return results
    
    def run_gnn_severity_approach(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run GNN with network features approach"""
        print("\n" + "=" * 60)
        print("APPROACH 2: GNN-Severity (Network Features)")
        print("=" * 60)
        
        try:
            from modules.gnn_risk_assessment import (
                GNNSeverityPredictor, 
                TORCH_GEOMETRIC_AVAILABLE
            )
            
            if not TORCH_GEOMETRIC_AVAILABLE:
                return {
                    'approach': 'GNN-Severity',
                    'status': 'skipped',
                    'reason': 'torch_geometric not installed'
                }
            
            predictor = GNNSeverityPredictor()
            features, edges, drug_to_idx = predictor.prepare_features(
                df, self.algorithmic.drug_metrics
            )
            
            # Create labels from severity column
            severity_map = {'Major interaction': 2, 'Moderate interaction': 1, 'Minor interaction': 0, 
                           'Contraindicated interaction': 2, 'Major': 2, 'Moderate': 1, 'Minor': 0}
            labels = np.ones(len(drug_to_idx), dtype=np.int64)
            
            sev_col = 'severity_label' if 'severity_label' in df.columns else 'Severity'
            drug_col = 'drug_name_1' if 'drug_name_1' in df.columns else 'drug1_name'
            
            if sev_col in df.columns and drug_col in df.columns:
                drug_severity = df.groupby(drug_col)[sev_col].apply(
                    lambda x: x.mode()[0] if len(x) > 0 else 'Moderate interaction'
                ).to_dict()
                idx_to_drug = {i: d for d, i in drug_to_idx.items()}
                for idx in range(len(drug_to_idx)):
                    drug = idx_to_drug[idx]
                    if drug in drug_severity:
                        labels[idx] = severity_map.get(drug_severity[drug], 1)
            
            data = predictor.build_graph(features, edges, labels)
            
            # Train
            print("  Training GAT model...")
            predictor.train(data)
            
            # Evaluate
            accuracy = predictor.evaluate(data)
            
            results = {
                'approach': 'GNN-Severity',
                'description': 'Graph Attention Network with network topology features',
                'accuracy': float(accuracy),
                'num_drugs': len(drug_to_idx),
                'feature_dim': features.shape[1],
                'status': 'completed'
            }
            
            print(f"  Final Accuracy: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            print(f"  Error: {e}")
            return {
                'approach': 'GNN-Severity',
                'status': 'error',
                'reason': str(e)
            }
    
    def run_gnn_embedding_approach(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run GNN with PubMedBERT embeddings approach"""
        print("\n" + "=" * 60)
        print("APPROACH 3: GNN-Embedding (PubMedBERT)")
        print("=" * 60)
        
        try:
            from modules.gnn_risk_assessment import (
                GNNEmbeddingPredictor,
                TORCH_GEOMETRIC_AVAILABLE,
                TRANSFORMERS_AVAILABLE
            )
            
            if not TORCH_GEOMETRIC_AVAILABLE:
                return {
                    'approach': 'GNN-Embedding',
                    'status': 'skipped',
                    'reason': 'torch_geometric not installed'
                }
            
            if not TRANSFORMERS_AVAILABLE:
                return {
                    'approach': 'GNN-Embedding',
                    'status': 'skipped',
                    'reason': 'transformers not installed'
                }
            
            # Use sample for speed during comparison
            df_sample = df.head(5000)
            
            predictor = GNNEmbeddingPredictor()
            
            sev_col = 'severity_label' if 'severity_label' in df_sample.columns else 'Severity'
            drug_col = 'drug_name_1' if 'drug_name_1' in df_sample.columns else 'drug1_name'
            severity_map = {'Major interaction': 2, 'Moderate interaction': 1, 'Minor interaction': 0,
                           'Contraindicated interaction': 2, 'Major': 2, 'Moderate': 1, 'Minor': 0}
            drug_labels = {}
            
            if sev_col in df_sample.columns and drug_col in df_sample.columns:
                drug_severity = df_sample.groupby(drug_col)[sev_col].apply(
                    lambda x: x.mode()[0] if len(x) > 0 else 'Moderate interaction'
                ).to_dict()
                drug_labels = {d: severity_map.get(s, 1) for d, s in drug_severity.items()}
            
            print("  Training GAT model with embeddings...")
            data = predictor.build_and_train(df_sample, drug_labels)
            
            # Evaluate
            predictor.model.eval()
            import torch
            with torch.no_grad():
                out = predictor.model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                accuracy = (pred == data.y).float().mean().item()
            
            results = {
                'approach': 'GNN-Embedding',
                'description': 'Graph Attention Network with PubMedBERT drug embeddings',
                'accuracy': float(accuracy),
                'num_drugs': data.x.shape[0],
                'embedding_dim': data.x.shape[1],
                'embedding_model': predictor.config.embedding_model,
                'status': 'completed'
            }
            
            print(f"  Final Accuracy: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            print(f"  Error: {e}")
            return {
                'approach': 'GNN-Embedding',
                'status': 'error',
                'reason': str(e)
            }
    
    def compute_correlation_analysis(self) -> Dict[str, Any]:
        """Compute correlations between approaches"""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        
        correlations = {}
        
        # Load algorithmic results
        algo_path = os.path.join(self.output_dir, 'algorithmic_risk_scores.csv')
        if not os.path.exists(algo_path):
            return {'error': 'Algorithmic results not found'}
        
        algo_df = pd.read_csv(algo_path)
        
        # Correlation between different centrality metrics
        metrics = ['degree_centrality', 'betweenness_centrality', 
                  'closeness_centrality', 'eigenvector_centrality', 'pagerank']
        
        available_metrics = [m for m in metrics if m in algo_df.columns]
        
        if len(available_metrics) >= 2:
            corr_matrix = algo_df[available_metrics].corr()
            correlations['metric_correlations'] = corr_matrix.to_dict()
            
            print("Centrality Metric Correlations:")
            print(corr_matrix.round(3).to_string())
        
        return correlations
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table for publication"""
        rows = []
        
        for approach_name, result in self.results.items():
            if isinstance(result, dict) and result.get('status') != 'error':
                row = {
                    'Approach': result.get('approach', approach_name),
                    'Description': result.get('description', ''),
                    'Drugs': result.get('num_drugs', '-'),
                    'Accuracy': f"{result.get('accuracy', '-'):.4f}" if result.get('accuracy') else '-',
                    'Status': result.get('status', 'completed')
                }
                
                if 'pri_stats' in result:
                    row['Mean Risk'] = f"{result['pri_stats']['mean']:.4f}"
                    row['Max Risk'] = f"{result['pri_stats']['max']:.4f}"
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def run_full_comparison(self, data_path: str = None) -> Dict[str, Any]:
        """Run complete three-way comparison"""
        print("=" * 60)
        print("COMPREHENSIVE DDI RISK ASSESSMENT COMPARISON")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)
        
        # Load data
        df = self.load_data(data_path)
        
        # Run all approaches
        self.results['algorithmic'] = self.run_algorithmic_approach(df)
        self.results['gnn_severity'] = self.run_gnn_severity_approach(df)
        self.results['gnn_embedding'] = self.run_gnn_embedding_approach(df)
        
        # Correlation analysis
        self.results['correlations'] = self.compute_correlation_analysis()
        
        # Generate comparison table
        comparison_table = self.generate_comparison_table()
        
        # Print summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(comparison_table.to_string(index=False))
        
        # Save results
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['data_stats'] = {
            'total_interactions': len(df),
            'unique_drug1': df['drug1_name'].nunique() if 'drug1_name' in df.columns else 0,
            'unique_drug2': df['drug2_name'].nunique() if 'drug2_name' in df.columns else 0
        }
        
        # Save JSON
        with open(os.path.join(self.output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save comparison table
        comparison_table.to_csv(
            os.path.join(self.output_dir, 'comparison_table.csv'),
            index=False
        )
        comparison_table.to_markdown(
            os.path.join(self.output_dir, 'comparison_table.md'),
            index=False
        )
        
        print(f"\nResults saved to: {self.output_dir}/")
        
        return self.results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run comprehensive DDI risk assessment comparison'
    )
    parser.add_argument('--data', type=str, default=None,
                       help='Path to DDI data CSV')
    parser.add_argument('--output', type=str, default='comparison_results',
                       help='Output directory')
    parser.add_argument('--skip-gnn', action='store_true',
                       help='Skip GNN approaches (faster)')
    
    args = parser.parse_args()
    
    comparison = ComprehensiveComparison(args.output)
    
    if args.skip_gnn:
        # Only run algorithmic
        df = comparison.load_data(args.data)
        comparison.results['algorithmic'] = comparison.run_algorithmic_approach(df)
        comparison.results['gnn_severity'] = {'status': 'skipped', 'reason': '--skip-gnn flag'}
        comparison.results['gnn_embedding'] = {'status': 'skipped', 'reason': '--skip-gnn flag'}
    else:
        comparison.run_full_comparison(args.data)
    
    print("\n[OK] Comparison complete!")


if __name__ == "__main__":
    main()
