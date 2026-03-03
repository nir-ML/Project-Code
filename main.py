#!/usr/bin/env python3
"""
AI-based Polypharmacy Risk-aware Drug Recommender System
Main Entry Point

This script demonstrates the modular architecture for drug-drug interaction
analysis and polypharmacy risk assessment.

Usage:
------
    python main.py [options]

Options:
    --drugs DRUG1,DRUG2,...    Comma-separated list of drug names to analyze
    --interactive              Run in interactive mode
    --train-model              Train the ML severity model
    --output FORMAT            Output format: clinical, patient, json, all (default: all)
    --save-report PATH         Save report to file
    --verbose                  Enable verbose output

Examples:
---------
    # Analyze specific drugs
    python main.py --drugs "Warfarin,Aspirin,Metoprolol,Lisinopril"
    
    # Interactive mode
    python main.py --interactive
    
    # Train ML model and analyze
    python main.py --drugs "Warfarin,Aspirin" --train-model
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from modules import Orchestrator, PipelineStatus


class PolypharmacyAnalyzer:
    """
    Main interface for the Polypharmacy Risk Analysis System
    """
    
    # Sample drug combinations for demonstration
    SAMPLE_DRUG_LISTS = {
        'cardiovascular_basic': [
            'Warfarin', 'Aspirin', 'Metoprolol', 'Lisinopril', 'Atorvastatin'
        ],
        'cardiovascular_combo': [
            'Warfarin', 'Clopidogrel', 'Aspirin', 'Heparin'
        ],
        'heart_failure': [
            'Digoxin', 'Furosemide', 'Spironolactone', 'Carvedilol', 'Lisinopril'
        ],
        'hypertension': [
            'Amlodipine', 'Lisinopril', 'Hydrochlorothiazide', 'Metoprolol'
        ],
        'diabetes_cardiac': [
            'Metformin', 'Glipizide', 'Atorvastatin', 'Lisinopril', 'Aspirin'
        ]
    }
    
    def __init__(self, data_path: str = None, verbose: bool = True):
        """
        Initialize the analyzer
        
        Args:
            data_path: Path to DDI CSV file (auto-detected if not provided)
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.orchestrator = None
        self.df = None
        
        # Find data file
        if data_path is None:
            data_path = self._find_data_file()
        
        self.data_path = data_path
        self.use_llm = True  # Default to using LLM
        
    def _find_data_file(self) -> str:
        """Auto-detect DDI data file"""
        possible_names = [
            'data/ddi_cardio_or_antithrombotic_labeled (1).csv',
            'data/ddi_cardio_or_antithrombotic_labeled.csv',
            'ddi_cardio_or_antithrombotic_labeled (1).csv',
            'ddi_cardio_or_antithrombotic_labeled.csv',
            'ddi_data.csv'
        ]
        
        script_dir = Path(__file__).parent
        
        for name in possible_names:
            path = script_dir / name
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            "DDI data file not found. Please provide the path using --data-path"
        )
    
    def initialize(self, train_model: bool = False, use_llm: bool = True):
        """
        Load data and initialize the orchestrator
        
        Args:
            train_model: Train ML severity model
            use_llm: Use BioMistral-7B for explanations (requires Ollama)
        """
        print("=" * 60)
        print("AI-based Polypharmacy Risk-aware Recommender System")
        print("=" * 60)
        print()
        
        # Load data
        print(f"Loading DDI database from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"   [OK] Loaded {len(self.df):,} drug-drug interactions")
        print()
        
        self.use_llm = use_llm
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(verbose=self.verbose)
        self.orchestrator.initialize(
            self.df, 
            train_severity_model=train_model,
            use_llm=use_llm
        )
        print()
        
    def analyze(self, drug_list: list) -> dict:
        """
        Analyze a list of drugs for interactions and risks
        
        Args:
            drug_list: List of drug names
            
        Returns:
            Analysis results dictionary
        """
        if self.orchestrator is None:
            self.initialize()
        
        return self.orchestrator.analyze_drugs(drug_list)
    
    def print_clinical_report(self, result: dict):
        """Print the clinical report"""
        report = result.get('data', {}).get('pipeline_results', {}).get('clinical_report', '')
        if report:
            print(report)
        else:
            print("No clinical report generated.")
    
    def print_patient_summary(self, result: dict):
        """Print the patient-friendly summary"""
        summary = result.get('data', {}).get('pipeline_results', {}).get('patient_summary', '')
        if summary:
            print(summary)
        else:
            print("No patient summary generated.")
    
    def get_structured_output(self, result: dict) -> dict:
        """Get the structured JSON output"""
        return result.get('data', {}).get('pipeline_results', {}).get('structured_output', {})
    
    def print_multi_objective_recommendations(self, result: dict):
        """Print the multi-objective recommendations (Paper methodology)"""
        pipeline_results = result.get('data', {}).get('pipeline_results', {})
        mo_recs = pipeline_results.get('multi_objective_recommendations', {})
        
        if not mo_recs:
            print("No multi-objective recommendations available.")
            return
        
        print("\n" + "=" * 60)
        print("MULTI-OBJECTIVE DRUG RECOMMENDATIONS")
        print("   (Paper Methodology: PRI + Centrality + Phenotype)")
        print("=" * 60)
        
        # Overall risk
        overall = mo_recs.get('overall_risk', {})
        print(f"\nOverall Polypharmacy Risk:")
        print(f"   • Risk Level: {overall.get('level', 'N/A')}")
        print(f"   • Risk Score: {overall.get('score', 0):.4f}")
        print(f"   • Total Interactions: {overall.get('total_interactions', 0)}")
        
        severity = overall.get('severity_breakdown', {})
        if severity:
            print(f"   • Severity Breakdown:")
            for sev, count in severity.items():
                print(f"     - {sev}: {count}")
        
        # Recommendations
        recs = mo_recs.get('recommendations', [])
        if recs:
            print(f"\nPrioritized Replacement Recommendations:")
            print("-" * 50)
            
            for i, rec in enumerate(recs, 1):
                target = rec.get('target_drug', 'Unknown')
                best_alt = rec.get('best_alternative', {})
                
                print(f"\n  {i}. Replace: {target}")
                print(f"     Risk Contribution: {rec.get('risk_contribution', 0):.4f}")
                print(f"     Severe Interactions: {rec.get('severe_interactions_in_list', 0)}")
                
                if best_alt:
                    print(f"\n     Best Alternative: {best_alt.get('drug_name', 'N/A')}")
                    print(f"        • ATC Match: {best_alt.get('atc_match_type', 'N/A')}")
                    print(f"        • Multi-Objective Score: {best_alt.get('multi_objective_score', 0):.4f}")
                    
                    metrics = best_alt.get('risk_metrics', {})
                    print(f"        • PRI Reduction: {metrics.get('pri_reduction', 0):.4f}")
                    print(f"        • Centrality Reduction: {metrics.get('centrality_reduction', 0):.4f}")
                    print(f"        • Severe Interaction Delta: {metrics.get('severe_interaction_delta', 0)}")
                    
                    phenotype = best_alt.get('phenotype_analysis', {})
                    avoided = phenotype.get('avoided', [])
                    introduced = phenotype.get('introduced', [])
                    if avoided:
                        print(f"        • Phenotypes Avoided: {', '.join(avoided)}")
                    if introduced:
                        print(f"        • Phenotypes Introduced: {', '.join(introduced)}")
                else:
                    print(f"     No suitable alternative found")
                
                # Show other alternatives
                all_alts = rec.get('all_alternatives', [])[1:3]  # Skip best, show next 2
                if all_alts:
                    print(f"\n     Other Options:")
                    for alt in all_alts:
                        print(f"       - {alt.get('drug_name', 'N/A')} (Score: {alt.get('multi_objective_score', 0):.4f})")
        
        # Summary
        summary = mo_recs.get('summary', {})
        if summary:
            print(f"\nSummary:")
            print(f"   • Drugs Analyzed: {summary.get('drugs_analyzed', 0)}")
            print(f"   • Drugs with Alternatives: {summary.get('drugs_with_alternatives', 0)}")
            print(f"   • Estimated Risk Reduction: {summary.get('estimated_risk_reduction', 0):.4f}")
        
        print("\n" + "=" * 60)
    
    def print_pri_analysis(self, result: dict):
        """Print the Polypharmacy Risk Index analysis"""
        pipeline_results = result.get('data', {}).get('pipeline_results', {})
        pri_data = pipeline_results.get('pri_analysis', {})
        
        if not pri_data:
            print("No PRI analysis available.")
            return
        
        print("\n" + "=" * 60)
        print("POLYPHARMACY RISK INDEX (PRI) ANALYSIS")
        print("   (Paper Methodology: Centrality Metrics)")
        print("=" * 60)
        
        # Sort by PRI score
        sorted_drugs = sorted(pri_data.items(), key=lambda x: -x[1].get('pri_score', 0))
        
        print(f"\n{'Drug':<20} {'PRI Score':<12} {'Degree':<10} {'Weighted':<10} {'Betweenness':<12}")
        print("-" * 64)
        
        for drug, metrics in sorted_drugs:
            print(f"{drug.title():<20} {metrics.get('pri_score', 0):<12.4f} "
                  f"{metrics.get('degree_centrality', 0):<10.4f} "
                  f"{metrics.get('weighted_degree', 0):<10.4f} "
                  f"{metrics.get('betweenness_centrality', 0):<12.4f}")
        
        # Highlight highest risk
        highest_risk = pipeline_results.get('highest_risk_drug')
        if highest_risk:
            print(f"\nHighest Risk Contributor: {highest_risk[0].title()} (PRI: {highest_risk[1]:.4f})")
        
        print("\n" + "=" * 60)
    
    def save_report(self, result: dict, output_path: str, format: str = 'all'):
        """Save report to file"""
        pipeline_results = result.get('data', {}).get('pipeline_results', {})
        
        base_path = Path(output_path).stem
        dir_path = Path(output_path).parent
        
        if format in ['clinical', 'all']:
            clinical_path = dir_path / f"{base_path}_clinical.txt"
            with open(clinical_path, 'w') as f:
                f.write(pipeline_results.get('clinical_report', ''))
            print(f"[OK] Clinical report saved to: {clinical_path}")
        
        if format in ['patient', 'all']:
            patient_path = dir_path / f"{base_path}_patient.txt"
            with open(patient_path, 'w') as f:
                f.write(pipeline_results.get('patient_summary', ''))
            print(f"[OK] Patient summary saved to: {patient_path}")
        
        if format in ['json', 'all']:
            import json
            json_path = dir_path / f"{base_path}_structured.json"
            with open(json_path, 'w') as f:
                json.dump(pipeline_results.get('structured_output', {}), f, indent=2)
            print(f"[OK] Structured output saved to: {json_path}")
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "=" * 60)
        print("Interactive Mode")
        print("   (Paper Methodology: DDI Network + PRI + Multi-Objective)")
        print("=" * 60)
        print("\nCommands:")
        print("  analyze <drug1>, <drug2>, ...  - Analyze drug interactions")
        print("  pri <drug1>, <drug2>, ...      - Show PRI analysis only")
        print("  recommend <drug1>, <drug2>...  - Show multi-objective recommendations")
        print("  sample <name>                  - Use sample drug list")
        print("  samples                        - List available sample lists")
        print("  help                           - Show this help")
        print("  quit                           - Exit")
        print()
        
        while True:
            try:
                user_input = input("\nEnter drugs or command: ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                    
                elif command == 'help':
                    print("\nCommands:")
                    print("  analyze <drug1>, <drug2>, ...  - Full analysis with reports")
                    print("  pri <drug1>, <drug2>, ...      - Show PRI analysis only")
                    print("  recommend <drug1>, <drug2>...  - Show multi-objective recommendations")
                    print("  sample <name>                  - Use sample drug list")
                    print("  samples                        - List available sample lists")
                    print("  quit                           - Exit")
                    
                elif command == 'samples':
                    print("\nAvailable sample drug lists:")
                    for name, drugs in self.SAMPLE_DRUG_LISTS.items():
                        print(f"  • {name}: {', '.join(drugs)}")
                
                elif command == 'pri':
                    if len(parts) < 2:
                        print("Please provide drug names separated by commas.")
                        continue
                    drugs = [d.strip() for d in parts[1].split(',')]
                    result = self.analyze(drugs)
                    self.print_pri_analysis(result)
                
                elif command == 'recommend':
                    if len(parts) < 2:
                        print("Please provide drug names separated by commas.")
                        continue
                    drugs = [d.strip() for d in parts[1].split(',')]
                    result = self.analyze(drugs)
                    self.print_multi_objective_recommendations(result)
                    
                elif command == 'sample':
                    if len(parts) < 2:
                        print("Please specify a sample name. Use 'samples' to see options.")
                        continue
                    sample_name = parts[1].strip()
                    if sample_name in self.SAMPLE_DRUG_LISTS:
                        drugs = self.SAMPLE_DRUG_LISTS[sample_name]
                        print(f"\nUsing sample '{sample_name}': {', '.join(drugs)}")
                        result = self.analyze(drugs)
                        self.print_clinical_report(result)
                        self.print_multi_objective_recommendations(result)
                    else:
                        print(f"Unknown sample: {sample_name}. Use 'samples' to see options.")
                    
                elif command == 'analyze':
                    if len(parts) < 2:
                        print("Please provide drug names separated by commas.")
                        continue
                    drugs = [d.strip() for d in parts[1].split(',')]
                    result = self.analyze(drugs)
                    self.print_clinical_report(result)
                    self.print_multi_objective_recommendations(result)
                    
                else:
                    # Assume it's a drug list
                    drugs = [d.strip() for d in user_input.split(',')]
                    if len(drugs) >= 1:
                        result = self.analyze(drugs)
                        self.print_clinical_report(result)
                        self.print_multi_objective_recommendations(result)
                    else:
                        print("Unknown command. Type 'help' for available commands.")
                        
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI-based Polypharmacy Risk-aware Drug Recommender System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --drugs "Warfarin,Aspirin,Metoprolol"
  python main.py --interactive
  python main.py --drugs "Warfarin,Aspirin" --output clinical --save-report report.txt
        """
    )
    
    parser.add_argument(
        '--drugs', '-d',
        type=str,
        help='Comma-separated list of drug names to analyze'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--train-model', '-t',
        action='store_true',
        help='Train the ML severity prediction model'
    )
    
    parser.add_argument(
        '--llm', '-l',
        action='store_true',
        default=True,
        help='Use BioMistral-7B LLM for explanations (requires Ollama)'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM, use template-based generation only'
    )
    
    parser.add_argument(
        '--output', '-o',
        choices=['clinical', 'patient', 'json', 'all'],
        default='all',
        help='Output format (default: all)'
    )
    
    parser.add_argument(
        '--save-report', '-s',
        type=str,
        help='Save report to file'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to DDI CSV data file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Disable verbose output'
    )
    
    parser.add_argument(
        '--sample',
        type=str,
        choices=list(PolypharmacyAnalyzer.SAMPLE_DRUG_LISTS.keys()),
        help='Use a predefined sample drug list'
    )
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbose = not args.quiet
    
    # Determine LLM usage
    use_llm = args.llm and not args.no_llm
    
    try:
        # Initialize analyzer
        analyzer = PolypharmacyAnalyzer(
            data_path=args.data_path,
            verbose=verbose
        )
        analyzer.initialize(train_model=args.train_model, use_llm=use_llm)
        
        # Run in appropriate mode
        if args.interactive:
            analyzer.interactive_mode()
            
        elif args.drugs or args.sample:
            # Get drug list
            if args.sample:
                drugs = PolypharmacyAnalyzer.SAMPLE_DRUG_LISTS[args.sample]
                print(f"\nUsing sample '{args.sample}': {', '.join(drugs)}")
            else:
                drugs = [d.strip() for d in args.drugs.split(',')]
            
            # Run analysis
            result = analyzer.analyze(drugs)
            
            # Output results
            if args.output in ['clinical', 'all']:
                analyzer.print_clinical_report(result)
            
            if args.output in ['patient', 'all']:
                analyzer.print_patient_summary(result)
            
            if args.output == 'json':
                import json
                print(json.dumps(analyzer.get_structured_output(result), indent=2))
            
            # Save report if requested
            if args.save_report:
                analyzer.save_report(result, args.save_report, args.output)
                
        else:
            # No drugs specified, run demo
            print("\n📌 No drugs specified. Running demonstration with sample data...\n")
            demo_drugs = ['Warfarin', 'Aspirin', 'Metoprolol', 'Lisinopril']
            print(f"Demo drug list: {', '.join(demo_drugs)}")
            print()
            
            result = analyzer.analyze(demo_drugs)
            analyzer.print_clinical_report(result)
            
            print("\n" + "=" * 60)
            print("Tip: Use --interactive mode or --drugs to analyze your own medications")
            print("   Example: python main.py --drugs 'Warfarin,Aspirin,Metoprolol'")
            print("=" * 60)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("   Use --data-path to specify the DDI data file location.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
