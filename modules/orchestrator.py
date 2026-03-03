"""
Orchestrator Module - Coordinates all modules in the polypharmacy analysis pipeline

Implements the paper's methodology:
1. Drug Risk Network construction
2. Polypharmacy Risk Index (PRI) computation
3. Multi-objective alternative recommendation
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum, auto

from .base_module import BaseModule, Result, PipelineStatus, Message
from .interaction_detector import InteractionDetector
from .severity_classifier import SeverityClassifier
from .alternative_finder import AlternativeFinder
from .report_generator import ReportGenerator
from .drug_risk_network import DrugRiskNetwork
from .recommender import AlternativeRecommender


class PipelineStage(Enum):
    """Pipeline execution stages"""
    INITIALIZATION = auto()
    DRUG_VALIDATION = auto()
    NETWORK_ANALYSIS = auto()  # New: Drug Risk Network
    INTERACTION_DETECTION = auto()
    SEVERITY_ANALYSIS = auto()
    PRI_COMPUTATION = auto()  # New: Polypharmacy Risk Index
    ALTERNATIVE_FINDING = auto()
    MULTI_OBJECTIVE_RANKING = auto()  # New: Paper's recommender
    REPORT_GENERATION = auto()
    COMPLETED = auto()
    FAILED = auto()


class Orchestrator(BaseModule):
    """
    Orchestrator Module
    
    Central coordinator for the polypharmacy risk analysis pipeline.
    Manages the execution flow between specialized modules.
    
    Pipeline Flow:
        User Input
            -> Drug List Input
            -> InteractionDetector (Detect DDIs)
            -> SeverityClassifier (Classify & Score)
            -> AlternativeFinder (Find Alternatives)
            -> ReportGenerator (Generate Reports)
            -> Final Report
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="Orchestrator",
            description="Coordinates the polypharmacy analysis pipeline"
        )
        self.verbose = verbose
        
        # Initialize child modules
        self.interaction_module = InteractionDetector()
        self.severity_module = SeverityClassifier()
        self.alternative_module = AlternativeFinder()
        self.explanation_module = ReportGenerator()
        
        # Paper-based components
        self.risk_network: Optional[DrugRiskNetwork] = None
        self.recommender: Optional[AlternativeRecommender] = None
        
        # Execution state
        self.current_stage = PipelineStage.INITIALIZATION
        self.execution_log = []
        self.module_results = {}
        
    def initialize(self, ddi_dataframe: pd.DataFrame, 
                   train_severity_model: bool = False,
                   use_llm: bool = True) -> bool:
        """
        Initialize all child modules with the DDI database
        
        Args:
            ddi_dataframe: DDI database DataFrame
            train_severity_model: Whether to train ML severity model
            use_llm: Whether to use BioMistral-7B for explanations
        """
        self._log("Initializing Orchestrator and child modules...")
        
        try:
            # Initialize Interaction Module
            self._log("  - Initializing InteractionDetector...")
            self.interaction_module.initialize(ddi_dataframe)
            
            # Initialize Severity Module
            self._log("  - Initializing SeverityClassifier...")
            self.severity_module.initialize(
                ddi_dataframe=ddi_dataframe if train_severity_model else None,
                train_model=train_severity_model
            )
            
            # Initialize Alternative Module
            self._log("  - Initializing AlternativeFinder...")
            self.alternative_module.initialize(ddi_dataframe)
            
            # Initialize Explanation Module with LLM support
            self._log("  - Initializing ReportGenerator...")
            self.explanation_module.initialize(use_llm=use_llm)
            
            # Initialize Drug Risk Network (Paper methodology)
            self._log("  - Building Drug Risk Network (Paper methodology)...")
            self.risk_network = DrugRiskNetwork()
            self.risk_network.build_network(ddi_dataframe)
            self._log(f"     Network: {len(self.risk_network.nodes)} nodes, {len(self.risk_network.edges)} edges")
            
            # Initialize Multi-Objective Recommender (Paper methodology)
            self._log("  - Initializing Multi-Objective Recommender...")
            self.recommender = AlternativeRecommender(self.risk_network)
            
            self._initialized = True
            self._log("All modules initialized successfully!")
            return True
            
        except Exception as e:
            self._log(f"Initialization failed: {e}")
            return False
    
    def _log(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.execution_log.append(entry)
        if self.verbose:
            print(entry)
    
    def _update_stage(self, stage: PipelineStage):
        """Update current pipeline stage"""
        self.current_stage = stage
        self._log(f"Stage: {stage.name}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> tuple:
        """Validate input contains drug list"""
        if 'drugs' not in input_data:
            return False, "Missing 'drugs' key in input"
        
        drugs = input_data['drugs']
        if not isinstance(drugs, list):
            return False, "'drugs' must be a list"
        
        if len(drugs) < 1:
            return False, "Need at least 1 drug for analysis"
        
        return True, ""
    
    def process(self, input_data: Dict[str, Any]) -> Result:
        """
        Execute the full polypharmacy analysis pipeline
        
        Implements the paper's methodology:
        1. Drug Risk Network analysis
        2. PRI (Polypharmacy Risk Index) computation
        3. Multi-objective alternative recommendation
        """
        start_time = datetime.now()
        self._log("═" * 60)
        self._log("Starting Polypharmacy Risk Analysis Pipeline")
        self._log("   (Paper Methodology: DDI Network + PRI + Multi-Objective)")
        self._log("═" * 60)
        
        drugs = input_data['drugs']
        self._log(f"Input: {len(drugs)} medications - {', '.join(drugs)}")
        
        accumulated_data = {'drugs': drugs}
        errors = []
        
        try:
            # Stage 1: Network-Based Risk Analysis (Paper methodology)
            self._update_stage(PipelineStage.NETWORK_ANALYSIS)
            if self.risk_network:
                network_risk = self.risk_network.compute_polypharmacy_risk(drugs)
                accumulated_data['network_risk'] = network_risk
                self._log(f"   Network Risk Level: {network_risk.get('risk_level', 'N/A')}")
                self._log(f"   Total Interactions: {network_risk.get('total_interactions', 0)}")
            
            # Stage 2: PRI Computation (Paper methodology)
            self._update_stage(PipelineStage.PRI_COMPUTATION)
            if self.risk_network:
                pri_data = {}
                for drug in drugs:
                    metrics = self.risk_network.get_drug_metrics(drug)
                    if metrics:
                        pri_data[drug] = {
                            'pri_score': metrics.get('pri_score', 0),
                            'degree_centrality': metrics.get('degree_centrality', 0),
                            'weighted_degree': metrics.get('weighted_degree', 0),
                            'betweenness_centrality': metrics.get('betweenness_centrality', 0)
                        }
                accumulated_data['pri_analysis'] = pri_data
                
                # Find highest risk contributor
                highest_risk = self.risk_network.get_highest_risk_drug(drugs)
                if highest_risk:
                    accumulated_data['highest_risk_drug'] = highest_risk
                    self._log(f"   Highest Risk Contributor: {highest_risk[0].title()} (PRI: {highest_risk[1]:.4f})")
            
            # Stage 3: Interaction Detection
            self._update_stage(PipelineStage.INTERACTION_DETECTION)
            interaction_result = self.interaction_module.execute({'drugs': drugs})
            self.module_results['interaction'] = interaction_result
            
            if interaction_result.status == PipelineStatus.FAILED:
                self._log(f"Interaction detection issues: {interaction_result.errors}")
                errors.extend(interaction_result.errors)
            
            # Extract data for next stages
            accumulated_data.update(interaction_result.data)
            interactions = interaction_result.data.get('interactions', [])
            self._log(f"   Found {len(interactions)} interactions")
            
            # Stage 4: Severity Analysis
            self._update_stage(PipelineStage.SEVERITY_ANALYSIS)
            severity_result = self.severity_module.execute({
                'interactions': interactions
            })
            self.module_results['severity'] = severity_result
            
            if severity_result.status == PipelineStatus.SUCCESS:
                accumulated_data.update(severity_result.data)
                risk = severity_result.data.get('risk_assessment', {})
                self._log(f"   Risk Level: {risk.get('risk_level', 'N/A')}, Score: {risk.get('overall_score', 0)}")
            
            # Stage 5: Multi-Objective Alternative Finding (Paper methodology)
            self._update_stage(PipelineStage.MULTI_OBJECTIVE_RANKING)
            
            # Use paper's multi-objective recommender
            if self.recommender:
                self._log("   Running Multi-Objective Recommender (Paper Algorithm)...")
                mo_recommendations = self.recommender.recommend_for_polypharmacy(
                    drug_list=drugs,
                    max_replacements=3
                )
                accumulated_data['multi_objective_recommendations'] = mo_recommendations
                
                if mo_recommendations.get('recommendations'):
                    self._log(f"   Generated {len(mo_recommendations['recommendations'])} prioritized recommendations")
                    for rec in mo_recommendations['recommendations'][:2]:
                        if rec.get('best_alternative'):
                            self._log(f"     - Replace {rec['target_drug']} with {rec['best_alternative']['drug_name']}")
            
            # Stage 6: Alternative Finding (legacy approach for comparison)
            self._update_stage(PipelineStage.ALTERNATIVE_FINDING)
            
            # Identify problematic drugs from high-severity interactions
            problematic_drugs = set()
            for inter in severity_result.data.get('analyzed_interactions', []):
                if inter.get('severity_label') in ['Contraindicated interaction', 'Major interaction']:
                    problematic_drugs.add(inter.get('drug_1', '').lower())
                    problematic_drugs.add(inter.get('drug_2', '').lower())
            
            if problematic_drugs:
                self._log(f"   Searching alternatives for {len(problematic_drugs)} problematic drugs")
                alternative_result = self.alternative_module.execute({
                    'problematic_drugs': list(problematic_drugs),
                    'current_drugs': drugs,
                    'all_drugs': drugs,
                    'analyzed_interactions': severity_result.data.get('analyzed_interactions', [])
                })
                self.module_results['alternative'] = alternative_result
                
                if alternative_result.status == PipelineStatus.SUCCESS:
                    accumulated_data.update(alternative_result.data)
                    alts_found = len(alternative_result.data.get('best_alternatives', {}))
                    self._log(f"   Found alternatives for {alts_found} drugs")
            else:
                self._log("   No high-risk drugs requiring alternatives")
            
            # Stage 7: Report Generation
            self._update_stage(PipelineStage.REPORT_GENERATION)
            explanation_result = self.explanation_module.execute(accumulated_data)
            self.module_results['explanation'] = explanation_result
            
            if explanation_result.status == PipelineStatus.SUCCESS:
                accumulated_data.update(explanation_result.data)
                self._log("   Reports generated successfully")
            
            # Complete
            self._update_stage(PipelineStage.COMPLETED)
            
            # Calculate execution time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self._log("═" * 60)
            self._log(f"Pipeline completed in {duration:.2f} seconds")
            self._log("═" * 60)
            
            return Result(
                module_name=self.name,
                status=PipelineStatus.SUCCESS,
                data={
                    'pipeline_results': accumulated_data,
                    'module_results': {
                        name: {
                            'status': result.status.value,
                            'execution_time': result.execution_time
                        }
                        for name, result in self.module_results.items()
                    },
                    'execution_summary': {
                        'total_duration_seconds': duration,
                        'stages_completed': self.current_stage.name,
                        'drugs_analyzed': len(drugs),
                        'interactions_found': len(interactions),
                        'risk_level': severity_result.data.get('risk_assessment', {}).get('risk_level', 'N/A'),
                        'network_risk_level': accumulated_data.get('network_risk', {}).get('risk_level', 'N/A')
                    }
                },
                errors=errors if errors else None,
                metadata={
                    'execution_log': self.execution_log
                }
            )
            
        except Exception as e:
            self._update_stage(PipelineStage.FAILED)
            self._log(f"Pipeline failed: {str(e)}")
            
            return Result(
                module_name=self.name,
                status=PipelineStatus.FAILED,
                data={'partial_results': accumulated_data},
                errors=[str(e)],
                metadata={'execution_log': self.execution_log}
            )
    
    def analyze_drugs(self, drug_list: List[str]) -> Dict[str, Any]:
        """
        Convenience method for drug analysis
        
        Args:
            drug_list: List of drug names to analyze
            
        Returns:
            Complete analysis results
        """
        result = self.execute({'drugs': drug_list})
        return {
            'success': result.status == PipelineStatus.SUCCESS,
            'data': result.data,
            'errors': result.errors,
            'reports': result.data.get('pipeline_results', {}).get('clinical_report', ''),
            'patient_summary': result.data.get('pipeline_results', {}).get('patient_summary', ''),
            'structured_output': result.data.get('pipeline_results', {}).get('structured_output', {})
        }
    
    def get_quick_summary(self, drug_list: List[str]) -> str:
        """Get a quick text summary of drug interactions"""
        result = self.analyze_drugs(drug_list)
        if result['success']:
            return result['reports']
        else:
            return f"Analysis failed: {result['errors']}"
    
    def get_execution_log(self) -> List[str]:
        """Get the execution log"""
        return self.execution_log.copy()
    
    def reset(self):
        """Reset the orchestrator state for a new analysis"""
        self.current_stage = PipelineStage.INITIALIZATION
        self.execution_log = []
        self.module_results = {}
