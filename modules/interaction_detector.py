"""
Interaction Module - Detects drug-drug interactions
"""

import pandas as pd
from typing import Dict, Any, List
from collections import defaultdict

from .base_module import BaseModule, Result, PipelineStatus


class InteractionDetector(BaseModule):
    """
    Interaction Module
    
    Responsible for:
    - Building drug interaction index from DDI database
    - Detecting all pairwise interactions in a drug list
    - Providing detailed interaction information
    
    Input: List of drug names
    Output: All detected DDIs with details
    
    Components:
    - Drug validation
    - Pairwise interaction lookup
    - Severity classification
    """
    
    def __init__(self):
        super().__init__(
            name="InteractionDetector",
            description="Detects drug-drug interactions from a list of medications"
        )
        self.df = None
        self.interactions_index = defaultdict(list)
        self.drug_database = {}
        self.drug_names = set()
        
    def initialize(self, ddi_dataframe: pd.DataFrame) -> bool:
        """Initialize with DDI database"""
        print(f"[{self.name}] Initializing...")
        
        self.df = ddi_dataframe
        self._build_drug_database()
        self._build_interaction_index()
        
        self._initialized = True
        print(f"[{self.name}] Ready - {len(self.drug_names):,} drugs, {len(self.df):,} interactions")
        return True
    
    def _build_drug_database(self):
        """Build drug lookup database"""
        drugs_1 = self.df[['drugbank_id_1', 'drug_name_1', 'atc_1', 
                           'is_cardiovascular_1', 'is_antithrombotic_1']].copy()
        drugs_1.columns = ['drugbank_id', 'drug_name', 'atc', 'is_cardiovascular', 'is_antithrombotic']
        
        drugs_2 = self.df[['drugbank_id_2', 'drug_name_2', 'atc_2',
                           'is_cardiovascular_2', 'is_antithrombotic_2']].copy()
        drugs_2.columns = ['drugbank_id', 'drug_name', 'atc', 'is_cardiovascular', 'is_antithrombotic']
        
        drug_db = pd.concat([drugs_1, drugs_2]).drop_duplicates(subset=['drugbank_id'])
        
        # Combine cardiovascular and antithrombotic into single cardiovascular flag
        drug_db['is_cardiovascular'] = drug_db['is_cardiovascular'] | drug_db['is_antithrombotic']
        
        self.drug_names = set(drug_db['drug_name'].str.lower())
        self.name_to_id = dict(zip(drug_db['drug_name'].str.lower(), drug_db['drugbank_id']))
        
        for _, row in drug_db.iterrows():
            self.drug_database[row['drug_name'].lower()] = row.to_dict()
    
    def _build_interaction_index(self):
        """Build bidirectional interaction lookup index"""
        for _, row in self.df.iterrows():
            drug1 = row['drug_name_1'].lower()
            drug2 = row['drug_name_2'].lower()
            
            # Combine cardiovascular and antithrombotic flags
            is_cv_1 = row['is_cardiovascular_1'] or row['is_antithrombotic_1']
            is_cv_2 = row['is_cardiovascular_2'] or row['is_antithrombotic_2']
            
            interaction_data = {
                'drug_1': row['drug_name_1'],
                'drug_2': row['drug_name_2'],
                'drugbank_id_1': row['drugbank_id_1'],
                'drugbank_id_2': row['drugbank_id_2'],
                'description': row['interaction_description'],
                'severity_label': row['severity_label'],
                'severity_confidence': row['severity_confidence'],
                'severity_numeric': row['severity_numeric'],
                'is_cardiovascular_1': is_cv_1,
                'is_cardiovascular_2': is_cv_2
            }
            
            self.interactions_index[(drug1, drug2)].append(interaction_data)
            self.interactions_index[(drug2, drug1)].append(interaction_data)
    
    def validate_input(self, input_data: Dict[str, Any]) -> tuple:
        """Validate input contains drug list"""
        if 'drugs' not in input_data:
            return False, "Missing 'drugs' key in input"
        
        drugs = input_data['drugs']
        if not isinstance(drugs, list):
            return False, "'drugs' must be a list"
        
        if len(drugs) < 2:
            return False, "Need at least 2 drugs for interaction analysis"
        
        return True, ""
    
    def validate_drugs(self, drug_list: List[str]) -> Dict[str, Any]:
        """Validate and match drug names"""
        validated = []
        unrecognized = []
        
        for drug in drug_list:
            drug_lower = drug.lower().strip()
            if drug_lower in self.drug_names:
                validated.append({
                    'input_name': drug,
                    'normalized_name': drug_lower,
                    'drugbank_id': self.name_to_id.get(drug_lower),
                    'info': self.drug_database.get(drug_lower, {}),
                    'status': 'valid'
                })
            else:
                # Fuzzy matching
                matches = [d for d in self.drug_names if drug_lower in d or d in drug_lower]
                if matches:
                    best_match = matches[0]
                    validated.append({
                        'input_name': drug,
                        'normalized_name': best_match,
                        'matched_name': best_match.title(),
                        'drugbank_id': self.name_to_id.get(best_match),
                        'info': self.drug_database.get(best_match, {}),
                        'status': 'fuzzy_match'
                    })
                else:
                    unrecognized.append(drug)
        
        return {
            'validated': validated,
            'unrecognized': unrecognized,
            'valid_count': len(validated),
            'unrecognized_count': len(unrecognized)
        }
    
    def find_interaction(self, drug1: str, drug2: str) -> List[Dict]:
        """Find interactions between two drugs"""
        key = (drug1.lower(), drug2.lower())
        return self.interactions_index.get(key, [])
    
    def process(self, input_data: Dict[str, Any]) -> Result:
        """
        Main processing: Detect all drug-drug interactions
        """
        drugs = input_data['drugs']
        
        # Validate drugs first
        validation_result = self.validate_drugs(drugs)
        valid_drugs = [d['normalized_name'] for d in validation_result['validated']]
        
        if len(valid_drugs) < 2:
            return Result(
                module_name=self.name,
                status=PipelineStatus.FAILED,
                data={'validation': validation_result},
                errors=["Not enough valid drugs for interaction analysis"]
            )
        
        # Detect all pairwise interactions
        interactions_found = []
        pairs_checked = 0
        
        for i in range(len(valid_drugs)):
            for j in range(i + 1, len(valid_drugs)):
                drug1, drug2 = valid_drugs[i], valid_drugs[j]
                pairs_checked += 1
                
                interactions = self.find_interaction(drug1, drug2)
                for inter in interactions:
                    interactions_found.append({
                        'pair': f"{drug1.title()} ↔ {drug2.title()}",
                        'drug_1': drug1.title(),
                        'drug_2': drug2.title(),
                        **inter
                    })
        
        # Sort by severity (most severe first)
        severity_order = {
            'Contraindicated interaction': 0, 
            'Major interaction': 1, 
            'Moderate interaction': 2, 
            'Minor interaction': 3
        }
        interactions_found.sort(key=lambda x: severity_order.get(x['severity_label'], 4))
        
        # Compute severity summary
        severity_summary = {}
        for inter in interactions_found:
            sev = inter['severity_label']
            severity_summary[sev] = severity_summary.get(sev, 0) + 1
        
        return Result(
            module_name=self.name,
            status=PipelineStatus.SUCCESS,
            data={
                'validation': validation_result,
                'interactions': interactions_found,
                'summary': {
                    'total_interactions': len(interactions_found),
                    'drugs_analyzed': len(valid_drugs),
                    'pairs_checked': pairs_checked,
                    'severity_breakdown': severity_summary
                }
            },
            metadata={
                'valid_drug_names': valid_drugs
            }
        )
