"""
Alternative Module - Finds safer drug alternatives
"""

import pandas as pd
from typing import Dict, Any, List, Set
from collections import defaultdict

from .base_module import BaseModule, Result, PipelineStatus


class AlternativeFinder(BaseModule):
    """
    Alternative Module
    
    Responsible for:
    - Finding therapeutic alternatives for problematic drugs
    - ATC-based drug class matching
    - Interaction-aware recommendations
    
    Input: Problematic drugs and interactions
    Output: Safer alternative drugs
    
    Components:
    - ATC class-based matching
    - Interaction profile comparison
    - Safety scoring
    """
    
    ATC_LEVELS = {
        1: 'Anatomical main group',
        2: 'Therapeutic subgroup', 
        3: 'Pharmacological subgroup',
        4: 'Chemical subgroup',
        5: 'Chemical substance'
    }
    
    def __init__(self):
        super().__init__(
            name="AlternativeFinder",
            description="Finds safer drug alternatives based on ATC classification"
        )
        self.df = None
        self.drug_atc_map = {}
        self.atc_to_drugs = defaultdict(set)
        self.drug_interactions = defaultdict(set)
        self.drug_severity_profile = defaultdict(lambda: defaultdict(int))
        
    def initialize(self, ddi_dataframe: pd.DataFrame) -> bool:
        """Initialize with DDI database"""
        print(f"[{self.name}] Initializing...")
        
        self.df = ddi_dataframe
        self._build_atc_index()
        self._build_interaction_profiles()
        
        self._initialized = True
        print(f"[{self.name}] Ready - {len(self.drug_atc_map):,} drugs indexed")
        return True
    
    def _build_atc_index(self):
        """Build ATC code to drug mapping"""
        # Process drug 1
        for _, row in self.df[['drug_name_1', 'atc_1']].drop_duplicates().iterrows():
            drug = row['drug_name_1'].lower()
            atc = row['atc_1']
            if pd.notna(atc) and atc:
                self.drug_atc_map[drug] = atc
                # Index at multiple ATC levels
                for level in [1, 3, 4, 5]:
                    prefix = atc[:level] if len(atc) >= level else atc
                    self.atc_to_drugs[prefix].add(drug)
        
        # Process drug 2
        for _, row in self.df[['drug_name_2', 'atc_2']].drop_duplicates().iterrows():
            drug = row['drug_name_2'].lower()
            atc = row['atc_2']
            if pd.notna(atc) and atc:
                self.drug_atc_map[drug] = atc
                for level in [1, 3, 4, 5]:
                    prefix = atc[:level] if len(atc) >= level else atc
                    self.atc_to_drugs[prefix].add(drug)
    
    def _build_interaction_profiles(self):
        """Build interaction and severity profiles for each drug"""
        for _, row in self.df.iterrows():
            drug1 = row['drug_name_1'].lower()
            drug2 = row['drug_name_2'].lower()
            severity = row['severity_label']
            
            # Track interactions
            self.drug_interactions[drug1].add(drug2)
            self.drug_interactions[drug2].add(drug1)
            
            # Track severity profile
            self.drug_severity_profile[drug1][severity] += 1
            self.drug_severity_profile[drug2][severity] += 1
    
    def validate_input(self, input_data: Dict[str, Any]) -> tuple:
        """Validate input"""
        if 'problematic_drugs' not in input_data and 'analyzed_interactions' not in input_data:
            return False, "Need 'problematic_drugs' or 'analyzed_interactions'"
        return True, ""
    
    def get_atc_level(self, atc: str, level: int) -> str:
        """Get ATC code at specific level"""
        if not atc or pd.isna(atc):
            return ""
        # ATC levels: 1(A), 3(A10), 4(A10B), 5(A10BA), 7(A10BA02)
        level_lengths = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
        return atc[:level_lengths.get(level, len(atc))]
    
    def find_same_class_drugs(self, drug: str, level: int = 3) -> List[Dict]:
        """Find drugs in the same ATC class"""
        drug_lower = drug.lower()
        atc = self.drug_atc_map.get(drug_lower)
        
        if not atc:
            return []
        
        level_lengths = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
        prefix = atc[:level_lengths.get(level, 4)]
        
        alternatives = []
        for alt_drug in self.atc_to_drugs.get(prefix, set()):
            if alt_drug != drug_lower:
                alternatives.append({
                    'drug_name': alt_drug.title(),
                    'atc': self.drug_atc_map.get(alt_drug, ''),
                    'matching_level': level,
                    'interactions_count': len(self.drug_interactions.get(alt_drug, set())),
                    'severity_profile': dict(self.drug_severity_profile.get(alt_drug, {}))
                })
        
        return alternatives
    
    def calculate_safety_score(self, drug: str, current_drugs: List[str]) -> float:
        """Calculate safety score for a potential alternative drug"""
        drug_lower = drug.lower()
        current_lower = [d.lower() for d in current_drugs]
        
        # Check interactions with current drugs
        drug_inters = self.drug_interactions.get(drug_lower, set())
        conflicts = drug_inters & set(current_lower)
        
        # Base score
        score = 100.0
        
        # Penalize for each interaction with current drugs
        score -= len(conflicts) * 15
        
        # Penalize based on severity profile
        profile = self.drug_severity_profile.get(drug_lower, {})
        score -= profile.get('Contraindicated interaction', 0) * 5
        score -= profile.get('Major interaction', 0) * 2
        score -= profile.get('Moderate interaction', 0) * 0.5
        
        return max(0, min(100, score))
    
    def find_alternatives(self, drug: str, avoid_drugs: Set[str], 
                         current_drugs: List[str]) -> List[Dict]:
        """Find alternatives for a specific drug"""
        drug_lower = drug.lower()
        alternatives = []
        
        # Search at different ATC levels (from specific to general)
        for level in [4, 3, 2]:
            same_class = self.find_same_class_drugs(drug, level)
            
            for alt in same_class:
                alt_lower = alt['drug_name'].lower()
                
                # Skip if in avoid list or is current drug
                if alt_lower in avoid_drugs or alt_lower == drug_lower:
                    continue
                
                # Calculate safety score
                safety_score = self.calculate_safety_score(alt['drug_name'], current_drugs)
                
                # Check for interactions with current drugs
                alt_interactions = self.drug_interactions.get(alt_lower, set())
                conflicts = alt_interactions & set(d.lower() for d in current_drugs if d.lower() != drug_lower)
                
                alternatives.append({
                    **alt,
                    'original_drug': drug.title(),
                    'safety_score': round(safety_score, 2),
                    'conflicts_with_current': [c.title() for c in conflicts],
                    'conflict_count': len(conflicts),
                    'atc_match_level': level,
                    'atc_match_description': self.ATC_LEVELS.get(level, 'Unknown')
                })
        
        # Sort by safety score, then by conflict count
        alternatives.sort(key=lambda x: (-x['safety_score'], x['conflict_count']))
        
        # Remove duplicates, keeping best version
        seen = set()
        unique_alts = []
        for alt in alternatives:
            if alt['drug_name'].lower() not in seen:
                seen.add(alt['drug_name'].lower())
                unique_alts.append(alt)
        
        return unique_alts[:10]  # Top 10 alternatives
    
    def process(self, input_data: Dict[str, Any]) -> Result:
        """
        Main processing: Find alternatives for problematic drugs
        """
        # Get list of drugs to find alternatives for
        if 'problematic_drugs' in input_data:
            problematic = input_data['problematic_drugs']
        else:
            # Extract from interactions - focus on high-severity drugs
            interactions = input_data.get('analyzed_interactions', [])
            problematic = set()
            for inter in interactions:
                if inter.get('severity_label') in ['Contraindicated interaction', 'Major interaction']:
                    problematic.add(inter.get('drug_1', inter.get('drug_name_1', '')).lower())
                    problematic.add(inter.get('drug_2', inter.get('drug_name_2', '')).lower())
            problematic = list(problematic)
        
        current_drugs = input_data.get('current_drugs', [])
        if not current_drugs:
            current_drugs = input_data.get('all_drugs', [])
        
        # Find alternatives for each problematic drug
        all_alternatives = {}
        drugs_without_alternatives = []
        
        avoid_set = set(d.lower() for d in current_drugs)
        
        for drug in problematic:
            if not drug:
                continue
            alts = self.find_alternatives(drug, avoid_set, current_drugs)
            if alts:
                all_alternatives[drug.title()] = alts
            else:
                drugs_without_alternatives.append(drug.title())
        
        # Summary statistics
        total_alternatives = sum(len(alts) for alts in all_alternatives.values())
        best_alternatives = {}
        for drug, alts in all_alternatives.items():
            if alts:
                best_alternatives[drug] = alts[0]  # Best alternative for each drug
        
        return Result(
            module_name=self.name,
            status=PipelineStatus.SUCCESS,
            data={
                'alternatives': all_alternatives,
                'best_alternatives': best_alternatives,
                'drugs_without_alternatives': drugs_without_alternatives,
                'summary': {
                    'problematic_drugs_count': len(problematic),
                    'drugs_with_alternatives': len(all_alternatives),
                    'total_alternatives_found': total_alternatives,
                    'drugs_without_alternatives': len(drugs_without_alternatives)
                }
            },
            metadata={
                'atc_levels_searched': [4, 3, 2]
            }
        )
