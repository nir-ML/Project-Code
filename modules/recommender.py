"""
Drug Alternative Recommender

For a given high-risk drug pair or polypharmacy set, the recommender:
1. Identifies the highest-risk drug contributor
2. Retrieves same-ATC candidate alternatives (ATC Level 3 pharmacological equivalence)
3. Computes Alternative Recommendation Score (ARS)

ARS Formula (from paper):
    ARS = 0.70 × Severity_Reduction + 0.30 × ∆PRI
    
Where:
    - Severity_Reduction = (W_orig - W_alt) / W_orig
    - W = weighted severity score (Contraindicated=10, Major=7, Moderate=4, Minor=1)
    - ∆PRI = (PRI_orig - PRI_alt) / 40,000 (normalized PRI improvement)
"""

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from .drug_risk_network import DrugRiskNetwork, DrugNode


@dataclass
class AlternativeCandidate:
    """Represents a potential alternative drug"""
    drug_name: str
    atc_code: str
    atc_match_level: int  # 3 = pharmacological, 4 = chemical
    pri_score: float
    pri_delta: float  # Reduction in PRI (normalized)
    severity_reduction: float  # Relative severity reduction
    new_interactions: int  # Interactions with remaining drugs
    ars_score: float  # Alternative Recommendation Score


class AlternativeRecommender:
    """
    Drug alternative recommender using Alternative Recommendation Score (ARS)
    
    ARS = 0.70 × Severity_Reduction + 0.30 × ∆PRI
    
    Severity weights (from paper):
    - Contraindicated: 10
    - Major: 7
    - Moderate: 4
    - Minor: 1
    """
    
    # ARS weights from paper
    ARS_WEIGHTS = {
        'severity_reduction': 0.70,
        'pri_delta': 0.30
    }
    
    # Severity weights for DDI scoring
    SEVERITY_WEIGHTS = {
        'Contraindicated interaction': 10.0,
        'Major interaction': 7.0,
        'Moderate interaction': 4.0,
        'Minor interaction': 1.0
    }
    
    # PRI normalization constant
    PRI_NORMALIZATION = 40000
    
    def __init__(self, risk_network: DrugRiskNetwork):
        self.network = risk_network
        self.atc_to_drugs: Dict[str, List[str]] = defaultdict(list)
        self._build_atc_index()
    
    def _build_atc_index(self) -> None:
        """Build ATC code to drug mapping for therapeutic similarity"""
        for drug_name, node in self.network.nodes.items():
            if node.atc_level_3:
                self.atc_to_drugs[node.atc_level_3].append(drug_name)
            if node.atc_level_4:
                self.atc_to_drugs[node.atc_level_4].append(drug_name)
    
    def identify_highest_risk_contributor(self, 
                                          drug_list: List[str]) -> Tuple[str, Dict]:
        """
        Step 1: Identify the highest-risk drug contributor
        
        Returns:
            Tuple of (drug_name, risk_metrics)
        """
        drugs_lower = [d.lower() for d in drug_list]
        
        # Compute contribution to total risk for each drug
        contributions = []
        
        for drug in drugs_lower:
            if drug not in self.network.nodes:
                continue
            
            node = self.network.nodes[drug]
            
            # Count interactions with other drugs in the list
            list_interactions = 0
            severe_interactions = 0
            phenotypes_involved = set()
            
            for other in drugs_lower:
                if other == drug or other not in self.network.nodes:
                    continue
                
                edge = self.network.adjacency.get(drug, {}).get(other)
                if edge:
                    list_interactions += 1
                    if edge.severity_label in ['Contraindicated interaction', 'Major interaction']:
                        severe_interactions += 1
                    phenotypes_involved.update(edge.phenotypes)
            
            # Contribution score
            contribution_score = (
                node.pri_score * 0.4 +
                (severe_interactions / max(1, len(drugs_lower) - 1)) * 0.4 +
                node.weighted_degree * 0.2
            )
            
            contributions.append({
                'drug': drug,
                'pri_score': node.pri_score,
                'list_interactions': list_interactions,
                'severe_interactions': severe_interactions,
                'phenotypes': list(phenotypes_involved),
                'contribution_score': contribution_score
            })
        
        # Sort by contribution score
        contributions.sort(key=lambda x: -x['contribution_score'])
        
        if contributions:
            highest = contributions[0]
            return (highest['drug'], highest)
        
        return (None, {})
    
    def get_atc_alternatives(self, 
                            drug: str, 
                            level: int = 4) -> List[str]:
        """
        Step 2: Retrieve same-ATC candidate alternatives
        
        Args:
            drug: Drug to find alternatives for
            level: ATC level (3 = pharmacological, 4 = chemical)
            
        Returns:
            List of alternative drug names
        """
        drug_lower = drug.lower()
        if drug_lower not in self.network.nodes:
            return []
        
        node = self.network.nodes[drug_lower]
        
        # Get ATC prefix at specified level
        if level == 4:
            atc_prefix = node.atc_level_4
        else:
            atc_prefix = node.atc_level_3
        
        if not atc_prefix:
            return []
        
        # Get all drugs with same ATC prefix
        alternatives = [
            d for d in self.atc_to_drugs.get(atc_prefix, [])
            if d != drug_lower
        ]
        
        return alternatives
    
    def compute_replacement_delta(self,
                                  original_drug: str,
                                  alternative_drug: str,
                                  current_drugs: List[str]) -> Dict[str, Any]:
        """
        Compute replacement metrics for ARS calculation
        
        Args:
            original_drug: Drug being replaced
            alternative_drug: Proposed alternative
            current_drugs: Full list of current drugs
            
        Returns:
            Delta metrics including severity weights for ARS
        """
        orig_lower = original_drug.lower()
        alt_lower = alternative_drug.lower()
        other_drugs = [d.lower() for d in current_drugs if d.lower() != orig_lower]
        
        orig_node = self.network.nodes.get(orig_lower)
        alt_node = self.network.nodes.get(alt_lower)
        
        if not orig_node or not alt_node:
            return {}
        
        # PRI delta (normalized by 40,000)
        pri_delta = (orig_node.pri_score - alt_node.pri_score) / self.PRI_NORMALIZATION
        
        # Calculate weighted severity sums for ARS
        orig_severity_sum = 0.0
        alt_severity_sum = 0.0
        orig_interactions = []
        alt_interactions = []
        
        for other in other_drugs:
            # Original drug's interactions
            edge = self.network.adjacency.get(orig_lower, {}).get(other)
            if edge:
                weight = self.SEVERITY_WEIGHTS.get(edge.severity_label, 1.0)
                orig_severity_sum += weight
                orig_interactions.append({
                    'drug': other,
                    'severity': edge.severity_label,
                    'weight': weight
                })
            
            # Alternative drug's interactions
            alt_edge = self.network.adjacency.get(alt_lower, {}).get(other)
            if alt_edge:
                weight = self.SEVERITY_WEIGHTS.get(alt_edge.severity_label, 1.0)
                alt_severity_sum += weight
                alt_interactions.append({
                    'drug': other,
                    'severity': alt_edge.severity_label,
                    'weight': weight
                })
        
        # Severity reduction (relative)
        if orig_severity_sum > 0:
            severity_reduction = (orig_severity_sum - alt_severity_sum) / orig_severity_sum
        else:
            severity_reduction = 0.0
        
        return {
            'pri_delta': pri_delta,
            'severity_reduction': severity_reduction,
            'original_severity_sum': orig_severity_sum,
            'alt_severity_sum': alt_severity_sum,
            'original_interactions': len(orig_interactions),
            'new_interactions': len(alt_interactions),
            'interaction_delta': len(orig_interactions) - len(alt_interactions)
        }
    
    def compute_ars(self, delta: Dict[str, Any]) -> float:
        """
        Compute Alternative Recommendation Score (ARS)
        
        ARS = 0.70 × Severity_Reduction + 0.30 × ∆PRI
        
        Higher score = better alternative
        """
        severity_reduction = max(0, delta.get('severity_reduction', 0))
        pri_delta = max(0, delta.get('pri_delta', 0))
        
        ars = (
            self.ARS_WEIGHTS['severity_reduction'] * severity_reduction +
            self.ARS_WEIGHTS['pri_delta'] * pri_delta
        )
        
        return ars
    
    def recommend_alternatives(self,
                              drug_list: List[str],
                              target_drug: str = None,
                              max_alternatives: int = 5) -> Dict[str, Any]:
        """
        Main recommender function
        
        Args:
            drug_list: Current drug list
            target_drug: Specific drug to replace (if None, uses highest risk)
            max_alternatives: Maximum alternatives to return
            
        Returns:
            Recommendations ranked by ARS (Alternative Recommendation Score)
        """
        drugs_lower = [d.lower() for d in drug_list]
        
        # Step 1: Identify target drug
        if target_drug:
            target = target_drug.lower()
            target_metrics = self.network.get_drug_metrics(target)
        else:
            target, target_metrics = self.identify_highest_risk_contributor(drug_list)
        
        if not target:
            return {'error': 'No valid target drug identified'}
        
        # Step 2: Get ATC alternatives (try level 4 first, then level 3)
        alternatives_l4 = self.get_atc_alternatives(target, level=4)
        alternatives_l3 = self.get_atc_alternatives(target, level=3)
        
        all_alternatives = list(set(alternatives_l4 + alternatives_l3))
        
        if not all_alternatives:
            return {
                'target_drug': target,
                'target_metrics': target_metrics,
                'alternatives': [],
                'message': 'No ATC-matched alternatives found'
            }
        
        # Step 3: Evaluate each alternative and compute ARS
        candidates = []
        
        for alt in all_alternatives:
            # Skip if alternative is already in the drug list
            if alt in drugs_lower:
                continue
            
            alt_node = self.network.nodes.get(alt)
            if not alt_node:
                continue
            
            # Compute delta metrics
            delta = self.compute_replacement_delta(target, alt, drug_list)
            
            if not delta:
                continue
            
            # Compute ARS score
            ars_score = self.compute_ars(delta)
            
            # Determine ATC match level
            target_node = self.network.nodes.get(target)
            if target_node and alt_node.atc_level_4 == target_node.atc_level_4:
                atc_level = 4
            else:
                atc_level = 3
            
            candidate = AlternativeCandidate(
                drug_name=alt,
                atc_code=alt_node.atc_code,
                atc_match_level=atc_level,
                pri_score=alt_node.pri_score,
                pri_delta=delta.get('pri_delta', 0),
                severity_reduction=delta.get('severity_reduction', 0),
                new_interactions=delta.get('new_interactions', 0),
                ars_score=ars_score
            )
            
            candidates.append((candidate, delta))
        
        # Rank by ARS score (higher = better)
        candidates.sort(key=lambda x: -x[0].ars_score)
        
        # Format output
        recommendations = []
        for candidate, delta in candidates[:max_alternatives]:
            recommendations.append({
                'drug_name': candidate.drug_name.title(),
                'atc_code': candidate.atc_code,
                'atc_match_level': candidate.atc_match_level,
                'atc_match_type': 'Chemical subgroup' if candidate.atc_match_level == 4 else 'Pharmacological subgroup',
                'ars_score': round(candidate.ars_score, 4),
                'pri_score': round(candidate.pri_score, 4),
                'risk_metrics': {
                    'severity_reduction': round(candidate.severity_reduction, 4),
                    'pri_delta': round(candidate.pri_delta, 4),
                    'interaction_delta': delta.get('interaction_delta', 0)
                },
                'new_interactions_with_current': candidate.new_interactions
            })
        
        return {
            'target_drug': {
                'name': target.title(),
                'pri_score': round(self.network.nodes[target].pri_score, 4) if target in self.network.nodes else 0,
                'reason': 'Highest risk contributor' if not target_drug else 'User specified'
            },
            'current_drugs': [d.title() for d in drugs_lower],
            'alternatives': recommendations,
            'total_candidates_evaluated': len(all_alternatives),
            'ranking_formula': 'ARS = 0.70 x Severity_Reduction + 0.30 x Delta_PRI'
        }
    
    def recommend_for_polypharmacy(self,
                                   drug_list: List[str],
                                   max_replacements: int = 3) -> Dict[str, Any]:
        """
        Recommend alternatives for multiple high-risk drugs
        
        Args:
            drug_list: Current drug list
            max_replacements: Maximum drugs to suggest replacing
            
        Returns:
            Comprehensive polypharmacy recommendations
        """
        drugs_lower = [d.lower() for d in drug_list]
        
        # Get risk analysis
        risk_analysis = self.network.compute_polypharmacy_risk(drug_list)
        
        # Sort drugs by contribution to risk
        drug_risks = []
        for drug in drugs_lower:
            if drug not in self.network.nodes:
                continue
            
            node = self.network.nodes[drug]
            
            # Count severe interactions
            severe_count = 0
            for other in drugs_lower:
                if other == drug:
                    continue
                edge = self.network.adjacency.get(drug, {}).get(other)
                if edge and edge.severity_label in ['Contraindicated interaction', 'Major interaction']:
                    severe_count += 1
            
            drug_risks.append({
                'drug': drug,
                'pri': node.pri_score,
                'severe_interactions': severe_count,
                'combined_risk': node.pri_score + (severe_count * 0.2)
            })
        
        drug_risks.sort(key=lambda x: -x['combined_risk'])
        
        # Generate recommendations for top risk contributors
        all_recommendations = []
        
        for i, drug_risk in enumerate(drug_risks[:max_replacements]):
            drug = drug_risk['drug']
            
            # Get alternatives for this drug
            rec = self.recommend_alternatives(
                drug_list=drug_list,
                target_drug=drug,
                max_alternatives=3
            )
            
            if rec.get('alternatives'):
                all_recommendations.append({
                    'priority': i + 1,
                    'target_drug': drug.title(),
                    'risk_contribution': round(drug_risk['combined_risk'], 4),
                    'severe_interactions_in_list': drug_risk['severe_interactions'],
                    'best_alternative': rec['alternatives'][0] if rec['alternatives'] else None,
                    'all_alternatives': rec['alternatives']
                })
        
        return {
            'overall_risk': {
                'score': risk_analysis.get('risk_score', 0),
                'level': risk_analysis.get('risk_level', 'UNKNOWN'),
                'total_interactions': risk_analysis.get('total_interactions', 0),
                'severity_breakdown': risk_analysis.get('severity_breakdown', {})
            },
            'recommendations': all_recommendations,
            'summary': {
                'drugs_analyzed': len(drugs_lower),
                'drugs_with_alternatives': len(all_recommendations),
                'estimated_risk_reduction': self._estimate_risk_reduction(all_recommendations)
            }
        }
    
    def _estimate_risk_reduction(self, recommendations: List[Dict]) -> float:
        """Estimate potential risk reduction from recommendations"""
        total_reduction = 0
        for rec in recommendations:
            if rec.get('best_alternative'):
                pri_red = rec['best_alternative'].get('risk_metrics', {}).get('pri_reduction', 0)
                total_reduction += max(0, pri_red)
        return round(total_reduction, 4)
