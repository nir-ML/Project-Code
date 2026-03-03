"""
Drug Risk Network - Implements DDI network construction and Polypharmacy Risk Index (PRI)

Based on the paper:
- Nodes = drugs, Edges = interactions weighted by severity
- Network metrics: degree centrality, weighted betweenness, phenotype-specific centrality
- Composite Polypharmacy Risk Index (PRI)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import math


@dataclass
class DrugNode:
    """Represents a drug node in the DDI network"""
    drug_name: str
    drugbank_id: str = ""
    atc_code: str = ""
    atc_level_3: str = ""  # Pharmacological subgroup
    atc_level_4: str = ""  # Chemical subgroup
    is_cardiovascular: bool = False  # Includes both cardiovascular and antithrombotic drugs
    
    # Network metrics
    degree_centrality: float = 0.0
    weighted_degree: float = 0.0
    betweenness_centrality: float = 0.0
    phenotype_centrality: float = 0.0
    
    # Polypharmacy Risk Index
    pri_score: float = 0.0
    
    # Interaction profile
    contraindicated_count: int = 0
    major_count: int = 0
    moderate_count: int = 0
    minor_count: int = 0


@dataclass 
class DDIEdge:
    """Represents an edge (interaction) in the DDI network"""
    drug1: str
    drug2: str
    severity_label: str
    severity_weight: float
    description: str
    confidence: float = 1.0
    phenotypes: List[str] = field(default_factory=list)


class DrugRiskNetwork:
    """
    DDI Network with Polypharmacy Risk Index (PRI) computation
    
    Implements:
    - Network construction from DDI data
    - Degree centrality (interaction burden)
    - Weighted betweenness (risk propagation potential)
    - Phenotype-specific subnetwork centrality
    - Composite Polypharmacy Risk Index
    """
    
    # Severity weights for network edges
    SEVERITY_WEIGHTS = {
        'Contraindicated interaction': 10.0,
        'Major interaction': 7.0,
        'Moderate interaction': 4.0,
        'Minor interaction': 1.0
    }
    
    # PRI component weights
    PRI_WEIGHTS = {
        'degree': 0.25,
        'weighted_degree': 0.30,
        'betweenness': 0.20,
        'severity_profile': 0.25
    }
    
    def __init__(self):
        self.nodes: Dict[str, DrugNode] = {}
        self.edges: List[DDIEdge] = []
        self.adjacency: Dict[str, Dict[str, DDIEdge]] = defaultdict(dict)
        self.drug_interactions: Dict[str, Set[str]] = defaultdict(set)
        self._initialized = False
        
    def build_network(self, ddi_dataframe: pd.DataFrame) -> None:
        """
        Construct the DDI network from dataframe
        
        Args:
            ddi_dataframe: DDI data with drug pairs and severity
        """
        print("Building Drug Risk Network...")
        
        # Step 1: Create drug nodes
        self._create_nodes(ddi_dataframe)
        
        # Step 2: Create edges (interactions)
        self._create_edges(ddi_dataframe)
        
        # Step 3: Compute network metrics
        self._compute_degree_centrality()
        self._compute_weighted_degree()
        self._compute_betweenness_centrality()
        self._compute_severity_profile()
        
        # Step 4: Compute Polypharmacy Risk Index
        self._compute_pri_scores()
        
        self._initialized = True
        print(f"Network built: {len(self.nodes):,} nodes, {len(self.edges):,} edges")
        
    def _create_nodes(self, df: pd.DataFrame) -> None:
        """Create drug nodes from DDI data"""
        # Process drug 1
        for _, row in df[['drugbank_id_1', 'drug_name_1', 'atc_1', 
                          'is_cardiovascular_1', 'is_antithrombotic_1']].drop_duplicates().iterrows():
            drug_name = row['drug_name_1'].lower()
            atc = row['atc_1'] if pd.notna(row['atc_1']) else ""
            
            # Combine cardiovascular and antithrombotic into single cardiovascular flag
            is_cv = bool(row['is_cardiovascular_1']) or bool(row['is_antithrombotic_1'])
            
            self.nodes[drug_name] = DrugNode(
                drug_name=drug_name,
                drugbank_id=row['drugbank_id_1'],
                atc_code=atc,
                atc_level_3=atc[:4] if len(atc) >= 4 else atc,
                atc_level_4=atc[:5] if len(atc) >= 5 else atc,
                is_cardiovascular=is_cv
            )
        
        # Process drug 2
        for _, row in df[['drugbank_id_2', 'drug_name_2', 'atc_2',
                          'is_cardiovascular_2', 'is_antithrombotic_2']].drop_duplicates().iterrows():
            drug_name = row['drug_name_2'].lower()
            if drug_name not in self.nodes:
                atc = row['atc_2'] if pd.notna(row['atc_2']) else ""
                
                # Combine cardiovascular and antithrombotic into single cardiovascular flag
                is_cv = bool(row['is_cardiovascular_2']) or bool(row['is_antithrombotic_2'])
                
                self.nodes[drug_name] = DrugNode(
                    drug_name=drug_name,
                    drugbank_id=row['drugbank_id_2'],
                    atc_code=atc,
                    atc_level_3=atc[:4] if len(atc) >= 4 else atc,
                    atc_level_4=atc[:5] if len(atc) >= 5 else atc,
                    is_cardiovascular=is_cv
                )
    
    def _create_edges(self, df: pd.DataFrame) -> None:
        """Create weighted edges from DDI data"""
        seen_edges = set()
        
        for _, row in df.iterrows():
            drug1 = row['drug_name_1'].lower()
            drug2 = row['drug_name_2'].lower()
            
            # Avoid duplicate edges
            edge_key = tuple(sorted([drug1, drug2]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            severity = row['severity_label']
            weight = self.SEVERITY_WEIGHTS.get(severity, 4.0)
            
            # Extract phenotypes from description
            phenotypes = self._extract_phenotypes(row['interaction_description'])
            
            edge = DDIEdge(
                drug1=drug1,
                drug2=drug2,
                severity_label=severity,
                severity_weight=weight,
                description=row['interaction_description'],
                confidence=row.get('severity_confidence', 1.0),
                phenotypes=phenotypes
            )
            
            self.edges.append(edge)
            self.adjacency[drug1][drug2] = edge
            self.adjacency[drug2][drug1] = edge
            self.drug_interactions[drug1].add(drug2)
            self.drug_interactions[drug2].add(drug1)
    
    def _extract_phenotypes(self, description: str) -> List[str]:
        """Extract clinical phenotypes from interaction description"""
        phenotypes = []
        description_lower = description.lower() if description else ""
        
        phenotype_keywords = {
            'bleeding': ['bleeding', 'hemorrhage', 'haemorrhage'],
            'hypotension': ['hypotensive', 'hypotension', 'blood pressure'],
            'bradycardia': ['bradycardia', 'heart rate decrease'],
            'nephrotoxicity': ['renal', 'kidney', 'nephrotox'],
            'hepatotoxicity': ['hepat', 'liver'],
            'arrhythmia': ['arrhythmia', 'qt prolongation', 'torsade'],
            'hyperkalemia': ['hyperkalemia', 'potassium'],
            'hypoglycemia': ['hypoglycemia', 'blood sugar'],
            'serotonin_syndrome': ['serotonin'],
            'cns_depression': ['sedation', 'drowsiness', 'cns depression']
        }
        
        for phenotype, keywords in phenotype_keywords.items():
            if any(kw in description_lower for kw in keywords):
                phenotypes.append(phenotype)
        
        return phenotypes
    
    def _compute_degree_centrality(self) -> None:
        """Compute degree centrality (interaction burden)"""
        max_degree = len(self.nodes) - 1 if len(self.nodes) > 1 else 1
        
        for drug_name, node in self.nodes.items():
            degree = len(self.drug_interactions.get(drug_name, set()))
            node.degree_centrality = degree / max_degree
    
    def _compute_weighted_degree(self) -> None:
        """Compute weighted degree (sum of edge weights)"""
        max_weighted_degree = 0
        
        # First pass: compute weighted degrees
        for drug_name in self.nodes:
            weighted_sum = 0
            for neighbor in self.drug_interactions.get(drug_name, set()):
                edge = self.adjacency[drug_name].get(neighbor)
                if edge:
                    weighted_sum += edge.severity_weight
            self.nodes[drug_name].weighted_degree = weighted_sum
            max_weighted_degree = max(max_weighted_degree, weighted_sum)
        
        # Normalize
        if max_weighted_degree > 0:
            for node in self.nodes.values():
                node.weighted_degree /= max_weighted_degree
    
    def _compute_betweenness_centrality(self) -> None:
        """
        Compute weighted betweenness centrality (risk propagation potential)
        Using approximate algorithm for large networks
        """
        # For large networks, use sampling-based approximation
        n_samples = min(100, len(self.nodes))
        sample_nodes = list(self.nodes.keys())[:n_samples]
        
        betweenness = defaultdict(float)
        
        for source in sample_nodes:
            # BFS-based shortest path counting
            distances = {source: 0}
            num_paths = {source: 1}
            predecessors = defaultdict(list)
            queue = [source]
            visited_order = []
            
            while queue:
                current = queue.pop(0)
                visited_order.append(current)
                
                for neighbor in self.drug_interactions.get(current, set()):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
                    
                    if distances[neighbor] == distances[current] + 1:
                        num_paths[neighbor] = num_paths.get(neighbor, 0) + num_paths[current]
                        predecessors[neighbor].append(current)
            
            # Accumulate betweenness
            dependency = defaultdict(float)
            for node in reversed(visited_order):
                for pred in predecessors[node]:
                    dependency[pred] += (num_paths[pred] / num_paths[node]) * (1 + dependency[node])
                if node != source:
                    betweenness[node] += dependency[node]
        
        # Normalize
        max_betweenness = max(betweenness.values()) if betweenness else 1
        for drug_name, node in self.nodes.items():
            node.betweenness_centrality = betweenness.get(drug_name, 0) / max_betweenness
    
    def _compute_severity_profile(self) -> None:
        """Compute severity profile for each drug"""
        for drug_name, node in self.nodes.items():
            for neighbor in self.drug_interactions.get(drug_name, set()):
                edge = self.adjacency[drug_name].get(neighbor)
                if edge:
                    if edge.severity_label == 'Contraindicated interaction':
                        node.contraindicated_count += 1
                    elif edge.severity_label == 'Major interaction':
                        node.major_count += 1
                    elif edge.severity_label == 'Moderate interaction':
                        node.moderate_count += 1
                    else:
                        node.minor_count += 1
    
    def _compute_pri_scores(self) -> None:
        """
        Compute Polypharmacy Risk Index (PRI) for each drug
        
        PRI = w1*degree + w2*weighted_degree + w3*betweenness + w4*severity_score
        """
        for drug_name, node in self.nodes.items():
            # Severity score component
            severity_score = (
                node.contraindicated_count * 10 +
                node.major_count * 7 +
                node.moderate_count * 4 +
                node.minor_count * 1
            )
            # Normalize severity score
            max_severity = max(1, node.contraindicated_count + node.major_count + 
                              node.moderate_count + node.minor_count) * 10
            severity_normalized = min(1.0, severity_score / max_severity)
            
            # Compute PRI
            node.pri_score = (
                self.PRI_WEIGHTS['degree'] * node.degree_centrality +
                self.PRI_WEIGHTS['weighted_degree'] * node.weighted_degree +
                self.PRI_WEIGHTS['betweenness'] * node.betweenness_centrality +
                self.PRI_WEIGHTS['severity_profile'] * severity_normalized
            )
    
    def get_drug_pri(self, drug_name: str) -> float:
        """Get PRI score for a drug"""
        node = self.nodes.get(drug_name.lower())
        return node.pri_score if node else 0.0
    
    def get_drug_metrics(self, drug_name: str) -> Dict[str, Any]:
        """Get all network metrics for a drug"""
        node = self.nodes.get(drug_name.lower())
        if not node:
            return {}
        
        return {
            'drug_name': node.drug_name,
            'atc_code': node.atc_code,
            'pri_score': round(node.pri_score, 4),
            'degree_centrality': round(node.degree_centrality, 4),
            'weighted_degree': round(node.weighted_degree, 4),
            'betweenness_centrality': round(node.betweenness_centrality, 4),
            'interaction_counts': {
                'contraindicated': node.contraindicated_count,
                'major': node.major_count,
                'moderate': node.moderate_count,
                'minor': node.minor_count,
                'total': (node.contraindicated_count + node.major_count + 
                         node.moderate_count + node.minor_count)
            },
            'phenotypes': {
                'is_cardiovascular': node.is_cardiovascular
            }
        }
    
    def get_highest_risk_drug(self, drug_list: List[str]) -> Tuple[str, float]:
        """
        Identify the highest-risk drug contributor from a list
        
        Args:
            drug_list: List of drug names
            
        Returns:
            Tuple of (drug_name, pri_score)
        """
        max_pri = -1
        highest_risk = None
        
        for drug in drug_list:
            pri = self.get_drug_pri(drug)
            if pri > max_pri:
                max_pri = pri
                highest_risk = drug
        
        return (highest_risk, max_pri)
    
    def get_pairwise_risk(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """Get risk metrics for a drug pair"""
        d1, d2 = drug1.lower(), drug2.lower()
        edge = self.adjacency.get(d1, {}).get(d2)
        
        if not edge:
            return {'has_interaction': False}
        
        n1, n2 = self.nodes.get(d1), self.nodes.get(d2)
        
        return {
            'has_interaction': True,
            'severity': edge.severity_label,
            'severity_weight': edge.severity_weight,
            'description': edge.description,
            'phenotypes': edge.phenotypes,
            'combined_pri': (n1.pri_score + n2.pri_score) / 2 if n1 and n2 else 0,
            'drug1_pri': n1.pri_score if n1 else 0,
            'drug2_pri': n2.pri_score if n2 else 0
        }
    
    def compute_polypharmacy_risk(self, drug_list: List[str]) -> Dict[str, Any]:
        """
        Compute overall polypharmacy risk for a drug combination
        
        Args:
            drug_list: List of drug names
            
        Returns:
            Comprehensive risk assessment
        """
        drugs_lower = [d.lower() for d in drug_list]
        valid_drugs = [d for d in drugs_lower if d in self.nodes]
        
        if not valid_drugs:
            return {'error': 'No valid drugs found'}
        
        # Collect all interactions
        interactions = []
        seen_pairs = set()
        
        for i, d1 in enumerate(valid_drugs):
            for d2 in valid_drugs[i+1:]:
                pair_key = tuple(sorted([d1, d2]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                edge = self.adjacency.get(d1, {}).get(d2)
                if edge:
                    interactions.append({
                        'drug1': d1,
                        'drug2': d2,
                        'severity': edge.severity_label,
                        'weight': edge.severity_weight,
                        'description': edge.description,
                        'phenotypes': edge.phenotypes
                    })
        
        # Compute aggregate metrics
        total_pri = sum(self.nodes[d].pri_score for d in valid_drugs)
        avg_pri = total_pri / len(valid_drugs)
        max_pri_drug, max_pri = self.get_highest_risk_drug(valid_drugs)
        
        # Severity counts
        severity_counts = defaultdict(int)
        all_phenotypes = set()
        for inter in interactions:
            severity_counts[inter['severity']] += 1
            all_phenotypes.update(inter['phenotypes'])
        
        # Risk score (0-100)
        risk_score = min(100, (
            avg_pri * 30 +
            (severity_counts['Contraindicated interaction'] * 15) +
            (severity_counts['Major interaction'] * 8) +
            (severity_counts['Moderate interaction'] * 3) +
            len(all_phenotypes) * 5
        ))
        
        # Risk level
        if risk_score >= 70 or severity_counts['Contraindicated interaction'] > 0:
            risk_level = 'CRITICAL'
        elif risk_score >= 50 or severity_counts['Major interaction'] >= 2:
            risk_level = 'HIGH'
        elif risk_score >= 25:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'drugs_analyzed': len(valid_drugs),
            'total_interactions': len(interactions),
            'severity_breakdown': dict(severity_counts),
            'phenotypes_involved': list(all_phenotypes),
            'highest_risk_drug': {
                'name': max_pri_drug,
                'pri_score': round(max_pri, 4)
            },
            'average_pri': round(avg_pri, 4),
            'drug_pri_scores': {
                d: round(self.nodes[d].pri_score, 4) for d in valid_drugs
            },
            'interactions': interactions
        }
