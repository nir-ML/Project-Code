#!/usr/bin/env python3
"""
DDI Risk Analysis Application with Alternative Recommendations
Three-column design:
  COL1: Drug input (list, narrative, or image)
  COL2: Analysis report with severity, risk & safer alternatives
  COL3: Chat with selected LLM to learn more
"""

import gradio as gr
import pandas as pd
import time
import json
import os
import re
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

# Optional: Image processing for prescription photos
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ============================================================
# Knowledge Graph Database with Alternative Recommendations
# ============================================================

class KnowledgeGraph:
    """Drug-Drug Interaction Knowledge Graph with Alternative Drug Finder"""
    
    def __init__(self):
        self.drugs = {}
        self.drugs_by_id = {}
        self.ddis = []
        self.ddi_index = {}  # {(drug1_id, drug2_id): ddi_info}
        self.side_effects = {}
        self.proteins = {}
        self.atc_index = defaultdict(list)  # {atc_prefix: [drug_ids]}
        self.loaded = False
        
        # Common drug name aliases (brand -> generic)
        self.aliases = {
            'aspirin': 'acetylsalicylic acid', 'tylenol': 'acetaminophen',
            'advil': 'ibuprofen', 'motrin': 'ibuprofen', 'coumadin': 'warfarin',
            'lipitor': 'atorvastatin', 'zocor': 'simvastatin', 'plavix': 'clopidogrel',
            'nexium': 'esomeprazole', 'prilosec': 'omeprazole', 'zoloft': 'sertraline',
            'prozac': 'fluoxetine', 'xanax': 'alprazolam', 'valium': 'diazepam',
            # ACE inhibitors
            'cardace': 'ramipril', 'altace': 'ramipril', 'tritace': 'ramipril',
            'vasotec': 'enalapril', 'zestril': 'lisinopril', 'prinivil': 'lisinopril',
            'lotensin': 'benazepril', 'capoten': 'captopril',
            # Beta blockers
            'lopressor': 'metoprolol', 'toprol': 'metoprolol', 'tenormin': 'atenolol',
            'coreg': 'carvedilol', 'inderal': 'propranolol',
            # Calcium channel blockers
            'norvasc': 'amlodipine', 'cardizem': 'diltiazem', 'calan': 'verapamil',
            # Diuretics
            'lasix': 'furosemide', 'bumex': 'bumetanide', 'demadex': 'torsemide',
            # Statins
            'crestor': 'rosuvastatin', 'lescol': 'fluvastatin', 'pravachol': 'pravastatin',
            # Anticoagulants
            'eliquis': 'apixaban', 'xarelto': 'rivaroxaban', 'pradaxa': 'dabigatran',
            # Diabetes
            'glucophage': 'metformin', 'januvia': 'sitagliptin', 'jardiance': 'empagliflozin',
            # Pain
            'celebrex': 'celecoxib', 'voltaren': 'diclofenac', 'aleve': 'naproxen',
            'ultram': 'tramadol', 'vicodin': 'hydrocodone', 'percocet': 'oxycodone',
            # Antibiotics
            'augmentin': 'amoxicillin', 'zithromax': 'azithromycin', 'cipro': 'ciprofloxacin',
            # GI
            'pepcid': 'famotidine', 'zantac': 'ranitidine', 'prevacid': 'lansoprazole',
        }
        
        # Severity weights for risk calculation
        self.severity_weights = {
            'contraindicated': 1.0,
            'major': 0.7,
            'moderate': 0.4,
            'minor': 0.15,
            'unknown': 0.2
        }
        
        # Drug name list for fuzzy matching
        self.drug_names = []
    
    def load(self):
        """Load knowledge graph from CSV files"""
        base = "knowledge_graph_fact_based/neo4j_export"
        
        # Load drugs
        if os.path.exists(f"{base}/drugs.csv"):
            df = pd.read_csv(f"{base}/drugs.csv", low_memory=False)
            for _, row in df.iterrows():
                name = str(row.get('name', '')).lower().strip()
                drug_id = row.get('drugbank_id', '')
                if name and drug_id:
                    drug_data = dict(row)
                    self.drugs[name] = drug_data
                    self.drugs_by_id[drug_id] = drug_data
                    
                    # Build ATC index for finding alternatives
                    atc = str(row.get('atc_codes', ''))
                    if atc and atc != 'nan':
                        for code in atc.split('|'):
                            if len(code) >= 4:
                                # Index by first 4 chars (therapeutic subgroup)
                                prefix = code[:4]
                                self.atc_index[prefix].append(drug_id)
                                # Also index by first 5 chars (chemical subgroup)
                                if len(code) >= 5:
                                    prefix5 = code[:5]
                                    self.atc_index[prefix5].append(drug_id)
        
        # Load DDIs and build index
        if os.path.exists(f"{base}/ddi_edges.csv"):
            df = pd.read_csv(f"{base}/ddi_edges.csv", low_memory=False)
            self.ddis = df.to_dict('records')
            
            # Build DDI lookup index
            for ddi in self.ddis:
                d1 = ddi.get('drug1_id', ddi.get('source', ''))
                d2 = ddi.get('drug2_id', ddi.get('target', ''))
                if d1 and d2:
                    # Store both directions for easy lookup
                    self.ddi_index[(d1, d2)] = ddi
                    self.ddi_index[(d2, d1)] = ddi
        
        # Load side effects
        if os.path.exists(f"{base}/side_effect_edges.csv"):
            df = pd.read_csv(f"{base}/side_effect_edges.csv", low_memory=False)
            for _, row in df.iterrows():
                drug_id = row.get('drug_id', '')
                if drug_id not in self.side_effects:
                    self.side_effects[drug_id] = []
                self.side_effects[drug_id].append({
                    'name': row.get('side_effect_name', row.get('umls_name', '')),
                    'umls_cui': row.get('umls_cui', ''),
                })
        
        # Load proteins
        if os.path.exists(f"{base}/drug_protein_edges.csv"):
            df = pd.read_csv(f"{base}/drug_protein_edges.csv", low_memory=False)
            for _, row in df.iterrows():
                drug_id = row.get('drug_id', '')
                if drug_id not in self.proteins:
                    self.proteins[drug_id] = []
                self.proteins[drug_id].append({
                    'name': row.get('protein_name', ''),
                    'gene': row.get('gene_name', ''),
                })
        
        # Build list of drug names for fuzzy matching
        self.drug_names = list(self.drugs.keys())
        
        self.loaded = True
        return f"Loaded {len(self.drugs):,} drugs, {len(self.ddis):,} DDIs, {len(self.atc_index)} ATC groups"
    
    def parse_drug_input(self, raw_input):
        """Parse drug input with various separators: +, comma, newline, 'and'"""
        # Normalize separators
        text = raw_input.lower()
        # Replace various separators with comma
        text = re.sub(r'\s*\+\s*', ',', text)  # + separator
        text = re.sub(r'\s+and\s+', ',', text)  # "and" separator
        text = re.sub(r'\s*[;|]\s*', ',', text)  # semicolon, pipe
        text = re.sub(r'\n+', ',', text)  # newlines
        
        # Split and clean
        drugs = [d.strip() for d in text.split(',') if d.strip()]
        # Remove duplicates while preserving order
        seen = set()
        unique_drugs = []
        for d in drugs:
            if d not in seen:
                seen.add(d)
                unique_drugs.append(d)
        return unique_drugs
    
    def fuzzy_match(self, query, threshold=0.6):
        """Find closest drug name matches using string similarity"""
        query = query.lower().strip()
        
        # First check exact match
        if query in self.drugs:
            return [(query, 1.0, self.drugs[query])]
        
        # Check aliases
        if query in self.aliases:
            alias_name = self.aliases[query]
            if alias_name in self.drugs:
                return [(alias_name, 1.0, self.drugs[alias_name])]
        
        # Check partial match
        for name in self.drug_names:
            if query in name or name in query:
                return [(name, 0.95, self.drugs[name])]
        
        # Fuzzy match using SequenceMatcher
        matches = []
        for name in self.drug_names:
            ratio = SequenceMatcher(None, query, name).ratio()
            if ratio >= threshold:
                matches.append((name, ratio, self.drugs[name]))
        
        # Sort by similarity
        matches.sort(key=lambda x: -x[1])
        return matches[:5]  # Return top 5 matches
    
    def identify_drugs(self, raw_input):
        """
        Parse input and identify drugs with fuzzy matching.
        Returns: (found_drugs, suggestions, not_found)
        """
        parsed_names = self.parse_drug_input(raw_input)
        
        found_drugs = []  # List of (input_name, matched_name, drug_data)
        suggestions = {}  # {input_name: [(match_name, score, drug_data), ...]}
        not_found = []    # Names with no good matches
        
        for name in parsed_names:
            matches = self.fuzzy_match(name)
            
            if not matches:
                not_found.append(name)
            elif matches[0][1] >= 0.95:  # High confidence match
                found_drugs.append((name, matches[0][0], matches[0][2]))
            elif matches[0][1] >= 0.6:   # Possible matches - need confirmation
                suggestions[name] = matches
            else:
                not_found.append(name)
        
        return found_drugs, suggestions, not_found
    
    def resolve(self, name):
        """Resolve drug name to database entry"""
        n = name.lower().strip()
        if n in self.aliases:
            n = self.aliases[n]
        if n in self.drugs:
            return self.drugs[n]
        for k, v in self.drugs.items():
            if n in k or k in n:
                return v
        return None
    
    def get_severity_weight(self, severity_str):
        """Get numerical weight for severity string"""
        sev = severity_str.lower() if severity_str else ''
        for key, weight in self.severity_weights.items():
            if key in sev:
                return weight
        return 0.2  # default unknown
    
    def get_interaction(self, drug1_id, drug2_id):
        """Get interaction between two drugs"""
        return self.ddi_index.get((drug1_id, drug2_id))
    
    def get_interactions(self, drug_ids):
        """Find all interactions between a set of drugs"""
        interactions = []
        id_list = list(drug_ids)
        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                ddi = self.get_interaction(id_list[i], id_list[j])
                if ddi:
                    interactions.append(ddi)
        return interactions
    
    def calculate_risk_score(self, drug_ids):
        """Calculate polypharmacy risk score for a set of drugs"""
        interactions = self.get_interactions(drug_ids)
        if not interactions:
            return 0.0, [], {}
        
        score = 0.0
        counts = {'contraindicated': 0, 'major': 0, 'moderate': 0, 'minor': 0}
        
        for i in interactions:
            sev = i.get('severity', '').lower()
            weight = self.get_severity_weight(sev)
            score += weight
            
            for key in counts:
                if key in sev:
                    counts[key] += 1
                    break
        
        # Normalize score (max 1.0)
        score = min(score / max(len(interactions), 1), 1.0)
        return score, interactions, counts
    
    def find_alternatives(self, drug_id, other_drug_ids, max_alternatives=5):
        """Find alternative drugs with lower interaction risk"""
        drug_data = self.drugs_by_id.get(drug_id, {})
        if not drug_data:
            return []
        
        atc = str(drug_data.get('atc_codes', ''))
        if not atc or atc == 'nan':
            return []
        
        # Get ATC prefixes for this drug
        prefixes = []
        for code in atc.split('|'):
            if len(code) >= 5:
                prefixes.append(code[:5])
            if len(code) >= 4:
                prefixes.append(code[:4])
        
        # Find candidate alternatives from same therapeutic class
        candidates = set()
        for prefix in prefixes:
            for cand_id in self.atc_index.get(prefix, []):
                if cand_id != drug_id and cand_id not in other_drug_ids:
                    candidates.add(cand_id)
        
        if not candidates:
            return []
        
        # Calculate risk for each alternative
        original_risk, _, _ = self.calculate_risk_score(set(other_drug_ids) | {drug_id})
        
        alternatives = []
        for cand_id in candidates:
            cand_data = self.drugs_by_id.get(cand_id, {})
            if not cand_data:
                continue
            
            # Calculate risk with this alternative
            test_ids = set(other_drug_ids) | {cand_id}
            alt_risk, alt_interactions, _ = self.calculate_risk_score(test_ids)
            
            # Get specific interaction with each other drug
            interactions_with_others = []
            for other_id in other_drug_ids:
                ddi = self.get_interaction(cand_id, other_id)
                if ddi:
                    interactions_with_others.append(ddi)
            
            if alt_risk < original_risk:
                alternatives.append({
                    'drug_id': cand_id,
                    'name': cand_data.get('name', ''),
                    'drugbank_id': cand_id,
                    'atc_codes': cand_data.get('atc_codes', ''),
                    'original_risk': original_risk,
                    'alternative_risk': alt_risk,
                    'risk_reduction': original_risk - alt_risk,
                    'num_interactions': len(interactions_with_others),
                    'interactions': interactions_with_others
                })
        
        # Sort by risk reduction (best first)
        alternatives.sort(key=lambda x: -x['risk_reduction'])
        return alternatives[:max_alternatives]
    
    # ============================================================
    # Polypharmacy Risk Index (PRI) Calculations
    # Based on publication_polypharmacy_alternatives/methods_brief_simple.tex
    # ============================================================
    
    def _compute_network_stats(self):
        """Compute network-wide statistics for PRI normalization (called once on load)"""
        if hasattr(self, '_network_stats'):
            return self._network_stats
        
        # Build adjacency info for each drug
        drug_interactions = {}  # drug_id -> list of (other_id, severity_weight)
        
        for (d1, d2), ddi in self.ddi_index.items():
            if d1 not in drug_interactions:
                drug_interactions[d1] = []
            sev = ddi.get('severity', '')
            weight = self.get_severity_weight(sev)
            drug_interactions[d1].append((d2, weight))
        
        # Compute max values for normalization
        max_degree = 0
        max_weighted = 0
        all_betweenness = []  # Simplified: degree-based proxy
        
        for drug_id, interactions in drug_interactions.items():
            degree = len(interactions)
            weighted = sum(w for _, w in interactions)
            max_degree = max(max_degree, degree)
            max_weighted = max(max_weighted, weighted)
            # Use degree as betweenness proxy (full betweenness is expensive)
            all_betweenness.append(degree)
        
        max_betweenness = max(all_betweenness) if all_betweenness else 1
        
        self._network_stats = {
            'drug_interactions': drug_interactions,
            'max_degree': max_degree or 1,
            'max_weighted': max_weighted or 1,
            'max_betweenness': max_betweenness or 1,
            'num_drugs': len(drug_interactions)
        }
        return self._network_stats
    
    def calculate_pri(self, drug_id):
        """
        Calculate Polypharmacy Risk Index (PRI) for a single drug.
        
        PRI(d) = 0.25 * C_degree + 0.30 * C_weighted + 0.20 * C_betweenness + 0.25 * S(d)
        
        Components:
        - C_degree: Normalized interaction count (how many drugs it interacts with)
        - C_weighted: Severity-weighted interaction sum (contraindicated=10, major=7, etc.)
        - C_betweenness: Network centrality (proxy: normalized degree)
        - S: Severity profile (proportion of severe interactions)
        
        Returns: dict with PRI score and component breakdown
        """
        stats = self._compute_network_stats()
        drug_ints = stats['drug_interactions'].get(drug_id, [])
        
        if not drug_ints:
            return {
                'pri': 0.0,
                'risk_level': 'Lower Risk',
                'degree': 0, 'c_degree': 0.0,
                'weighted_sum': 0, 'c_weighted': 0.0,
                'c_betweenness': 0.0,
                'severity_profile': 0.0,
                'num_interactions': 0,
                'num_severe': 0
            }
        
        # Component calculations
        degree = len(drug_ints)
        weighted_sum = sum(w for _, w in drug_ints)
        
        # Severity profile: proportion of severe (contraindicated + major) interactions
        severe_count = sum(1 for _, w in drug_ints if w >= 0.7)  # major (0.7) or contraindicated (1.0)
        severity_profile = severe_count / degree if degree > 0 else 0
        
        # Normalize components (0-1 scale)
        c_degree = degree / stats['max_degree']
        c_weighted = weighted_sum / stats['max_weighted']
        c_betweenness = degree / stats['max_betweenness']  # Proxy for betweenness
        
        # PRI formula from methods brief
        pri = (0.25 * c_degree + 
               0.30 * c_weighted + 
               0.20 * c_betweenness + 
               0.25 * severity_profile)
        
        # Risk classification from methods brief
        if pri > 0.5:
            risk_level = 'High Risk'
        elif pri >= 0.3:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'Lower Risk'
        
        return {
            'pri': round(pri, 3),
            'risk_level': risk_level,
            'degree': degree,
            'c_degree': round(c_degree, 3),
            'weighted_sum': round(weighted_sum, 2),
            'c_weighted': round(c_weighted, 3),
            'c_betweenness': round(c_betweenness, 3),
            'severity_profile': round(severity_profile, 3),
            'num_interactions': degree,
            'num_severe': severe_count
        }
    
    def calculate_regimen_pri(self, drug_ids):
        """
        Calculate aggregate PRI statistics for a medication regimen.
        
        Returns: dict with per-drug PRIs, average, and max risk drug
        """
        drug_pris = {}
        max_pri = 0
        max_pri_drug = None
        
        for drug_id in drug_ids:
            pri_data = self.calculate_pri(drug_id)
            drug_name = self.drugs_by_id.get(drug_id, {}).get('name', drug_id)
            drug_pris[drug_name] = pri_data
            
            if pri_data['pri'] > max_pri:
                max_pri = pri_data['pri']
                max_pri_drug = drug_name
        
        avg_pri = sum(p['pri'] for p in drug_pris.values()) / len(drug_pris) if drug_pris else 0
        
        return {
            'drug_pris': drug_pris,
            'average_pri': round(avg_pri, 3),
            'max_pri': round(max_pri, 3),
            'highest_risk_drug': max_pri_drug,
            'num_high_risk': sum(1 for p in drug_pris.values() if p['risk_level'] == 'High Risk'),
            'num_medium_risk': sum(1 for p in drug_pris.values() if p['risk_level'] == 'Medium Risk')
        }
    
    def calculate_ars(self, original_drug_id, alternative_drug_id, other_drug_ids):
        """
        Calculate Alternative Recommendation Score (ARS) for a drug substitution.
        
        ARS = 0.70 * (Severity Reduction / W_max) + 0.30 * (PRI_original - PRI_alternative)
        
        Where:
        - Severity Reduction: Sum of severity weight reductions with regimen drugs
        - W_max: 10 * |other_drugs| (maximum possible reduction)
        - PRI diff: Improvement in network-wide risk
        
        Returns: dict with ARS score and components
        """
        other_ids = list(other_drug_ids)
        
        # Calculate severity weights with original and alternative
        original_severity_sum = 0
        alt_severity_sum = 0
        
        for other_id in other_ids:
            # Original drug's interactions
            orig_ddi = self.get_interaction(original_drug_id, other_id)
            if orig_ddi:
                original_severity_sum += self.get_severity_weight(orig_ddi.get('severity', ''))
            
            # Alternative drug's interactions
            alt_ddi = self.get_interaction(alternative_drug_id, other_id)
            if alt_ddi:
                alt_severity_sum += self.get_severity_weight(alt_ddi.get('severity', ''))
        
        # Severity reduction
        severity_reduction = original_severity_sum - alt_severity_sum
        w_max = 1.0 * len(other_ids)  # Max possible reduction (contraindicated weight = 1.0 in our system)
        normalized_sev_red = severity_reduction / w_max if w_max > 0 else 0
        
        # PRI improvement
        orig_pri = self.calculate_pri(original_drug_id)['pri']
        alt_pri = self.calculate_pri(alternative_drug_id)['pri']
        pri_improvement = orig_pri - alt_pri
        
        # ARS formula (0.70 severity + 0.30 PRI)
        ars = 0.70 * max(0, normalized_sev_red) + 0.30 * max(0, pri_improvement)
        
        return {
            'ars': round(ars, 3),
            'severity_reduction': round(severity_reduction, 3),
            'normalized_sev_red': round(normalized_sev_red, 3),
            'original_severity': round(original_severity_sum, 3),
            'alternative_severity': round(alt_severity_sum, 3),
            'original_pri': round(orig_pri, 3),
            'alternative_pri': round(alt_pri, 3),
            'pri_improvement': round(pri_improvement, 3)
        }
    
    def find_alternatives_with_ars(self, drug_id, other_drug_ids, max_alternatives=5):
        """
        Find alternative drugs ranked by Alternative Recommendation Score (ARS).
        Enhanced version of find_alternatives with PRI/ARS metrics.
        """
        drug_data = self.drugs_by_id.get(drug_id, {})
        if not drug_data:
            return []
        
        atc = str(drug_data.get('atc_codes', ''))
        if not atc or atc == 'nan':
            return []
        
        # Get ATC prefixes for this drug
        prefixes = []
        for code in atc.split('|'):
            if len(code) >= 5:
                prefixes.append(code[:5])
            if len(code) >= 4:
                prefixes.append(code[:4])
        
        # Find candidate alternatives from same therapeutic class
        candidates = set()
        for prefix in prefixes:
            for cand_id in self.atc_index.get(prefix, []):
                if cand_id != drug_id and cand_id not in other_drug_ids:
                    candidates.add(cand_id)
        
        if not candidates:
            return []
        
        # Original PRI for comparison
        orig_pri_data = self.calculate_pri(drug_id)
        
        alternatives = []
        for cand_id in candidates:
            cand_data = self.drugs_by_id.get(cand_id, {})
            if not cand_data:
                continue
            
            # Calculate ARS for this substitution
            ars_data = self.calculate_ars(drug_id, cand_id, other_drug_ids)
            
            # Only include if there's improvement
            if ars_data['ars'] > 0 or ars_data['pri_improvement'] > 0:
                # Get specific interactions with regimen drugs
                interactions_with_others = []
                for other_id in other_drug_ids:
                    ddi = self.get_interaction(cand_id, other_id)
                    if ddi:
                        interactions_with_others.append(ddi)
                
                alt_pri_data = self.calculate_pri(cand_id)
                
                alternatives.append({
                    'drug_id': cand_id,
                    'name': cand_data.get('name', ''),
                    'drugbank_id': cand_id,
                    'atc_codes': cand_data.get('atc_codes', ''),
                    # ARS metrics
                    'ars': ars_data['ars'],
                    'severity_reduction': ars_data['severity_reduction'],
                    'normalized_sev_red': ars_data['normalized_sev_red'],
                    'pri_improvement': ars_data['pri_improvement'],
                    # PRI comparison
                    'original_pri': orig_pri_data['pri'],
                    'alternative_pri': alt_pri_data['pri'],
                    'alt_risk_level': alt_pri_data['risk_level'],
                    # Interaction details
                    'num_interactions': len(interactions_with_others),
                    'interactions': interactions_with_others,
                    # Legacy compatibility fields
                    'risk_reduction': ars_data['ars'],  # Map to old field for compatibility
                    'original_risk': orig_pri_data['pri'],
                    'alternative_risk': alt_pri_data['pri']
                })
        
        # Sort by ARS (highest first = best alternative)
        alternatives.sort(key=lambda x: -x['ars'])
        return alternatives[:max_alternatives]


# ============================================================
# LLM Client
# ============================================================

class LLMClient:
    """Ollama LLM client - uses Llama3"""
    
    MODELS = {
        "Llama3": "llama3:latest"
    }
    
    DEFAULT_MODEL = "Llama3"
    
    def generate(self, prompt, model_name=None):
        import urllib.request
        if model_name is None:
            model_name = self.DEFAULT_MODEL
        model = self.MODELS.get(model_name, "llama3:latest")
        try:
            data = json.dumps({
                "model": model, "prompt": prompt, "stream": False,
                "options": {"num_predict": 1500, "temperature": 0.3}
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate", data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                response = json.loads(resp.read().decode()).get('response', '')
                return response.strip()
        except Exception as e:
            return f"[LLM Error: {str(e)[:50]}]"


def generate_llm_summary(interactions, drug_names, risk_level, regimen_pri=None,
                        resolved_drugs=None, shared_se=None, shared_proteins=None):
    """
    Use LLM to generate a comprehensive, human-readable clinical summary
    based on ALL knowledge graph data: interactions, indications, side effects, proteins.
    """
    if not interactions:
        return f"No significant drug-drug interactions were found between {', '.join(drug_names)} in our knowledge graph database."
    
    # Build structured interaction data for LLM (with source)
    interaction_details = []
    for i, inter in enumerate(interactions, 1):
        d1 = inter.get('drug1_name', inter.get('drug1_id', 'Unknown')).title()
        d2 = inter.get('drug2_name', inter.get('drug2_id', 'Unknown')).title()
        sev = inter.get('severity', 'Unknown')
        desc = inter.get('description', 'No description available')
        source = inter.get('source', 'DrugBank')
        
        sev_level = 'contraindicated' if 'contraindicated' in sev.lower() else \
                    'major' if 'major' in sev.lower() else \
                    'moderate' if 'moderate' in sev.lower() else 'minor'
        
        interaction_details.append(f"{i}. {d1} + {d2}: {sev_level.upper()} - {desc} [Source: {source}]")
    
    # Build drug indication context from KG
    indication_context = ""
    if resolved_drugs:
        indications = []
        for drug in resolved_drugs:
            name = drug.get('name', 'Unknown').title()
            indication = str(drug.get('indication', '')).strip()
            if indication and indication != 'nan' and len(indication) > 5:
                # Truncate long indications
                short_ind = indication[:200] + '...' if len(indication) > 200 else indication
                indications.append(f"- {name}: {short_ind}")
        if indications:
            indication_context = f"\n\nDRUG INDICATIONS (from DrugBank KG):\n" + "\n".join(indications)
    
    # Build PRI context if available
    pri_context = ""
    if regimen_pri and regimen_pri.get('drug_pris'):
        high_risk_drugs = [name for name, data in regimen_pri['drug_pris'].items() 
                          if data.get('risk_level') == 'High Risk']
        if high_risk_drugs:
            pri_context = f"\nHigh-risk drugs in regimen (by network centrality): {', '.join(high_risk_drugs)}"
    
    # Build shared side effects context from KG
    se_context = ""
    if shared_se:
        se_list = [f"{se} (affects: {', '.join(drugs)})" for se, drugs in list(shared_se.items())[:8]]
        se_context = f"\n\nSHARED SIDE EFFECTS (from DrugBank KG - additive toxicity risk):\n" + "\n".join([f"- {s}" for s in se_list])
    
    # Build shared protein targets context from KG
    protein_context = ""
    if shared_proteins:
        prot_list = [f"{prot} ({data['gene']}, shared by: {', '.join(data['drugs'])})" 
                     for prot, data in list(shared_proteins.items())[:5]]
        protein_context = f"\n\nSHARED PROTEIN TARGETS (from DrugBank KG - mechanistic overlap):\n" + "\n".join([f"- {p}" for p in prot_list])
    
    # Create comprehensive prompt for LLM
    prompt = f"""Based on the following KNOWLEDGE GRAPH DATA from DrugBank (759,774 DDIs, 4,313 drugs), write a comprehensive clinical summary.

DRUG REGIMEN: {', '.join(drug_names)}
OVERALL RISK LEVEL: {risk_level}{pri_context}

INTERACTIONS FROM KNOWLEDGE GRAPH:
{chr(10).join(interaction_details)}{indication_context}{se_context}{protein_context}

Write a 4-6 sentence clinical summary in flowing prose that:
1. States the overall risk level and number of interactions found in the knowledge graph
2. Explains the clinical implications of each interaction (bleeding risk, hypoglycemia, etc.)
3. Notes any shared side effects that increase additive toxicity risk
4. Mentions relevant protein targets if they explain mechanistic drug-drug interactions
5. Briefly considers whether the drug indications justify the risk profile

Use professional medical language. Be comprehensive but concise. Base ALL statements on the KG data provided above - do NOT invent information."""

    try:
        summary = llm.generate(prompt)  # Uses Mistral 7B
        # Clean up the response
        summary = summary.strip()
        if not summary or summary.startswith("[LLM Error"):
            # Fallback to rule-based summary
            return _fallback_summary(interactions, resolved_drugs, shared_se, shared_proteins)
        return summary
    except Exception as e:
        return _fallback_summary(interactions, resolved_drugs, shared_se, shared_proteins)


def _fallback_summary(interactions, resolved_drugs=None, shared_se=None, shared_proteins=None):
    """Fallback rule-based summary if LLM fails - includes all KG data"""
    summaries = []
    
    # Interaction summaries
    for i in interactions:
        d1 = i.get('drug1_name', i.get('drug1_id', '?')).title()
        d2 = i.get('drug2_name', i.get('drug2_id', '?')).title()
        sev = i.get('severity', 'Unknown').lower()
        desc = str(i.get('description', ''))
        source = i.get('source', 'DrugBank')
        
        severity_text = 'contraindicated' if 'contraindicated' in sev else \
                       'major' if 'major' in sev else \
                       'moderate' if 'moderate' in sev else 'minor'
        
        effect = desc.split('.')[0] if '.' in desc else desc[:80]
        summaries.append(f"**{d1}** and **{d2}** have a {severity_text} interaction — {effect.lower()} (Source: {source})")
    
    result = ". ".join(summaries) + "."
    
    # Add shared side effects
    if shared_se:
        top_se = list(shared_se.keys())[:5]
        result += f" Notably, these drugs share common side effects including {', '.join(top_se)}, increasing additive toxicity risk."
    
    # Add shared proteins
    if shared_proteins:
        top_proteins = list(shared_proteins.keys())[:3]
        result += f" The drugs share target proteins ({', '.join(top_proteins)}), explaining mechanistic interaction pathways."
    
    result += " *Data sourced from DrugBank Knowledge Graph.*"
    return result


def generate_llm_monitoring(interactions, drug_names, shared_se=None, shared_proteins=None):
    """
    Use LLM to generate evidence-based monitoring recommendations with rationale
    based on knowledge graph interaction data. Includes source citations.
    """
    if not interactions:
        return None
    
    # Build interaction context for LLM
    interaction_details = []
    detected_risks = set()
    
    for inter in interactions:
        d1 = inter.get('drug1_name', inter.get('drug1_id', 'Unknown')).title()
        d2 = inter.get('drug2_name', inter.get('drug2_id', 'Unknown')).title()
        desc = str(inter.get('description', '')).lower()
        sev = inter.get('severity', 'Unknown')
        
        interaction_details.append(f"- {d1} + {d2} ({sev}): {inter.get('description', '')}")
        
        # Detect risk categories from description
        if 'bleeding' in desc or 'anticoagul' in desc or 'hemorrhag' in desc:
            detected_risks.add('bleeding')
        if 'serotonin' in desc:
            detected_risks.add('serotonin_syndrome')
        if 'qt' in desc or 'arrhythm' in desc or 'torsade' in desc:
            detected_risks.add('cardiac')
        if 'hypotension' in desc or 'blood pressure' in desc:
            detected_risks.add('hypotension')
        if 'renal' in desc or 'kidney' in desc or 'nephro' in desc:
            detected_risks.add('renal')
        if 'hepat' in desc or 'liver' in desc:
            detected_risks.add('hepatic')
        if 'hypoglycemi' in desc or 'blood sugar' in desc or 'glucose' in desc:
            detected_risks.add('hypoglycemia')
        if 'cns' in desc or 'sedation' in desc or 'drowsiness' in desc:
            detected_risks.add('cns_depression')
        if 'hyperkalemi' in desc or 'potassium' in desc:
            detected_risks.add('hyperkalemia')
    
    # Build shared effects context
    shared_context = ""
    if shared_se:
        top_se = list(shared_se.keys())[:5]
        shared_context += f"\nShared Side Effects (from SIDER database): {', '.join(top_se)}"
    if shared_proteins:
        protein_names = [f"{name} ({data['gene']})" for name, data in list(shared_proteins.items())[:3]]
        shared_context += f"\nShared Drug Targets (from UniProt): {', '.join(protein_names)}"
    
    # Create prompt for LLM
    prompt = f"""Based on the following DRUG-DRUG INTERACTION DATA from DrugBank Knowledge Graph, generate specific monitoring recommendations with clinical rationale.

DRUG REGIMEN: {', '.join(drug_names)}
DETECTED CLINICAL RISKS: {', '.join(detected_risks) if detected_risks else 'General monitoring advised'}

INTERACTIONS FROM DRUGBANK:
{chr(10).join(interaction_details)}
{shared_context}

Generate a monitoring plan with 3-5 specific recommendations. For EACH recommendation:
1. State what to monitor (lab test, vital sign, or symptom)
2. Explain WHY based on the specific interaction data above
3. Suggest monitoring frequency if applicable

Format each recommendation as:
**[Category]:** [What to monitor] - [Rationale based on the drug interactions]

End with: "Data sources: DrugBank, SIDER (side effects), UniProt (protein targets)"

Be specific to the drugs listed. Do NOT add generic recommendations not supported by the interaction data."""

    try:
        recommendations = llm.generate(prompt)  # Uses Mistral 7B
        recommendations = recommendations.strip()
        if not recommendations or recommendations.startswith("[LLM Error"):
            return _fallback_monitoring(interactions, detected_risks)
        return recommendations
    except Exception as e:
        return _fallback_monitoring(interactions, detected_risks)


def _fallback_monitoring(interactions, detected_risks):
    """Fallback rule-based monitoring if LLM fails"""
    monitoring = []
    
    if 'bleeding' in detected_risks:
        monitoring.append("**Bleeding Risk:** Monitor PT/INR, CBC, and signs of bleeding - *Rationale: Anticoagulant interaction detected in DrugBank*")
    if 'serotonin_syndrome' in detected_risks:
        monitoring.append("**Serotonin Syndrome:** Monitor mental status, temperature, reflexes - *Rationale: Serotonergic drug combination identified*")
    if 'cardiac' in detected_risks:
        monitoring.append("**Cardiac:** ECG monitoring for QT prolongation - *Rationale: QT-affecting interaction in DrugBank*")
    if 'hypotension' in detected_risks:
        monitoring.append("**Blood Pressure:** Regular BP monitoring - *Rationale: Hypotensive interaction detected*")
    if 'renal' in detected_risks:
        monitoring.append("**Renal Function:** Monitor creatinine, BUN, GFR - *Rationale: Nephrotoxic interaction in DrugBank*")
    if 'hepatic' in detected_risks:
        monitoring.append("**Hepatic Function:** Monitor LFTs (AST, ALT, bilirubin) - *Rationale: Hepatotoxic interaction detected*")
    if 'hypoglycemia' in detected_risks:
        monitoring.append("**Blood Glucose:** Monitor for hypoglycemia symptoms - *Rationale: Glucose-affecting interaction in DrugBank*")
    if 'cns_depression' in detected_risks:
        monitoring.append("**CNS Depression:** Monitor alertness, respiratory rate - *Rationale: CNS depressant combination detected*")
    if 'hyperkalemia' in detected_risks:
        monitoring.append("**Electrolytes:** Monitor potassium levels - *Rationale: Hyperkalemia risk from interaction*")
    
    if not monitoring:
        monitoring.append("**General:** Regular clinical assessment recommended")
    
    monitoring.append("\n*Data sources: DrugBank Knowledge Graph*")
    
    return "\n".join(monitoring)


def generate_llm_alternatives(alternatives_map, drug_names, interactions):
    """
    Use LLM to generate human-readable alternative drug recommendations
    based on ARS scores from the knowledge graph.
    All data is grounded in DrugBank knowledge graph with clinical evidence.
    """
    if not alternatives_map:
        return None
    
    # Clinical evidence database for common therapeutic alternatives
    clinical_evidence = {
        ('warfarin', 'dabigatran'): "RE-LY trial: 35% fewer intracranial hemorrhages vs warfarin (Connolly et al., NEJM 2009)",
        ('warfarin', 'apixaban'): "ARISTOTLE trial: 31% reduction in major bleeding (Granger et al., NEJM 2011)",
        ('warfarin', 'rivaroxaban'): "ROCKET-AF: established non-inferiority with predictable dosing (Patel et al., NEJM 2011)",
        ('propranolol', 'atenolol'): "ACC/AHA guidelines: cardioselective agents preferred in reactive airway disease/diabetes",
        ('propranolol', 'metoprolol'): "ACC/AHA guidelines: cardioselective beta-blockers have fewer respiratory effects",
        ('propranolol', 'bisoprolol'): "ESC Heart Failure guidelines: bisoprolol has cardioselective advantages",
        ('simvastatin', 'pravastatin'): "ACC guidelines: pravastatin recommended for patients on multiple medications (fewer CYP3A4 interactions)",
        ('simvastatin', 'atorvastatin'): "FDA 2011 warning: simvastatin dose limits due to CYP3A4 interactions",
        ('simvastatin', 'rosuvastatin'): "Clinical guidelines: rosuvastatin has minimal CYP450 metabolism",
        ('indomethacin', 'naproxen'): "AGS Beers Criteria: avoid indomethacin in older adults due to CNS toxicity",
        ('indomethacin', 'diclofenac'): "Henry et al., BMJ 1996: indomethacin has higher GI complication rates",
        ('indomethacin', 'ibuprofen'): "AGS Beers Criteria: ibuprofen preferred over indomethacin in elderly",
        ('aspirin', 'clopidogrel'): "CAPRIE trial: clopidogrel non-inferior with different bleeding profile",
        ('clopidogrel', 'ticagrelor'): "PLATO trial: ticagrelor showed mortality benefit in ACS",
        ('metformin', 'sitagliptin'): "ADA guidelines: DPP-4 inhibitors as alternatives in renal impairment",
        ('amlodipine', 'diltiazem'): "ACC/AHA guidelines: both effective for hypertension with different profiles",
    }
    
    # Build detailed knowledge graph data for each alternative
    kg_data_sections = []
    
    for original_drug, alts in alternatives_map.items():
        if not alts:
            continue
        
        orig_lower = original_drug.lower()
        
        # Get current interactions for this drug from KG
        original_interactions = []
        for inter in interactions:
            d1 = inter.get('drug1_name', inter.get('drug1_id', '')).lower()
            d2 = inter.get('drug2_name', inter.get('drug2_id', '')).lower()
            if orig_lower in [d1, d2]:
                sev = inter.get('severity', 'Unknown')
                desc = inter.get('description', '')[:150]
                other_drug = d1 if orig_lower == d2 else d2
                original_interactions.append({
                    'other_drug': other_drug.title(),
                    'severity': sev,
                    'mechanism': desc
                })
        
        # Determine therapeutic class from ATC
        first_alt = alts[0] if alts else {}
        atc_codes = first_alt.get('atc_codes', '')
        therapeutic_class = _get_therapeutic_class(atc_codes)
        
        # Build section for this drug's alternatives
        drug_section = f"\n**{therapeutic_class.upper()} - ALTERNATIVES FOR {original_drug.upper()}**\n"
        drug_section += f"DrugBank Knowledge Graph Analysis:\n"
        drug_section += f"\nCurrent Interaction Problems:\n"
        for oi in original_interactions[:3]:
            drug_section += f"  • {original_drug.title()} + {oi['other_drug']}: {oi['severity']}\n"
            if oi['mechanism']:
                drug_section += f"    Mechanism: {oi['mechanism']}...\n"
        
        drug_section += f"\nTherapeutic Alternatives (same ATC class, ranked by ARS):\n"
        drug_section += f"{'Substitution':<40} {'Sev.Red':<10} {'ΔPRI':<10} {'ARS':<10}\n"
        drug_section += "-" * 70 + "\n"
        
        for alt in alts[:3]:
            alt_name = alt.get('name', 'Unknown').title()
            ars = alt.get('ars', 0)
            sev_red = alt.get('normalized_sev_red', 0)
            pri_imp = alt.get('pri_improvement', 0)
            drugbank_id = alt.get('drugbank_id', '')
            atc = alt.get('atc_codes', '')[:15]
            orig_pri = alt.get('original_pri', 0)
            alt_pri = alt.get('alternative_pri', 0)
            alt_risk = alt.get('alt_risk_level', 'Unknown')
            num_int = alt.get('num_interactions', 0)
            
            # Get alternative's interactions from KG data
            alt_interactions = alt.get('interactions', [])
            
            drug_section += f"{original_drug.title()} → {alt_name:<25} {sev_red:.3f}      {pri_imp:.3f}     {ars:.3f}\n"
            drug_section += f"  [{drugbank_id}] ATC: {atc}\n"
            drug_section += f"  PRI: {orig_pri:.3f} → {alt_pri:.3f} | Risk: {alt_risk} | Regimen interactions: {num_int}\n"
            
            # Add clinical evidence if available
            evidence_key = (orig_lower, alt_name.lower())
            if evidence_key in clinical_evidence:
                drug_section += f"  📚 Clinical Evidence: {clinical_evidence[evidence_key]}\n"
            
            # Show specific interaction details from KG
            if alt_interactions:
                drug_section += f"  Known interactions with regimen:\n"
                for ai in alt_interactions[:2]:
                    ai_sev = ai.get('severity', 'Unknown')
                    ai_desc = ai.get('description', '')[:80]
                    drug_section += f"    - {ai_sev}: {ai_desc}...\n"
            else:
                drug_section += f"  ✓ No severe interactions with current regimen drugs\n"
            drug_section += "\n"
        
        kg_data_sections.append(drug_section)
    
    if not kg_data_sections:
        return None
    
    # Extract expected drug names for validation
    expected_drugs = list(alternatives_map.keys())
    
    # Build the LLM prompt with all KG data - natural language style
    prompt = f"""You are a clinical pharmacist writing a consultation note. Convert the knowledge graph data below into a natural language recommendation paragraph.

=== KNOWLEDGE GRAPH DATA ===
Drugs to replace: {', '.join(expected_drugs)}
{''.join(kg_data_sections)}
=== END DATA ===

Write a natural language clinical recommendation in paragraph form. For each high-risk drug, explain:
1. Why the current drug is problematic (cite the PRI score and interaction concerns)
2. The recommended alternative and why it's safer (mention ARS score, severity reduction %, PRI improvement)
3. Any supporting clinical evidence if provided

Write in flowing prose like a clinical consultation note, NOT bullet points or tables. Example style:
"Based on knowledge graph analysis, **Warfarin** (PRI: 0.73, High Risk) is contributing significantly to the regimen's interaction burden. We recommend considering **Dabigatran** as a safer alternative. This substitution achieves a 94% severity reduction and lowers the network risk from PRI 0.73 to 0.04, supported by the RE-LY trial showing 35% fewer intracranial hemorrhages. The alternative has no severe interactions with the other regimen drugs."

Use the EXACT drug names and numbers from the data. End with:
"*Recommendations derived from DrugBank Knowledge Graph (759,774 DDIs) with ARS scoring.*"
"""

    try:
        recommendation = llm.generate(prompt)  # Uses Mistral 7B
        recommendation = recommendation.strip()
        
        # Validate: ensure response mentions at least one expected drug
        if not recommendation or recommendation.startswith("[LLM Error"):
            return _fallback_alternatives(alternatives_map, interactions)
        
        # Check if LLM hallucinated (response doesn't mention any expected drugs)
        mentions_expected = any(drug.lower() in recommendation.lower() for drug in expected_drugs)
        if not mentions_expected:
            return _fallback_alternatives(alternatives_map, interactions)
            
        return recommendation
    except Exception as e:
        return _fallback_alternatives(alternatives_map, interactions)


def _get_therapeutic_class(atc_codes):
    """Map ATC codes to therapeutic class names"""
    if not atc_codes:
        return "Unknown Class"
    code = atc_codes.split('|')[0][:3] if '|' in atc_codes else atc_codes[:3]
    classes = {
        'B01': 'Antithrombotic Agents',
        'C01': 'Cardiac Therapy',
        'C02': 'Antihypertensives',
        'C03': 'Diuretics',
        'C07': 'Beta-Blockers',
        'C08': 'Calcium Channel Blockers',
        'C09': 'ACE Inhibitors/ARBs',
        'C10': 'Lipid Modifying Agents (Statins)',
        'A10': 'Antidiabetics',
        'M01': 'NSAIDs',
        'N02': 'Analgesics',
        'N03': 'Antiepileptics',
        'N05': 'Psychotropics',
        'N06': 'Psychoanaleptics',
        'J01': 'Antibiotics',
        'R03': 'Anti-Asthmatics',
    }
    return classes.get(code, f"ATC {code}")


def _fallback_alternatives(alternatives_map, interactions=None):
    """Fallback rule-based alternatives in natural language prose"""
    
    # Clinical evidence database
    clinical_evidence = {
        ('warfarin', 'dabigatran'): "the RE-LY trial demonstrated 35% fewer intracranial hemorrhages (NEJM 2009)",
        ('warfarin', 'apixaban'): "the ARISTOTLE trial showed 31% reduction in major bleeding (NEJM 2011)",
        ('warfarin', 'rivaroxaban'): "ROCKET-AF established non-inferiority for stroke prevention (NEJM 2011)",
        ('propranolol', 'atenolol'): "ACC/AHA guidelines recommend cardioselective agents",
        ('simvastatin', 'pravastatin'): "ACC guidelines favor agents with fewer CYP3A4 interactions",
        ('indomethacin', 'naproxen'): "the AGS Beers Criteria advises avoiding indomethacin in elderly patients",
    }
    
    paragraphs = []
    for original_drug, alts in alternatives_map.items():
        if not alts:
            continue
        
        top_alt = alts[0]
        alt_name = top_alt.get('name', 'Unknown').title()
        ars = top_alt.get('ars', 0)
        sev_red = top_alt.get('normalized_sev_red', 0)
        pri_imp = top_alt.get('pri_improvement', 0)
        drugbank_id = top_alt.get('drugbank_id', '')
        orig_pri = top_alt.get('original_pri', 0)
        alt_pri = top_alt.get('alternative_pri', 0)
        alt_risk = top_alt.get('alt_risk_level', 'Unknown')
        num_int = top_alt.get('num_interactions', 0)
        atc_codes = top_alt.get('atc_codes', '')
        
        therapeutic_class = _get_therapeutic_class(atc_codes)
        quality = "excellent" if ars > 0.4 else "strong" if ars > 0.2 else "reasonable"
        severity_pct = int(sev_red * 100)
        pri_pct = int(pri_imp * 100)
        
        # Build natural language paragraph
        para = f"Based on analysis of the DrugBank knowledge graph, **{original_drug.title()}** (PRI: {orig_pri:.2f}) is contributing to the regimen's interaction burden. "
        para += f"We recommend considering **{alt_name}** ({drugbank_id}) from the {therapeutic_class} class as a {quality} alternative. "
        para += f"This substitution achieves a {severity_pct}% severity reduction and a {pri_pct}% improvement in polypharmacy risk, "
        para += f"resulting in an Alternative Recommendation Score (ARS) of {ars:.3f}. "
        para += f"The alternative would lower the drug's risk profile from PRI {orig_pri:.3f} to {alt_pri:.3f} ({alt_risk} risk level)"
        
        if num_int > 0:
            para += f", with only {num_int} interactions in the current regimen"
        
        para += ". "
        
        # Add clinical evidence if available
        evidence_key = (original_drug.lower(), alt_name.lower())
        if evidence_key in clinical_evidence:
            para += f"Clinical evidence supporting this switch includes: {clinical_evidence[evidence_key]}."
        
        paragraphs.append(para)
    
    if paragraphs:
        result = "\n\n".join(paragraphs)
        result += "\n\n*All alternative recommendations are derived from the DrugBank Knowledge Graph (759,774 DDIs, 4,313 drugs) with DDInter-validated severity classification.*"
        return result
    return None


# ============================================================
# CONVERSATIONAL AI ASSISTANT - Natural ChatGPT-like Experience
# ============================================================

class ConversationMemory:
    """
    Stores conversation context and analysis reports for natural dialogue
    """
    def __init__(self):
        self.report = ""  # Full analysis report
        self.drugs = []
        self.risk_level = ""
        self.interactions = []
        self.alternatives = {}
        self.regimen_pri = {}  # PRI data for each drug
        self.conversation_history = []  # [(role, message), ...]
        self.max_history = 10  # Keep last N exchanges
        
    def update_from_analysis(self, report, drugs, risk, interactions, alternatives, regimen_pri=None):
        """Store analysis results in memory"""
        self.report = report
        self.drugs = drugs
        self.risk_level = risk
        self.interactions = interactions
        self.alternatives = alternatives
        self.regimen_pri = regimen_pri or {}
        
    def add_message(self, role, content):
        """Add message to conversation history"""
        self.conversation_history.append((role, content))
        # Trim to max history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def get_history_text(self):
        """Get formatted conversation history"""
        if not self.conversation_history:
            return ""
        
        history = []
        for role, content in self.conversation_history[-8:]:  # Last 4 exchanges
            if role == "user":
                history.append(f"User: {content}")
            else:
                history.append(f"Assistant: {content}")
        return "\n".join(history)
    
    def clear(self):
        """Clear memory for new session"""
        self.report = ""
        self.drugs = []
        self.risk_level = ""
        self.interactions = []
        self.alternatives = {}
        self.regimen_pri = {}
        self.conversation_history = []


class NaturalChatAssistant:
    """
    Knowledge Graph-First Conversational Assistant
    
    Prioritizes data from the knowledge graph and only uses LLM's 
    own knowledge when KG data is insufficient.
    """
    
    SYSTEM_PROMPT = """You are a clinical pharmacology assistant powered by a DrugBank Knowledge Graph with advanced risk metrics.

CRITICAL: Your answers must be based on the KNOWLEDGE GRAPH DATA provided below.
- ALWAYS cite specific data from the knowledge graph when available
- When referencing interactions, quote the exact description from the KG
- Mention DrugBank IDs when discussing specific drugs
- Use PRI (Polypharmacy Risk Index) scores to explain drug risk levels
- Use ARS (Alternative Recommendation Score) to justify alternative drug suggestions
- If data is NOT in your knowledge context, clearly state: "This information is not in my knowledge graph, but based on general pharmacology..."

RISK METRICS YOU HAVE ACCESS TO:
1. PRI (Polypharmacy Risk Index): Composite score (0-1) based on:
   - Degree centrality (number of interactions)
   - Weighted degree (severity-weighted interactions)
   - Betweenness centrality (network position)
   - Severity profile (proportion of severe interactions)
   Risk levels: High Risk (>0.5), Medium Risk (0.3-0.5), Lower Risk (<0.3)

2. ARS (Alternative Recommendation Score): Ranks safer alternatives based on:
   - 70% Severity Reduction with current regimen
   - 30% PRI Improvement (lower network-wide risk)
   Higher ARS = Better alternative

Your style:
- Warm and professional, like a knowledgeable pharmacist
- Educational without being condescending
- Use markdown formatting (bold, bullet points) for clarity
- When discussing risk, reference specific PRI scores and risk levels
- When suggesting alternatives, cite ARS scores and explain why they're safer
- Keep responses focused and evidence-based

For greetings/casual chat: respond briefly and naturally.
For drug questions: provide detailed KG-sourced information with risk metrics."""

    def __init__(self, knowledge_graph, llm_client):
        self.kg = knowledge_graph
        self.llm = llm_client
        self.memory = ConversationMemory()
        
    def update_memory(self, report, drugs, risk, interactions, alternatives, regimen_pri=None):
        """Update conversation memory with new analysis including PRI"""
        self.memory.update_from_analysis(report, drugs, risk, interactions, alternatives, regimen_pri)
    
    def extract_drug_from_query(self, message):
        """Extract drug names mentioned in the user's query"""
        msg_lower = message.lower()
        mentioned_drugs = []
        
        # Check analyzed drugs first
        for drug in self.memory.drugs:
            if drug.lower() in msg_lower:
                mentioned_drugs.append(drug)
        
        # Check knowledge graph for any other drug mentions
        for drug_name in self.kg.drugs.keys():
            if drug_name in msg_lower and drug_name not in [d.lower() for d in mentioned_drugs]:
                mentioned_drugs.append(drug_name)
                if len(mentioned_drugs) >= 4:  # Limit to avoid huge context
                    break
        
        return mentioned_drugs
    
    def get_drug_details_from_kg(self, drug_name):
        """Extract comprehensive drug information from knowledge graph"""
        drug_data = self.kg.drugs.get(drug_name.lower(), {})
        if not drug_data:
            return None
        
        details = {
            'name': drug_name.title(),
            'drugbank_id': drug_data.get('drugbank_id', 'Unknown'),
            'description': str(drug_data.get('description', ''))[:500] if str(drug_data.get('description', '')) != 'nan' else '',
            'mechanism': str(drug_data.get('mechanism_of_action', ''))[:600] if str(drug_data.get('mechanism_of_action', '')) != 'nan' else '',
            'pharmacodynamics': str(drug_data.get('pharmacodynamics', ''))[:400] if str(drug_data.get('pharmacodynamics', '')) != 'nan' else '',
            'indication': str(drug_data.get('indication', ''))[:300] if str(drug_data.get('indication', '')) != 'nan' else '',
            'atc_codes': str(drug_data.get('atc_codes', '')) if str(drug_data.get('atc_codes', '')) != 'nan' else '',
            'half_life': str(drug_data.get('half_life', '')) if str(drug_data.get('half_life', '')) != 'nan' else '',
            'route': str(drug_data.get('route', '')) if str(drug_data.get('route', '')) != 'nan' else '',
        }
        
        # Get protein targets
        db_id = drug_data.get('drugbank_id', '')
        proteins = self.kg.proteins.get(db_id, [])[:5]
        if proteins:
            details['proteins'] = [{'name': p.get('name', ''), 'uniprot': p.get('uniprot_id', '')} for p in proteins if p.get('name')]
        
        # Get side effects
        side_effects = self.kg.side_effects.get(db_id, [])[:10]
        if side_effects:
            details['side_effects'] = [se.get('name', '') for se in side_effects if se.get('name')]
        
        return details
    
    def get_interaction_details(self, drug1, drug2):
        """Get specific interaction details from KG"""
        for inter in self.memory.interactions:
            d1 = inter.get('drug1_name', inter.get('drug1_id', '')).lower()
            d2 = inter.get('drug2_name', inter.get('drug2_id', '')).lower()
            if (drug1.lower() in [d1, d2]) and (drug2.lower() in [d1, d2]):
                return inter
        return None
        
    def build_knowledge_context(self, user_message=""):
        """Build comprehensive knowledge context from KG data"""
        context_parts = []
        
        # Extract drugs mentioned in query
        query_drugs = self.extract_drug_from_query(user_message)
        all_drugs = list(set(self.memory.drugs + query_drugs))
        
        if not all_drugs and not self.memory.drugs:
            context_parts.append("=== NO ANALYSIS YET ===")
            context_parts.append("User hasn't analyzed any drugs yet. Guide them to use the left panel.")
            return "\n".join(context_parts)
        
        # Add analysis summary
        if self.memory.report:
            context_parts.append("=== ANALYSIS SUMMARY ===")
            context_parts.append(f"Analyzed drugs: {', '.join(self.memory.drugs)}")
            context_parts.append(f"Risk Level: {self.memory.risk_level}")
            context_parts.append(f"Total interactions found: {len(self.memory.interactions)}")
        
        # Add detailed drug information from KG
        context_parts.append("\n=== KNOWLEDGE GRAPH: DRUG DATA ===")
        for drug_name in all_drugs[:5]:
            details = self.get_drug_details_from_kg(drug_name)
            if details:
                context_parts.append(f"\n### {details['name']} ({details['drugbank_id']})")
                
                if details.get('description'):
                    context_parts.append(f"**Description:** {details['description']}")
                
                if details.get('mechanism'):
                    context_parts.append(f"**Mechanism of Action:** {details['mechanism']}")
                
                if details.get('pharmacodynamics'):
                    context_parts.append(f"**Pharmacodynamics:** {details['pharmacodynamics']}")
                
                if details.get('indication'):
                    context_parts.append(f"**Indication:** {details['indication']}")
                
                if details.get('proteins'):
                    prot_str = ', '.join([f"{p['name']} ({p['uniprot']})" for p in details['proteins'][:3]])
                    context_parts.append(f"**Protein Targets:** {prot_str}")
                
                if details.get('side_effects'):
                    se_str = ', '.join(details['side_effects'][:8])
                    context_parts.append(f"**Known Side Effects:** {se_str}")
                
                if details.get('half_life'):
                    context_parts.append(f"**Half-life:** {details['half_life']}")
        
        # Add interaction details from KG
        if self.memory.interactions:
            context_parts.append("\n=== KNOWLEDGE GRAPH: INTERACTIONS ===")
            for inter in self.memory.interactions[:8]:
                d1 = inter.get('drug1_name', inter.get('drug1_id', ''))
                d2 = inter.get('drug2_name', inter.get('drug2_id', ''))
                sev = inter.get('severity', 'Unknown')
                desc = str(inter.get('description', ''))[:400]
                context_parts.append(f"\n**{d1} + {d2}** (Severity: {sev})")
                context_parts.append(f"Description: {desc}")
        
        # Add alternatives from KG with ARS scores
        if self.memory.alternatives:
            context_parts.append("\n=== KNOWLEDGE GRAPH: ALTERNATIVES WITH ARS ===")
            for drug, alts in list(self.memory.alternatives.items())[:4]:
                if alts:
                    context_parts.append(f"\n**Alternatives to {drug.title()}:**")
                    for alt in alts[:4]:
                        alt_name = alt['name']
                        ars = alt.get('ars', 0)
                        sev_red = alt.get('normalized_sev_red', 0) * 100
                        pri_imp = alt.get('pri_improvement', 0) * 100
                        context_parts.append(f"- {alt_name}: ARS={ars:.3f} (Severity reduction: {sev_red:.0f}%, PRI improvement: {pri_imp:.1f}%)")
        
        # Add PRI risk assessment data
        if self.memory.regimen_pri and self.memory.regimen_pri.get('drug_pris'):
            context_parts.append("\n=== POLYPHARMACY RISK INDEX (PRI) ===")
            context_parts.append("PRI Formula: 0.25×Degree + 0.30×WeightedDegree + 0.20×Betweenness + 0.25×SeverityProfile")
            context_parts.append("Risk Levels: High Risk (>0.5), Medium Risk (0.3-0.5), Lower Risk (<0.3)")
            context_parts.append(f"Average Regimen PRI: {self.memory.regimen_pri.get('average_pri', 0):.3f}")
            context_parts.append(f"Highest Risk Drug: {self.memory.regimen_pri.get('highest_risk_drug', 'N/A')}")
            
            for drug_name, pri_data in self.memory.regimen_pri['drug_pris'].items():
                context_parts.append(f"\n**{drug_name.title()}:**")
                context_parts.append(f"  PRI Score: {pri_data['pri']:.3f} ({pri_data['risk_level']})")
                context_parts.append(f"  Total Interactions: {pri_data['num_interactions']}")
                context_parts.append(f"  Severe Interactions: {pri_data['num_severe']}")
        
        return "\n".join(context_parts)
    
    def respond(self, user_message, model_name="Llama3"):
        """
        Generate a KG-informed response to the user's message
        """
        # Add user message to history
        self.memory.add_message("user", user_message)
        
        # Build comprehensive knowledge context
        knowledge = self.build_knowledge_context(user_message)
        history = self.memory.get_history_text()
        
        prompt = f"""{self.SYSTEM_PROMPT}

=== KNOWLEDGE GRAPH DATA ===
{knowledge}

=== CONVERSATION HISTORY ===
{history}

=== USER MESSAGE ===
{user_message}

Provide a helpful response. Base your answer on the KNOWLEDGE GRAPH DATA above. 
If the information isn't in the knowledge graph, state that clearly before adding general knowledge:"""

        # Generate response
        response = self.llm.generate(prompt, model_name)
        
        # Store assistant response in history
        self.memory.add_message("assistant", response)
        
        return response


# ============================================================
# Global instances & state
# ============================================================

kg = KnowledgeGraph()
llm = LLMClient()
chat_assistant = None  # Natural conversation assistant
current_analysis = {"drugs": [], "interactions": [], "risk": "", "report": "", "alternatives": {}, "regimen_pri": {}}
identified_drugs = {"confirmed": [], "suggestions": {}, "not_found": []}

def get_chat_assistant():
    """Get or initialize natural chat assistant"""
    global chat_assistant
    if chat_assistant is None:
        if not kg.loaded:
            kg.load()
        chat_assistant = NaturalChatAssistant(kg, llm)
    return chat_assistant


# ============================================================
# Drug Identification with Fuzzy Matching
# ============================================================

def identify_drugs_preview(drug_input, progress=gr.Progress()):
    """
    Step 1: Parse input and identify drugs with fuzzy matching.
    Shows compact preview with tables for suggestions.
    """
    if not drug_input or not drug_input.strip():
        return "Please enter drug names", "", gr.update(visible=False)
    
    # Load KG if needed
    progress(0.2, desc="Loading Knowledge Graph...")
    if not kg.loaded:
        kg.load()
    
    progress(0.5, desc="Identifying drugs...")
    
    found_drugs, suggestions, not_found = kg.identify_drugs(drug_input)
    
    # Store for later use
    identified_drugs["confirmed"] = found_drugs
    identified_drugs["suggestions"] = suggestions
    identified_drugs["not_found"] = not_found
    
    total_found = len(found_drugs)
    total_issues = len(suggestions) + len(not_found)
    
    # === Build compact preview ===
    preview = "## Drug Identification\n\n"
    
    # Summary paragraph
    if found_drugs:
        matched_names = [f"**{d[1]}**" for d in found_drugs]
        preview += f"Successfully identified **{total_found}** drug(s): {', '.join(matched_names)}.\n\n"
    
    # Confirmed drugs table (if any name mappings occurred)
    name_changes = [(i, m, d) for i, m, d in found_drugs if i.lower() != m.lower()]
    if name_changes:
        preview += "| Your Input | Matched As | DrugBank |\n"
        preview += "|------------|------------|----------|\n"
        for inp, matched, data in name_changes:
            db_id = data.get('drugbank_id', '')
            preview += f"| {inp} | {matched} | [{db_id}](https://go.drugbank.com/drugs/{db_id}) |\n"
        preview += "\n"
    
    # Suggestions for misspelled names
    if suggestions:
        preview += "### Possible Matches (please verify)\n\n"
        preview += "The following terms were not found exactly. Did you mean:\n\n"
        preview += "| You Entered | Suggestion | Match % |\n"
        preview += "|-------------|------------|--------|\n"
        for input_name, matches in suggestions.items():
            top = matches[0]
            preview += f"| `{input_name}` | **{top[0]}** | {int(top[1]*100)}% |\n"
        preview += "\n*Edit your input with the correct spelling and click 'Identify' again.*\n\n"
    
    # Not found
    if not_found:
        preview += f"### Not Found: `{', '.join(not_found)}`\n\n"
        preview += "Try using generic drug names instead of brand names, or check spelling.\n\n"
    
    # Next step guidance
    preview += "---\n\n"
    if total_found >= 2:
        preview += f"**Ready!** {total_found} drugs confirmed. Click **'Generate Comprehensive Report'** to check interactions"
        if total_issues > 0:
            preview += f" (or fix the {total_issues} unmatched term(s) above first)"
        preview += ".\n"
    elif total_found == 1:
        preview += "Only 1 drug confirmed — enter at least 2 drugs to analyze interactions.\n"
    else:
        preview += "No drugs identified. Check spelling or try generic names (e.g., 'acetaminophen' instead of 'Tylenol').\n"
    
    # Build confirmed list for editing
    confirmed_list = ", ".join([d[1] for d in found_drugs])
    
    progress(1.0, desc="Done")
    
    # Show edit group if we found any drugs
    show_edit = len(found_drugs) > 0
    return preview, confirmed_list, gr.update(visible=show_edit)


# ============================================================
# Narrative & Image Drug Extraction
# ============================================================

def extract_drugs_from_narrative(narrative_text, progress=gr.Progress()):
    """
    Extract drug names from a natural language narrative.
    Example: "I take cardace 5mg in the morning and aspirin at night for my heart"
    Returns: comma-separated drug list
    """
    if not narrative_text or not narrative_text.strip():
        return "", "*Enter a description of your medications*"
    
    progress(0.2, desc="Loading drug database...")
    if not kg.loaded:
        kg.load()
    
    progress(0.4, desc="Analyzing text...")
    
    # Normalize text
    text = narrative_text.lower()
    
    # Common patterns to remove (dosages, frequencies, etc.)
    text = re.sub(r'\d+\s*(mg|ml|mcg|g|iu|units?)\b', ' ', text)  # Remove dosages
    text = re.sub(r'\b(once|twice|thrice|\d+\s*times?)\s*(daily|a\s*day|per\s*day|weekly)?\b', ' ', text)
    text = re.sub(r'\b(morning|evening|night|afternoon|bedtime|before|after|with)\s*(meals?|food|breakfast|lunch|dinner)?\b', ' ', text)
    text = re.sub(r'\b(tablet|capsule|pill|dose|dosage)s?\b', ' ', text)
    
    # Extract potential drug words (2+ char words)
    words = re.findall(r'\b[a-z]{2,}\b', text)
    
    # Also try 2-word combinations (for drugs like "folic acid")
    words_list = text.split()
    bigrams = [' '.join(words_list[i:i+2]) for i in range(len(words_list)-1)]
    
    # Try to match each word/bigram against database
    found_drugs = []
    matched_terms = set()
    
    # Check bigrams first (longer matches preferred)
    for phrase in bigrams:
        phrase = phrase.strip()
        if len(phrase) < 3 or phrase in matched_terms:
            continue
        
        # Check exact match first
        if phrase in kg.aliases:
            found_drugs.append(kg.aliases[phrase])
            matched_terms.add(phrase)
            continue
            
        # Check if it's a known drug name
        for drug_name in kg.drug_names:
            if phrase == drug_name.lower():
                found_drugs.append(drug_name)
                matched_terms.add(phrase)
                break
        else:
            # Fuzzy match if close enough
            for drug_name in kg.drug_names[:500]:  # Check top drugs
                if SequenceMatcher(None, phrase, drug_name.lower()).ratio() > 0.85:
                    found_drugs.append(drug_name)
                    matched_terms.add(phrase)
                    break
    
    # Then check single words
    for word in words:
        if word in matched_terms or len(word) < 3:
            continue
        
        # Check aliases
        if word in kg.aliases:
            found_drugs.append(kg.aliases[word])
            matched_terms.add(word)
            continue
        
        # Exact match check
        for drug_name in kg.drug_names:
            if word == drug_name.lower():
                found_drugs.append(drug_name)
                matched_terms.add(word)
                break
        else:
            # Higher threshold for single words to avoid false positives
            for drug_name in kg.drug_names[:500]:
                if len(word) >= 4 and SequenceMatcher(None, word, drug_name.lower()).ratio() > 0.88:
                    found_drugs.append(drug_name)
                    matched_terms.add(word)
                    break
    
    progress(1.0, desc="Done")
    
    # Remove duplicates preserving order
    seen = set()
    unique_drugs = []
    for d in found_drugs:
        d_lower = d.lower()
        if d_lower not in seen:
            seen.add(d_lower)
            unique_drugs.append(d)
    
    if unique_drugs:
        drug_list = ", ".join(unique_drugs)
        status = f"**Found {len(unique_drugs)} drug(s):** {drug_list}\n\n*Review and click 'Check Drugs' to validate*"
        return drug_list, status
    else:
        return "", "*No drugs detected in the text. Try including drug names like 'aspirin', 'metoprolol', 'cardace', etc.*"


def extract_drugs_from_image(image, progress=gr.Progress()):
    """
    Extract drug names from an uploaded image using OCR.
    Works with photos of prescription labels, medication bottles, etc.
    """
    if image is None:
        return "", "*Upload an image of your prescription or medication*"
    
    if not HAS_OCR:
        return "", "**OCR not available.** Install pytesseract: `pip install pytesseract pillow` and ensure tesseract is installed on your system."
    
    progress(0.2, desc="Processing image...")
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to grayscale for better OCR
        image = image.convert('L')
        
        progress(0.5, desc="Extracting text with OCR...")
        
        # Run OCR
        extracted_text = pytesseract.image_to_string(image)
        
        progress(0.7, desc="Finding drug names...")
        
        if not extracted_text.strip():
            return "", "*Could not extract text from image. Try a clearer photo.*"
        
        # Use the narrative extractor on OCR text
        drug_list, status = extract_drugs_from_narrative(extracted_text, progress=progress)
        
        if drug_list:
            status = f"**OCR Text:** _{extracted_text[:200]}{'...' if len(extracted_text) > 200 else ''}_\n\n" + status
        else:
            status = f"**OCR Text:** _{extracted_text[:200]}{'...' if len(extracted_text) > 200 else ''}_\n\n*No drugs detected. The image may be unclear or not contain drug names.*"
        
        return drug_list, status
        
    except Exception as e:
        return "", f"**Error processing image:** {str(e)}"


# ============================================================
# Knowledge Graph Analysis with Alternatives
# ============================================================

def analyze_ddi(drug_input, progress=gr.Progress()):
    """Analyze drug interactions and find safer alternatives"""
    
    if not drug_input or not drug_input.strip():
        return (
            "Please enter drug names separated by commas, + signs, or newlines\n\n**Examples:**\n- `warfarin, aspirin, ibuprofen`\n- `Warfarin + Aspirin`\n- `sildenafil + nitroglycerin`",
            gr.update(visible=False), gr.update(), gr.update(), ""
        )
    
    # Load KG if needed
    progress(0.1, desc="Loading Knowledge Graph...")
    if not kg.loaded:
        kg.load()
    
    # Parse drugs with smart separator handling
    progress(0.2, desc="Identifying drugs...")
    found_drugs, suggestions, not_found = kg.identify_drugs(drug_input)
    
    if len(found_drugs) < 2:
        # Not enough drugs - show preview instead
        preview = "**Need at least 2 identified drugs for interaction analysis.**\n\n"
        preview += "---\n\n"
        if found_drugs:
            preview += f"Found: {', '.join([d[1] for d in found_drugs])}\n\n"
        if suggestions:
            preview += "**Did you mean:**\n"
            for input_name, matches in suggestions.items():
                top_match = matches[0]
                preview += f"- `{input_name}` → **{top_match[0]}** ({int(top_match[1]*100)}% match)\n"
            preview += "\n*Edit your input with the correct spelling and try again.*\n\n"
        if not_found:
            preview += f"Not found: {', '.join(not_found)}\n"
        return (preview, gr.update(visible=False), gr.update(), gr.update(), "")
    
    # Resolve confirmed drugs
    progress(0.3, desc="Resolving drug data...")
    resolved = [d[2] for d in found_drugs]  # drug_data is third element
    not_found_final = [s for s in suggestions.keys()] + not_found
    
    if not resolved:
        input_drugs = kg.parse_drug_input(drug_input)
        return (f"No drugs found in database: {', '.join(input_drugs)}", gr.update(visible=False), gr.update(), gr.update(), "")
    
    # Get drug IDs
    drug_ids = [d.get('drugbank_id', '') for d in resolved if d.get('drugbank_id')]
    
    # Calculate risk score
    progress(0.4, desc="Calculating polypharmacy risk...")
    risk_score, interactions, counts = kg.calculate_risk_score(drug_ids)
    
    # Populate drug names in interactions for better reporting
    for interaction in interactions:
        d1_id = interaction.get('drug1_id', '')
        d2_id = interaction.get('drug2_id', '')
        d1_data = kg.drugs_by_id.get(d1_id, {})
        d2_data = kg.drugs_by_id.get(d2_id, {})
        interaction['drug1_name'] = d1_data.get('name', d1_id)
        interaction['drug2_name'] = d2_data.get('name', d2_id)
    
    # Calculate Polypharmacy Risk Index (PRI) for each drug
    progress(0.5, desc="Computing PRI metrics...")
    regimen_pri = kg.calculate_regimen_pri(drug_ids)
    
    risk_level = "CRITICAL" if risk_score >= 0.8 else "HIGH" if risk_score >= 0.5 else "MODERATE" if risk_score >= 0.2 else "LOW"
    
    # Find alternatives with ARS scoring for high-risk drugs
    progress(0.6, desc="Finding safer alternatives with ARS...")
    alternatives_map = {}
    
    if interactions:
        # Find which drugs should have alternatives suggested:
        # 1. Drugs involved in contraindicated/major interactions
        # 2. High-PRI drugs (PRI > 0.5) - per methods brief recommendations
        candidate_drugs = set()
        
        # Add drugs in severe interactions
        for i in interactions:
            sev = i.get('severity', '').lower()
            if 'contraindicated' in sev or 'major' in sev:
                candidate_drugs.add(i.get('drug1_id', ''))
                candidate_drugs.add(i.get('drug2_id', ''))
        
        # Also add high-PRI drugs (>0.5 = High Risk threshold)
        if regimen_pri and 'drug_pris' in regimen_pri:
            for drug_name, pri_data in regimen_pri.get('drug_pris', {}).items():
                if pri_data.get('pri', 0) > 0.5:
                    # Find the drug_id for this drug_name
                    for did in drug_ids:
                        ddata = kg.drugs_by_id.get(did, {})
                        if ddata.get('name', '').lower() == drug_name.lower():
                            candidate_drugs.add(did)
                            break
        
        # Find alternatives with ARS for each candidate drug
        for drug_id in candidate_drugs:
            if drug_id in drug_ids:
                other_ids = [d for d in drug_ids if d != drug_id]
                alternatives = kg.find_alternatives_with_ars(drug_id, other_ids)
                if alternatives:
                    drug_name = kg.drugs_by_id.get(drug_id, {}).get('name', drug_id)
                    alternatives_map[drug_name] = alternatives
    
    # Get shared side effects
    progress(0.75, desc="Analyzing shared effects...")
    all_se = {}
    all_proteins = {}
    for drug in resolved:
        did = drug.get('drugbank_id', '')
        for se in kg.side_effects.get(did, []):
            n = se['name']
            if n not in all_se: all_se[n] = []
            all_se[n].append(drug.get('name', ''))
        for p in kg.proteins.get(did, []):
            n = p['name']
            if n and n not in all_proteins: all_proteins[n] = {'drugs': [], 'gene': p['gene']}
            if n: all_proteins[n]['drugs'].append(drug.get('name', ''))
    
    shared_se = {k: v for k, v in all_se.items() if len(v) > 1}
    shared_proteins = {k: v for k, v in all_proteins.items() if len(v['drugs']) > 1}
    
    # Store for chat context (including PRI data)
    current_analysis["drugs"] = [d.get('name', '') for d in resolved]
    current_analysis["interactions"] = interactions
    current_analysis["risk"] = risk_level
    current_analysis["alternatives"] = alternatives_map
    current_analysis["regimen_pri"] = regimen_pri
    
    # Build comprehensive report with PRI/ARS metrics
    progress(0.9, desc="Generating report...")
    report = build_report(resolved, not_found_final, interactions, risk_level, risk_score, counts, 
                         shared_se, shared_proteins, alternatives_map, regimen_pri)
    current_analysis["report"] = report
    
    # Update chat assistant's memory with the new analysis including PRI
    assistant = get_chat_assistant()
    assistant.update_memory(
        report=report,
        drugs=current_analysis["drugs"],
        risk=risk_level,
        interactions=interactions,
        alternatives=alternatives_map,
        regimen_pri=regimen_pri
    )
    
    # Prepare checkbox choices for drug selection panel
    current_drug_names = [d.get('name', '').title() for d in resolved]
    
    # Collect all alternative names (flatten)
    alt_choices = []
    for original_drug, alts in alternatives_map.items():
        for alt in alts[:3]:  # Top 3 alternatives per drug
            alt_name = alt['name'].title()
            if alt_name not in alt_choices and alt_name.lower() not in [d.lower() for d in current_drug_names]:
                alt_choices.append(f"{alt_name} (replaces {original_drug.title()})")
    
    progress(1.0, desc="Complete")
    
    # Return: report, selection panel visibility, current drugs checkbox, alternatives checkbox
    return (
        report,
        gr.update(visible=True),  # selection_panel
        gr.update(choices=current_drug_names, value=current_drug_names),  # current_drugs_check (all selected)
        gr.update(choices=alt_choices, value=[]),  # alternatives_check (none selected)
        ""  # selection_status
    )


# ============================================================
# Re-analyze with Selected Drugs
# ============================================================

def reanalyze_with_selection(current_drugs_selected, alternatives_selected, progress=gr.Progress()):
    """Re-run analysis with user's selected drugs (original + chosen alternatives)"""
    
    # Extract drug names from selections
    selected_drugs = list(current_drugs_selected) if current_drugs_selected else []
    
    # Extract alternative drug names (format: "DrugName (replaces Original)")
    for alt in (alternatives_selected or []):
        # Parse "DrugName (replaces Original)"
        if " (replaces " in alt:
            drug_name = alt.split(" (replaces ")[0].strip()
        else:
            drug_name = alt.strip()
        if drug_name and drug_name not in selected_drugs:
            selected_drugs.append(drug_name)
    
    if len(selected_drugs) < 2:
        return (
            f"Select at least 2 drugs to analyze. Currently selected: {', '.join(selected_drugs) if selected_drugs else 'none'}",
            gr.update(), gr.update(), gr.update(),
            f"Need at least 2 drugs (have {len(selected_drugs)})"
        )
    
    # Run analysis with the selected drug combination
    drug_input = ", ".join(selected_drugs)
    
    status_msg = f"Re-analyzing with: **{', '.join(selected_drugs)}**"
    
    # Call analyze_ddi with new combination
    result = analyze_ddi(drug_input, progress)
    
    # Update status with what was analyzed
    new_status = f"Analyzed **{len(selected_drugs)}** drugs: {', '.join(selected_drugs)}"
    
    # Result is a tuple (report, panel_visible, current_drugs, alternatives, status)
    # Return with updated status
    return (result[0], result[1], result[2], result[3], new_status)


def build_report(drugs, not_found, interactions, risk_level, risk_score, counts, 
                 shared_se, shared_proteins, alternatives_map, regimen_pri=None):
    """Build clean, modern DDI report with PRI/ARS metrics"""
    
    drug_names = [d.get('name', 'Unknown').title() for d in drugs]
    
    # iOS-style risk colors with gradients
    risk_styles = {
        'CRITICAL': ('linear-gradient(135deg, #FF3B30 0%, #FF2D55 100%)', '#FFF5F5', '#FF3B30', 'Avoid this combination'),
        'HIGH': ('linear-gradient(135deg, #FF9500 0%, #FF3B30 100%)', '#FFF8F0', '#FF9500', 'Use with extreme caution'),
        'MODERATE': ('linear-gradient(135deg, #FFCC00 0%, #FF9500 100%)', '#FFFBF0', '#FF9500', 'Monitor closely'),
        'LOW': ('linear-gradient(135deg, #34C759 0%, #30B0C7 100%)', '#F0FFF5', '#34C759', 'Generally safe')
    }
    gradient, bg_color, text_color, advice = risk_styles.get(risk_level, ('linear-gradient(135deg, #8E8E93 0%, #636366 100%)', '#F5F5F7', '#8E8E93', 'Unknown'))
    
    # === iOS-STYLE RISK CARD ===
    pri_display = ""
    if regimen_pri:
        avg_pri = regimen_pri.get('average_pri', 0)
        pri_gradient = 'linear-gradient(135deg, #FF3B30 0%, #FF2D55 100%)' if avg_pri > 0.5 else 'linear-gradient(135deg, #FF9500 0%, #FFCC00 100%)' if avg_pri >= 0.3 else 'linear-gradient(135deg, #34C759 0%, #30B0C7 100%)'
        pri_display = f"""<div style="display:flex; flex-direction:column; align-items:center; gap:4px; padding:12px 16px; background:rgba(0,0,0,0.03); border-radius:12px;">
<span style="font-size:11px; font-weight:600; color:#8E8E93; text-transform:uppercase; letter-spacing:0.05em;">PRI Score</span>
<span style="font-size:24px; font-weight:700; background:{pri_gradient}; -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{avg_pri:.2f}</span>
</div>"""
    
    report = f"""<div style="background:{bg_color}; border-radius:20px; padding:24px; margin-bottom:24px; box-shadow:0 2px 12px rgba(0,0,0,0.06);">
<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
<div>
<div style="display:inline-flex; align-items:center; gap:8px; padding:8px 18px; background:{gradient}; border-radius:24px; margin-bottom:12px; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
<span style="font-size:14px; font-weight:700; letter-spacing:0.03em; color:white; text-transform:uppercase;">{risk_level} RISK</span>
</div>
<p style="margin:0; color:#000; font-size:17px; font-weight:600;">{', '.join(drug_names)}</p>
<p style="margin:6px 0 0 0; color:#8E8E93; font-size:14px;">{advice}</p>
</div>
{pri_display}
</div>
</div>
"""
    
    if not_found:
        report += f"\n<p style='color:#94a3b8; font-size:0.85rem;'>Not found: {', '.join(not_found)}</p>"
    
    # === INTERACTIONS TABLE ===
    if interactions:
        severity_order = {'contraindicated': 0, 'major': 1, 'moderate': 2, 'minor': 3}
        sorted_interactions = sorted(interactions, 
            key=lambda x: severity_order.get(x.get('severity', '').lower().split()[0], 4))
        
        report += f"\n\n### Interactions ({len(interactions)})\n"
        report += "| Drug Pair | Severity | Effect |\n"
        report += "|-----------|----------|--------|\n"
        
        for i in sorted_interactions:
            d1 = i.get('drug1_name', i.get('drug1_id', '?')).title()
            d2 = i.get('drug2_name', i.get('drug2_id', '?')).title()
            sev = i.get('severity', 'Unknown')
            sev_lower = sev.lower()
            
            # Clean severity labels
            if 'contraindicated' in sev_lower:
                sev_badge = "**Contraindicated**"
            elif 'major' in sev_lower:
                sev_badge = "**Major**"
            elif 'moderate' in sev_lower:
                sev_badge = "Moderate"
            else:
                sev_badge = "Minor"
            
            desc = str(i.get('description', 'No description'))
            if len(desc) > 100:
                desc = desc[:97] + "..."
            
            report += f"| {d1} + {d2} | {sev_badge} | {desc} |\n"
        
        # === Clinical Summary (Collapsible) ===
        llm_summary = generate_llm_summary(
            sorted_interactions, drug_names, risk_level, regimen_pri,
            resolved_drugs=drugs, shared_se=shared_se, shared_proteins=shared_proteins
        )
        report += f"""

<details open>
<summary style="font-weight:600; font-size:15px; cursor:pointer; padding:12px 0;">Clinical Summary</summary>

{llm_summary}

</details>"""
    else:
        report += f"\n\n### No Interactions Found\nNo known drug-drug interactions detected between {', '.join(drug_names)}."
    
    # === PRI RISK ASSESSMENT (collapsible) ===
    if regimen_pri and regimen_pri.get('drug_pris'):
        pri_content = ""
        
        # Show highest risk drug prominently
        if regimen_pri.get('highest_risk_drug'):
            highest = regimen_pri['highest_risk_drug']
            highest_pri = regimen_pri['max_pri']
            pri_content += f"**Highest Risk:** {highest.title()} (PRI: {highest_pri:.3f})\n\n"
        
        # Summary counts
        high_count = regimen_pri.get('num_high_risk', 0)
        med_count = regimen_pri.get('num_medium_risk', 0)
        if high_count > 0 or med_count > 0:
            risk_counts = []
            if high_count > 0:
                risk_counts.append(f"{high_count} High Risk")
            if med_count > 0:
                risk_counts.append(f"{med_count} Medium Risk")
            pri_content += f"{' | '.join(risk_counts)}\n\n"
        
        # PRI details table
        pri_content += """| Drug | PRI Score | Risk Level | Interactions | Severe |
|------|-----------|------------|--------------|--------|"""
        for drug_name, pri_data in regimen_pri['drug_pris'].items():
            pri_content += f"\n| {drug_name.title()} | {pri_data['pri']:.3f} | {pri_data['risk_level']} | {pri_data['num_interactions']} | {pri_data['num_severe']} |"
        
        pri_content += """

**Formula:** PRI = 0.25×Degree + 0.30×WeightedDegree + 0.20×Betweenness + 0.25×SeverityProfile
**Thresholds:** High Risk (>0.5) • Medium Risk (0.3-0.5) • Lower Risk (<0.3)"""
        
        report += f"""

<details>
<summary style="font-weight:600; font-size:15px; cursor:pointer; padding:12px 0;">Polypharmacy Risk Index</summary>

{pri_content}

</details>"""
    
    # === LLM-Generated Alternatives Recommendations (Collapsible) ===
    if alternatives_map:
        alternatives_text = generate_llm_alternatives(alternatives_map, drug_names, interactions)
        if alternatives_text:
            report += f"""

<details>
<summary style="font-weight:600; font-size:15px; cursor:pointer; padding:12px 0;">Alternative Recommendations</summary>

{alternatives_text}

<details style="margin-top:12px;">
<summary style="font-size:13px;">View ARS Score Details</summary>

| Original Drug | Alternative | ARS Score | Severity Reduction | PRI Improvement |
|---------------|-------------|-----------|-------------------|-----------------|"""
            for orig, alts in alternatives_map.items():
                if not alts:
                    continue
                for alt in alts[:3]:
                    ars = alt.get('ars', 0)
                    sev_red = alt.get('normalized_sev_red', 0) * 100
                    pri_imp = alt.get('pri_improvement', 0) * 100
                    alt_link = f"[{alt['name'].title()}](https://go.drugbank.com/drugs/{alt['drugbank_id']})"
                    report += f"\n| {orig.title()} | {alt_link} | **{ars:.3f}** | {sev_red:.0f}% | {pri_imp:.1f}% |"
            
            report += """

**ARS Formula:** ARS = 0.70×(Severity Reduction) + 0.30×(PRI Improvement)
*Higher ARS scores indicate better alternatives that reduce both interaction severity and overall polypharmacy network risk.*

</details>

</details>"""
    
    # === LLM-Generated Monitoring Recommendations (Collapsible) ===
    if interactions:
        monitoring_recommendations = generate_llm_monitoring(
            interactions, drug_names, shared_se, shared_proteins
        )
        if monitoring_recommendations:
            report += f"""

<details>
<summary style="font-weight:600; font-size:15px; cursor:pointer; padding:12px 0;">Monitoring Recommendations</summary>

{monitoring_recommendations}

</details>"""
    
    # === COLLAPSIBLE: Drug Details ===
    report += """

<details>
<summary>Drug Details</summary>

| Drug | DrugBank ID |
|------|-------------|"""
    for d in drugs:
        db_id = d.get('drugbank_id', 'N/A')
        report += f"\n| {d.get('name', 'Unknown').title()} | [{db_id}](https://go.drugbank.com/drugs/{db_id}) |"
    report += "\n\n</details>"
    
    # === COLLAPSIBLE: Molecular (only if data exists) ===
    if shared_se or shared_proteins:
        report += """

<details>
<summary>Molecular Overlap</summary>

"""
        if shared_proteins:
            prots = [f"{name} ({data['gene']})" for name, data in list(shared_proteins.items())[:3]]
            report += f"**Shared targets:** {', '.join(prots)}\n\n"
        if shared_se:
            ses = [f"{name}" for name, _ in list(shared_se.items())[:5]]
            report += f"**Overlapping side effects:** {', '.join(ses)}\n\n"
        report += "</details>"
    
    # === COMPACT FOOTER ===
    report += f"""

---
<p style="text-align:center; font-size:0.8em; color:#888;">
Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} • DrugBank • SIDER • UniProt<br/>
<em>For educational purposes only — consult healthcare professionals</em>
</p>
"""
    
    return report


# ============================================================
# LLM Chat
# ============================================================

def extract_text_from_message(msg):
    """Extract plain text from various Gradio 6 message formats"""
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        # Check for 'content' key (standard format)
        if 'content' in msg:
            content = msg['content']
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Content might be list of parts
                texts = []
                for part in content:
                    if isinstance(part, str):
                        texts.append(part)
                    elif isinstance(part, dict) and 'text' in part:
                        texts.append(part['text'])
                return ' '.join(texts)
        # Check for 'text' key directly
        if 'text' in msg:
            return msg['text']
    if isinstance(msg, list):
        # List of message parts
        texts = []
        for part in msg:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and 'text' in part:
                texts.append(part['text'])
        return ' '.join(texts)
    return str(msg)


def chat(message, history, model_name):
    """
    Natural Conversation Chat
    
    Uses conversation memory and stored analysis report for
    fluid, ChatGPT-like dialogue about drug interactions.
    """
    try:
        # Extract clean text from message (handles Gradio 6 format)
        clean_message = extract_text_from_message(message)
        if not clean_message.strip():
            return history, ""
        
        # Get the natural chat assistant
        assistant = get_chat_assistant()
        
        # Generate natural response using conversation memory
        response = assistant.respond(clean_message, model_name)
        
        # Handle empty or error response
        if not response or response.startswith("[LLM Error"):
            response = f"I apologize, but I'm having trouble connecting to the language model. Please make sure Ollama is running (`ollama serve`) and the model is available.\n\nError: {response}"
        
        # Return in Gradio 4.x tuple format [(user_msg, assistant_msg), ...]
        if history is None:
            history = []
        
        history.append((clean_message, response))
        return history, ""
    except Exception as e:
        # Handle any unexpected errors
        if history is None:
            history = []
        error_msg = f"An error occurred: {str(e)}"
        history.append((str(message), error_msg))
        return history, ""


# ============================================================
# Gradio Interface
# ============================================================

def create_app():
    
    # Beautiful iOS-style CSS with vibrant colors
    custom_css = """
    :root {
        --blue: #007AFF;
        --green: #34C759;
        --red: #FF3B30;
        --orange: #FF9500;
        --gray-1: #F2F2F7;
        --gray-2: #E5E5EA;
        --gray-3: #8E8E93;
        --text: #1C1C1E;
        --white: #FFFFFF;
    }
    
    /* Full width layout */
    .gradio-container { 
        max-width: 100% !important; 
        background: var(--gray-1) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .main { max-width: 100% !important; padding: 20px 30px !important; }
    .contain, .wrap { max-width: 100% !important; }
    
    /* Section title cards */
    .section-title { 
        display: block !important;
        font-size: 13px !important; 
        font-weight: 700 !important; 
        color: var(--white) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 20px !important; 
        padding: 12px 18px !important;
        background: linear-gradient(135deg, #5AC8FA 0%, #AF52DE 100%) !important;
        border-radius: 10px !important;
        box-shadow: 0 3px 10px rgba(90,200,250,0.3) !important;
    }
    
    /* Clean buttons */
    .gr-button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
    }
    .gr-button-primary {
        background: var(--blue) !important;
        color: white !important;
    }
    .gr-button-secondary {
        background: var(--white) !important;
        color: var(--blue) !important;
        border: 1px solid var(--blue) !important;
    }
    .gr-button-stop {
        background: var(--white) !important;
        color: var(--red) !important;
        border: 1px solid var(--red) !important;
    }
    button[size="lg"] {
        padding: 14px 28px !important;
        font-size: 15px !important;
    }
    
    /* Form inputs */
    .gr-textbox, .gr-dropdown { 
        border-radius: 10px !important;
        border: 1px solid var(--gray-2) !important;
        background: var(--white) !important;
    }
    .gr-textbox:focus-within, .gr-dropdown:focus-within {
        border-color: var(--blue) !important;
    }
    
    /* Three-column panels - remove Gradio default boxes */
    .column-panel,
    .gr-column,
    .gradio-column,
    [class*="column"],
    .gr-row > div,
    .gr-block { 
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        border-radius: 0 !important;
    }
    /* Keep padding on main columns only */
    .gr-row > div {
        padding: 0 12px !important;
    }
    .gr-row > div:first-child {
        padding-left: 0 !important;
    }
    .gr-row > div:last-child {
        padding-right: 0 !important;
    }
    
    /* Collapsible sections */
    details { 
        background: #f8f9fa;
        padding: 14px 18px; 
        border-radius: 12px; 
        margin: 12px 0; 
        border: 1px solid var(--gray-2);
    }
    details[open] {
        background: var(--white);
        border-color: var(--blue);
    }
    details summary { 
        cursor: pointer; 
        font-weight: 600; 
        color: var(--blue);
        font-size: 14px;
        list-style: none;
    }
    details summary::-webkit-details-marker { display: none; }
    
    /* Chatbot */
    .chatbot { 
        border-radius: 16px !important; 
        border: 1px solid var(--gray-2) !important;
        background: linear-gradient(180deg, #fafafa 0%, #f5f5f7 100%) !important;
    }
    .chatbot .message, .chatbot .message p, .chatbot .message span,
    .chatbot .wrap, .chatbot .wrap p, .chatbot .wrap span,
    .chatbot [data-testid="bot"], .chatbot [data-testid="user"],
    .chatbot .prose, .chatbot .prose p, .chatbot .prose span,
    .chatbot .markdown, .chatbot .markdown p {
        font-size: 10px !important;
        line-height: 1.35 !important;
        font-weight: 400 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif !important;
    }
    .chatbot .message {
        border-radius: 18px !important;
        padding: 6px 10px !important;
        max-width: 85% !important;
    }
    .chatbot .user, .chatbot .user p, .chatbot [data-testid="user"] {
        background: linear-gradient(135deg, #5AC8FA 0%, #AF52DE 100%) !important;
        color: white !important;
        font-size: 10px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif !important;
    }
    .chatbot .bot, .chatbot .bot p, .chatbot [data-testid="bot"] {
        background: var(--white) !important;
        border: 1px solid var(--gray-2) !important;
        color: #1D1D1F !important;
        font-size: 10px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Tables */
    table { 
        width: 100%; 
        border-collapse: collapse;
        font-size: 13px;
        margin: 16px 0;
        border-radius: 12px;
        overflow: hidden;
        background: var(--white);
    }
    th { 
        background: #f8f9fa;
        font-weight: 600; 
        text-align: left;
        padding: 12px 16px;
        color: var(--blue);
        font-size: 11px;
        text-transform: uppercase;
        border-bottom: 2px solid var(--blue);
    }
    td { 
        padding: 12px 16px; 
        border-bottom: 1px solid var(--gray-2);
    }
    tr:last-child td { border-bottom: none; }
    
    /* Tabs */
    .tab-nav {
        background: var(--gray-1) !important;
        border-radius: 8px !important;
        padding: 3px !important;
    }
    .tab-nav button {
        font-weight: 600 !important;
        font-size: 13px !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        border: none !important;
        background: transparent !important;
    }
    .tab-nav button.selected {
        background: var(--white) !important;
        color: var(--blue) !important;
    }
    
    /* Accordion */
    .accordion { 
        border-radius: 12px !important;
        border: 1px solid var(--gray-2) !important;
        background: var(--white) !important;
    }
    
    /* Checkboxes */
    .gr-checkbox-group label {
        font-size: 14px !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
    }
    
    /* Typography */
    .markdown-text h3 {
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
        color: var(--text) !important;
    }
    .markdown-text p {
        line-height: 1.6 !important;
    }
    
    /* Links */
    a { color: var(--blue) !important; text-decoration: none !important; }
    
    /* Image upload */
    .gr-image {
        border-radius: 12px !important;
        border: 2px dashed var(--blue) !important;
        background: linear-gradient(135deg, #f8f9fa 0%, #e8f4fd 100%) !important;
        min-height: 220px !important;
        overflow: visible !important;
    }
    .gr-image:hover {
        border-color: #AF52DE !important;
        background: linear-gradient(135deg, #e8f4fd 0%, #f3e8ff 100%) !important;
    }
    .gr-image img {
        max-height: 180px !important;
        object-fit: contain !important;
        border-radius: 8px !important;
    }
    .gr-image [data-testid="image"], .gr-image .image-container,
    .gr-image .upload-container {
        min-height: auto !important;
        overflow: visible !important;
    }
    /* Hide ALL icons in image upload */
    .gr-image .icon-buttons,
    .gr-image svg,
    .gr-image .upload-icon,
    .gr-image [data-testid="upload-icon"],
    .gr-image .icon,
    .gr-image button[aria-label],
    .gr-image .source-selection,
    .gr-image .webcam-icon,
    .gr-image .upload-text,
    .gr-image .source-container,
    .gr-image .source-icon,
    .gr-image [data-testid="upload"],
    .gr-image .wrap svg,
    .gr-image .upload svg,
    .gr-image path,
    .gr-image .pending,
    .gr-image .upload-button,
    [data-testid="image"] svg,
    [data-testid="image"] .upload,
    [data-testid="image"] .icon-wrap {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--gray-1); }
    ::-webkit-scrollbar-thumb { background: var(--gray-3); border-radius: 3px; }
    """
    
    with gr.Blocks(title="DDI Risk Analyzer", css=custom_css) as app:
        
        gr.HTML("""
        <div style="background:linear-gradient(135deg, #5AC8FA 0%, #AF52DE 100%); border-radius:20px; padding:32px 40px; margin-bottom:24px; box-shadow:0 8px 32px rgba(90,200,250,0.3); text-align:center;">
        <h1 style="margin:0; font-size:32px; font-weight:800; color:#fff; letter-spacing:-0.5px;">Drug Interaction Analyzer</h1>
        <p style="margin:10px 0 0 0; color:rgba(255,255,255,0.9); font-size:16px; font-weight:500;">AI-based DDI Risk Analysis and Alternative Recommendation</p>
        </div>
        """)
        
        # Three columns layout with equal spacing - full width
        with gr.Row(equal_height=True):
            # ============================================================
            # COLUMN 1: Drug Input (List, Narrative, or Image)
            # ============================================================
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("INPUT", elem_classes=["section-title"])
                
                with gr.Tabs():
                    # Tab 1: Drug List
                    with gr.Tab("Drugs"):
                        drug_input = gr.Textbox(
                            label="Enter medications",
                            placeholder="warfarin, aspirin, metoprolol, lisinopril",
                            lines=3,
                            info="Enter drug names separated by commas"
                        )
                    
                    # Tab 2: Narrative Description
                    with gr.Tab("Text"):
                        narrative_input = gr.Textbox(
                            label="Describe your medications",
                            placeholder="I take cardace 5mg in the morning and aspirin 81mg at night for my heart condition...",
                            lines=4,
                            info="We'll extract drug names automatically"
                        )
                        extract_narrative_btn = gr.Button("Extract Drugs", variant="secondary")
                        narrative_status = gr.Markdown("*Describe your medications in your own words*")
                    
                    # Tab 3: Image Upload
                    with gr.Tab("Photo"):
                        image_input = gr.Image(
                            label="Upload prescription or medication photo",
                            type="pil",
                            height=240,
                            sources=["upload", "clipboard"],
                            show_download_button=False,
                            show_share_button=False
                        )
                        extract_image_btn = gr.Button("Extract Drugs from Photo", variant="secondary")
                        image_status = gr.Markdown("*Upload or paste a photo of your prescription*")
                
                with gr.Row():
                    quick_check_btn = gr.Button("Validate", variant="primary")
                    clear_btn = gr.Button("Clear", variant="stop")
                
                # Drug identification preview
                preview_output = gr.Markdown(
                    value="*Enter medications and click Validate to identify them*"
                )
                
                # Editable confirmed drugs list (shows after checking)
                with gr.Group(visible=False) as edit_group:
                    gr.Markdown("**Confirmed medications**")
                    confirmed_drugs = gr.Textbox(
                        label="",
                        placeholder="Edit drug names here...",
                        lines=2,
                        info="Edit and click Re-validate to update"
                    )
                    with gr.Row():
                        recheck_btn = gr.Button("Re-validate", variant="secondary", size="sm")
                        use_list_btn = gr.Button("Use List", variant="primary", size="sm")
                
                # Edit panel (shows after analysis for alternatives)
                with gr.Accordion("Alternative Options", open=True, visible=False) as selection_panel:
                    gr.Markdown("**Current medications**")
                    current_drugs_check = gr.CheckboxGroup(choices=[], value=[], label="", interactive=True)
                    gr.Markdown("**Suggested alternatives**")
                    alternatives_check = gr.CheckboxGroup(choices=[], value=[], label="", interactive=True)
                    
                    reanalyze_btn = gr.Button("Re-analyze", variant="primary")
                    selection_status = gr.Markdown("")
                
                analyze_btn = gr.Button("Analyze Interactions", variant="primary", size="lg")
            
            # ============================================================
            # COLUMN 2: Report
            # ============================================================
            with gr.Column(scale=3, min_width=400):
                gr.Markdown("REPORT", elem_classes=["section-title"])
                
                report_output = gr.Markdown(
                    value="*Enter medications and click Analyze Interactions*"
                )
            
            # ============================================================
            # COLUMN 3: Chat Assistant
            # ============================================================
            with gr.Column(scale=2, min_width=350):
                gr.Markdown("ASSISTANT", elem_classes=["section-title"])
                
                gr.Markdown("""<div style="display:flex; align-items:center; gap:8px; padding:8px 12px; background:#f8f9fa; border-radius:8px; margin-bottom:12px;">
<div style="width:6px; height:6px; background:#34C759; border-radius:50%;"></div>
<span style="font-size:11px; color:#3C3C43; font-family:-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;">Llama 3 7B</span>
</div>""")
                
                # Hidden dropdown for backend compatibility
                model_select = gr.Dropdown(
                    choices=list(LLMClient.MODELS.keys()),
                    value="Llama3",
                    visible=False
                )
                
                chatbot = gr.Chatbot(
                    height=400, 
                    label="",
                    show_label=False,
                    avatar_images=(None, None),
                    bubble_full_width=False
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask about drug interactions, alternatives, or mechanisms...",
                        label="",
                        show_label=False,
                        lines=1,
                        scale=4,
                        container=False
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Event handlers
        quick_check_btn.click(
            identify_drugs_preview, 
            inputs=[drug_input], 
            outputs=[preview_output, confirmed_drugs, edit_group], 
            show_progress="full"
        )
        
        # Re-check edited drug list
        recheck_btn.click(
            identify_drugs_preview, 
            inputs=[confirmed_drugs], 
            outputs=[preview_output, confirmed_drugs, edit_group], 
            show_progress="full"
        )
        
        # Use confirmed list and copy to main input
        def use_confirmed_list(drugs):
            """Copy confirmed drugs to main input"""
            return drugs
        
        use_list_btn.click(
            use_confirmed_list,
            inputs=[confirmed_drugs],
            outputs=[drug_input]
        )
        
        # Wrapper to use confirmed_drugs if available, else drug_input
        def analyze_with_fallback(confirmed, raw_input, progress=gr.Progress()):
            """Use confirmed drugs if available, otherwise use raw input"""
            drugs_to_analyze = confirmed if confirmed and confirmed.strip() else raw_input
            return analyze_ddi(drugs_to_analyze, progress)
        
        analyze_btn.click(
            analyze_with_fallback, 
            inputs=[confirmed_drugs, drug_input], 
            outputs=[report_output, selection_panel, current_drugs_check, alternatives_check, selection_status], 
            show_progress="full"
        )
        
        drug_input.submit(
            identify_drugs_preview,
            inputs=[drug_input],
            outputs=[preview_output, confirmed_drugs, edit_group],
            show_progress="full"
        )
        
        reanalyze_btn.click(
            reanalyze_with_selection,
            inputs=[current_drugs_check, alternatives_check],
            outputs=[report_output, selection_panel, current_drugs_check, alternatives_check, selection_status],
            show_progress="full"
        )
        
        # Narrative extraction handler
        extract_narrative_btn.click(
            extract_drugs_from_narrative,
            inputs=[narrative_input],
            outputs=[drug_input, narrative_status],
            show_progress="full"
        )
        
        # Image extraction handler
        extract_image_btn.click(
            extract_drugs_from_image,
            inputs=[image_input],
            outputs=[drug_input, image_status],
            show_progress="full"
        )
        
        def clear_all():
            return (
                "",  # drug_input
                "",  # narrative_input 
                None,  # image_input
                "*Enter medications and click Validate to identify them*",  # preview_output
                "",  # confirmed_drugs
                gr.update(visible=False),  # edit_group
                "*Enter medications and click Analyze Interactions*",  # report_output
                [],  # chatbot
                gr.update(visible=False),  # selection_panel
                gr.update(choices=[], value=[]),  # current_drugs_check
                gr.update(choices=[], value=[]),  # alternatives_check
                "",  # selection_status
                "*Describe your medications in your own words*",  # narrative_status
                "*Upload or paste a photo of your prescription*"  # image_status
            )
        
        clear_btn.click(
            clear_all, 
            outputs=[drug_input, narrative_input, image_input, preview_output, confirmed_drugs, edit_group, report_output, chatbot,
                     selection_panel, current_drugs_check, alternatives_check, selection_status, narrative_status, image_status]
        )
        
        send_btn.click(chat, inputs=[chat_input, chatbot, model_select], outputs=[chatbot, chat_input])
        chat_input.submit(chat, inputs=[chat_input, chatbot, model_select], outputs=[chatbot, chat_input])
    
    return app


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DDI Risk Analyzer with Alternatives")
    print("="*60)
    
    print("\nLoading Knowledge Graph...")
    result = kg.load()
    print(f"   {result}")
    
    print("\nStarting application...")
    print("   URL: http://0.0.0.0:7860")
    print("   Press Ctrl+C to stop\n")
    
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
