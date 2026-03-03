"""
FAERS Integration Module - FDA Adverse Event Reporting System API Client

Provides external validation of drug interaction risk using real-world adverse event data
from the FDA FAERS database via the OpenFDA API.

API Documentation: https://open.fda.gov/apis/drug/event/
Rate Limit: 240 requests/minute without API key
"""

import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class FAERSAdverseEvent:
    """Represents an adverse event from FAERS"""
    reaction_term: str
    count: int
    is_serious: bool = False


@dataclass
class FAERSDrugProfile:
    """FAERS safety profile for a single drug"""
    drug_name: str
    total_reports: int = 0
    serious_reports: int = 0
    death_reports: int = 0
    adverse_events: List[FAERSAdverseEvent] = field(default_factory=list)
    top_signals: List[str] = field(default_factory=list)
    faers_risk_score: float = 0.0
    serious_event_ratio: float = 0.0
    query_success: bool = False
    error_message: str = ""


@dataclass
class FAERSInteractionSignal:
    """FAERS signal for a drug-drug interaction"""
    drug1: str
    drug2: str
    concomitant_reports: int = 0
    drug1_alone_reports: int = 0
    drug2_alone_reports: int = 0
    interaction_signal_score: float = 0.0
    common_adverse_events: List[str] = field(default_factory=list)
    query_success: bool = False
    error_message: str = ""


class FAERSClient:
    """
    Client for FDA FAERS API queries
    
    Endpoints:
    - Drug adverse events: /drug/event.json
    - Serious events filter: serious:1
    - Death reports filter: seriousnessdeath:1
    """
    
    BASE_URL = "https://api.fda.gov/drug/event.json"
    RATE_LIMIT_DELAY = 0.25  # 4 requests/second = 240/minute
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.request_count = 0
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, search: str, count: Optional[str] = None, 
                     limit: int = 100) -> Dict:
        """Make API request with rate limiting"""
        self._rate_limit()
        
        params = {
            "search": search,
            "limit": limit
        }
        if count:
            params["count"] = count
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"error": "No results found", "status": 404}
            else:
                return {"error": f"API error: {response.status_code}", 
                       "status": response.status_code}
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": -1}
    
    def get_drug_total_reports(self, drug_name: str) -> int:
        """Get total adverse event reports for a drug"""
        search = f'patient.drug.medicinalproduct:"{drug_name}"'
        result = self._make_request(search, limit=1)
        
        if "meta" in result:
            return result["meta"]["results"]["total"]
        return 0
    
    def get_drug_serious_reports(self, drug_name: str) -> int:
        """Get serious adverse event reports for a drug"""
        search = f'patient.drug.medicinalproduct:"{drug_name}" AND serious:1'
        result = self._make_request(search, limit=1)
        
        if "meta" in result:
            return result["meta"]["results"]["total"]
        return 0
    
    def get_drug_death_reports(self, drug_name: str) -> int:
        """Get death reports for a drug"""
        search = f'patient.drug.medicinalproduct:"{drug_name}" AND seriousnessdeath:1'
        result = self._make_request(search, limit=1)
        
        if "meta" in result:
            return result["meta"]["results"]["total"]
        return 0
    
    def get_drug_top_reactions(self, drug_name: str, limit: int = 20) -> List[Tuple[str, int]]:
        """Get top adverse reactions for a drug"""
        search = f'patient.drug.medicinalproduct:"{drug_name}"'
        result = self._make_request(search, 
                                   count="patient.reaction.reactionmeddrapt.exact",
                                   limit=limit)
        
        if "results" in result:
            return [(r["term"], r["count"]) for r in result["results"]]
        return []
    
    def get_concomitant_reports(self, drug1: str, drug2: str) -> int:
        """Get reports where both drugs were taken together"""
        search = (f'patient.drug.medicinalproduct:"{drug1}" AND '
                 f'patient.drug.medicinalproduct:"{drug2}"')
        result = self._make_request(search, limit=1)
        
        if "meta" in result:
            return result["meta"]["results"]["total"]
        return 0
    
    def get_drug_profile(self, drug_name: str) -> FAERSDrugProfile:
        """Get complete FAERS safety profile for a drug"""
        profile = FAERSDrugProfile(drug_name=drug_name)
        
        try:
            # Get counts
            profile.total_reports = self.get_drug_total_reports(drug_name)
            
            if profile.total_reports == 0:
                profile.error_message = "No FAERS data found for this drug"
                return profile
                
            profile.serious_reports = self.get_drug_serious_reports(drug_name)
            profile.death_reports = self.get_drug_death_reports(drug_name)
            
            # Calculate ratios
            if profile.total_reports > 0:
                profile.serious_event_ratio = profile.serious_reports / profile.total_reports
            
            # Get top reactions
            top_reactions = self.get_drug_top_reactions(drug_name, limit=10)
            profile.adverse_events = [
                FAERSAdverseEvent(reaction_term=term, count=count)
                for term, count in top_reactions
            ]
            profile.top_signals = [term for term, _ in top_reactions[:5]]
            
            # Calculate risk score (weighted combination)
            # Higher score = higher real-world safety concern
            profile.faers_risk_score = (
                0.4 * min(1.0, profile.serious_event_ratio * 2) +  # Serious ratio (0-1)
                0.3 * min(1.0, profile.death_reports / max(profile.total_reports, 1) * 100) +  # Death ratio
                0.3 * min(1.0, profile.total_reports / 100000)  # Volume (normalized)
            )
            
            profile.query_success = True
            
        except Exception as e:
            profile.error_message = str(e)
            
        return profile
    
    def get_interaction_signal(self, drug1: str, drug2: str) -> FAERSInteractionSignal:
        """Analyze interaction signal between two drugs"""
        signal = FAERSInteractionSignal(drug1=drug1, drug2=drug2)
        
        try:
            # Get concomitant reports
            signal.concomitant_reports = self.get_concomitant_reports(drug1, drug2)
            signal.drug1_alone_reports = self.get_drug_total_reports(drug1)
            signal.drug2_alone_reports = self.get_drug_total_reports(drug2)
            
            # Calculate interaction signal score
            # Using disproportionality analysis concept
            if signal.drug1_alone_reports > 0 and signal.drug2_alone_reports > 0:
                expected = (signal.drug1_alone_reports * signal.drug2_alone_reports) / 1e6
                observed = signal.concomitant_reports
                if expected > 0:
                    signal.interaction_signal_score = observed / max(expected, 1)
            
            # Get common adverse events when taken together
            if signal.concomitant_reports > 0:
                search = (f'patient.drug.medicinalproduct:"{drug1}" AND '
                         f'patient.drug.medicinalproduct:"{drug2}"')
                result = self._make_request(
                    search, 
                    count="patient.reaction.reactionmeddrapt.exact",
                    limit=10
                )
                if "results" in result:
                    signal.common_adverse_events = [r["term"] for r in result["results"]]
            
            signal.query_success = True
            
        except Exception as e:
            signal.error_message = str(e)
            
        return signal


class FAERSValidator:
    """
    Validates network-based risk scores against FAERS real-world data
    
    Provides external validation by correlating:
    - Network PRI scores with FAERS serious event ratios
    - Network interaction weights with FAERS concomitant report signals
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = FAERSClient(api_key)
        self.drug_profiles: Dict[str, FAERSDrugProfile] = {}
        self.interaction_signals: Dict[Tuple[str, str], FAERSInteractionSignal] = {}
        
    def validate_drug_risk(self, drug_name: str, network_pri: float) -> Dict[str, Any]:
        """Validate a drug's network PRI against FAERS data"""
        if drug_name not in self.drug_profiles:
            self.drug_profiles[drug_name] = self.client.get_drug_profile(drug_name)
        
        profile = self.drug_profiles[drug_name]
        
        return {
            "drug_name": drug_name,
            "network_pri": network_pri,
            "faers_risk_score": profile.faers_risk_score,
            "serious_event_ratio": profile.serious_event_ratio,
            "total_reports": profile.total_reports,
            "death_reports": profile.death_reports,
            "top_signals": profile.top_signals,
            "validation_status": "success" if profile.query_success else "failed",
            "error": profile.error_message
        }
    
    def validate_interaction_risk(self, drug1: str, drug2: str, 
                                  network_weight: float) -> Dict[str, Any]:
        """Validate interaction risk against FAERS concomitant data"""
        key = tuple(sorted([drug1, drug2]))
        
        if key not in self.interaction_signals:
            self.interaction_signals[key] = self.client.get_interaction_signal(
                drug1, drug2
            )
        
        signal = self.interaction_signals[key]
        
        return {
            "drug_pair": f"{drug1} + {drug2}",
            "network_weight": network_weight,
            "concomitant_reports": signal.concomitant_reports,
            "interaction_signal_score": signal.interaction_signal_score,
            "common_adverse_events": signal.common_adverse_events,
            "validation_status": "success" if signal.query_success else "failed",
            "error": signal.error_message
        }
    
    def batch_validate_drugs(self, drugs_with_pri: List[Tuple[str, float]], 
                            progress_callback=None) -> List[Dict[str, Any]]:
        """Validate multiple drugs against FAERS"""
        results = []
        
        for i, (drug_name, pri) in enumerate(drugs_with_pri):
            result = self.validate_drug_risk(drug_name, pri)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(drugs_with_pri), drug_name)
        
        return results
    
    def calculate_correlation(self, validations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlation between network and FAERS metrics"""
        import numpy as np
        
        # Filter successful validations
        valid = [v for v in validations if v["validation_status"] == "success" 
                 and v["total_reports"] > 100]  # Minimum data threshold
        
        if len(valid) < 5:
            return {
                "pearson_correlation": None,
                "spearman_correlation": None,
                "sample_size": len(valid),
                "error": "Insufficient data for correlation"
            }
        
        network_scores = [v["network_pri"] for v in valid]
        faers_scores = [v["faers_risk_score"] for v in valid]
        
        # Pearson correlation
        pearson = np.corrcoef(network_scores, faers_scores)[0, 1]
        
        # Spearman rank correlation
        from scipy import stats
        spearman, p_value = stats.spearmanr(network_scores, faers_scores)
        
        return {
            "pearson_correlation": float(pearson),
            "spearman_correlation": float(spearman),
            "spearman_p_value": float(p_value),
            "sample_size": len(valid),
            "network_mean": float(np.mean(network_scores)),
            "faers_mean": float(np.mean(faers_scores))
        }


def test_faers_connection():
    """Test FAERS API connectivity"""
    print("Testing FAERS API connection...")
    client = FAERSClient()
    
    # Test with common drugs
    test_drugs = ["Warfarin", "Aspirin", "Metoprolol"]
    
    for drug in test_drugs:
        print(f"\n  Testing: {drug}")
        total = client.get_drug_total_reports(drug)
        serious = client.get_drug_serious_reports(drug)
        print(f"    Total reports: {total:,}")
        print(f"    Serious reports: {serious:,}")
        if total > 0:
            print(f"    Serious ratio: {serious/total:.2%}")
    
    print("\n[OK] FAERS API connection successful")


if __name__ == "__main__":
    test_faers_connection()
