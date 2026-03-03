"""
LLM Client - Interface for BioMistral-7B via Ollama
"""

import json
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class OllamaClient(BaseLLMClient):
    """
    Ollama client for running BioMistral-7B locally
    
    Setup:
        brew install ollama
        ollama pull biomistral
        # or: ollama pull llama3:8b (fallback)
    """
    
    def __init__(self, 
                 model: str = "biomistral",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 120):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.api_generate = f"{base_url}/api/generate"
        self.api_tags = f"{base_url}/api/tags"
        
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(self.api_tags, timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                return self.model.split(':')[0] in model_names or len(models) > 0
            return False
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(self.api_tags, timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m.get('name', '') for m in models]
            return []
        except:
            return []
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 **kwargs) -> LLMResponse:
        """
        Generate response from BioMistral
        
        Args:
            prompt: The user prompt
            system_prompt: System instructions
            temperature: Creativity (0-1)
            max_tokens: Max response length
            
        Returns:
            LLMResponse with generated content
        """
        try:
            # Build the full prompt
            full_prompt = ""
            if system_prompt:
                full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            else:
                full_prompt = f"<s>[INST] {prompt} [/INST]"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                self.api_generate,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    content=result.get('response', ''),
                    model=self.model,
                    tokens_used=result.get('eval_count', 0),
                    success=True
                )
            else:
                return LLMResponse(
                    content="",
                    model=self.model,
                    success=False,
                    error=f"API error: {response.status_code}"
                )
                
        except requests.exceptions.Timeout:
            return LLMResponse(
                content="",
                model=self.model,
                success=False,
                error="Request timed out"
            )
        except requests.exceptions.ConnectionError:
            return LLMResponse(
                content="",
                model=self.model,
                success=False,
                error="Cannot connect to Ollama. Is it running? (ollama serve)"
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                success=False,
                error=str(e)
            )


class BioMistralClient(OllamaClient):
    """
    Specialized client for BioMistral-7B medical LLM
    
    BioMistral is fine-tuned on biomedical literature and understands:
    - Drug interactions
    - Medical terminology
    - Clinical recommendations
    - Pharmacology concepts
    """
    
    # Medical system prompts
    CLINICAL_SYSTEM_PROMPT = """You are a clinical pharmacology AI assistant specialized in drug-drug interactions (DDI) and polypharmacy risk assessment. 

Your role is to:
1. Explain drug interactions in clear, professional clinical language
2. Assess the severity and clinical significance of interactions based on the DDI network data
3. Provide actionable recommendations for healthcare providers
4. Use evidence-based reasoning from the interaction descriptions

Severity Levels in the DDI Network:
- Contraindicated interaction: Must never be used together, life-threatening risk
- Major interaction: Serious clinical consequences, requires intervention
- Moderate interaction: May require monitoring or dose adjustment
- Minor interaction: Limited clinical significance

Always be accurate, concise, and clinically relevant. Base your analysis on the actual interaction descriptions provided."""

    PATIENT_SYSTEM_PROMPT = """You are a friendly healthcare assistant explaining medication safety to patients.

Your role is to:
1. Explain drug interactions in simple, non-technical language
2. Reassure patients while being honest about risks
3. Encourage patients to talk to their healthcare providers
4. Avoid medical jargon

Be warm, supportive, and easy to understand. Never frighten patients unnecessarily."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        # Try biomistral first, fallback to other medical models
        super().__init__(model="biomistral", base_url=base_url, timeout=timeout)
        
        # Check if biomistral is available, if not try alternatives
        if not self._check_model_available("biomistral"):
            alternatives = ["meditron", "medllama", "llama3:8b", "mistral"]
            for alt in alternatives:
                if self._check_model_available(alt):
                    self.model = alt
                    print(f"Using {alt} (biomistral not found)")
                    break
    
    def _check_model_available(self, model: str) -> bool:
        """Check if a specific model is available"""
        try:
            response = requests.get(self.api_tags, timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                return model.split(':')[0] in model_names
            return False
        except:
            return False
    
    def generate_clinical_explanation(self, 
                                      interactions: List[Dict],
                                      risk_assessment: Dict,
                                      drugs: List[str]) -> str:
        """
        Generate clinical explanation for DDIs
        
        Args:
            interactions: List of detected interactions
            risk_assessment: Risk score and level
            drugs: List of drug names
            
        Returns:
            Clinical explanation text
        """
        # Build context prompt
        prompt = self._build_clinical_prompt(interactions, risk_assessment, drugs)
        
        response = self.generate(
            prompt=prompt,
            system_prompt=self.CLINICAL_SYSTEM_PROMPT,
            temperature=0.3,  # Lower temperature for clinical accuracy
            max_tokens=1500
        )
        
        if response.success:
            return response.content
        else:
            return f"[LLM unavailable: {response.error}]"
    
    def generate_patient_explanation(self,
                                     interactions: List[Dict],
                                     risk_assessment: Dict,
                                     drugs: List[str]) -> str:
        """
        Generate patient-friendly explanation
        
        Args:
            interactions: List of detected interactions
            risk_assessment: Risk score and level
            drugs: List of drug names
            
        Returns:
            Patient-friendly explanation text
        """
        prompt = self._build_patient_prompt(interactions, risk_assessment, drugs)
        
        response = self.generate(
            prompt=prompt,
            system_prompt=self.PATIENT_SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=800
        )
        
        if response.success:
            return response.content
        else:
            return f"[LLM unavailable: {response.error}]"
    
    def generate_alternative_rationale(self,
                                       original_drug: str,
                                       alternative_drug: str,
                                       reason: str) -> str:
        """
        Generate explanation for why an alternative is recommended
        """
        prompt = f"""Explain why {alternative_drug} might be a safer alternative to {original_drug} for a patient.

Context: {reason}

Provide a brief (2-3 sentence) clinical rationale."""

        response = self.generate(
            prompt=prompt,
            system_prompt=self.CLINICAL_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=200
        )
        
        return response.content if response.success else ""
    
    def generate_ddi_recommendation(self,
                                    drug1: str,
                                    drug2: str,
                                    severity: str,
                                    description: str,
                                    confidence: float = None) -> str:
        """
        Generate specific recommendation for a single DDI based on network description
        
        Args:
            drug1: First drug name
            drug2: Second drug name  
            severity: Severity label from DDI network
            description: Interaction description from DDI network
            confidence: Confidence score if available
            
        Returns:
            Specific clinical recommendation
        """
        severity_context = {
            'Contraindicated interaction': 'This combination should generally be avoided. ',
            'Major interaction': 'This is a clinically significant interaction requiring intervention. ',
            'Moderate interaction': 'This interaction may require monitoring or adjustment. ',
            'Minor interaction': 'This interaction has limited clinical significance. '
        }
        
        prompt = f"""Based on this DDI network entry, provide a specific clinical recommendation:

DRUG INTERACTION:
- Drug 1: {drug1}
- Drug 2: {drug2}
- Severity: {severity}
- Confidence: {confidence if confidence else 'Not specified'}

DDI NETWORK DESCRIPTION:
"{description}"

{severity_context.get(severity, '')}

Provide:
1. What the interaction means clinically (1-2 sentences)
2. Specific action to take (continue/modify/stop/monitor)
3. If monitoring needed, what to monitor

Keep response under 100 words."""

        response = self.generate(
            prompt=prompt,
            system_prompt=self.CLINICAL_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=300
        )
        
        return response.content if response.success else ""
    
    def generate_polypharmacy_summary(self,
                                      interactions: List[Dict],
                                      risk_assessment: Dict,
                                      drugs: List[str],
                                      alternatives: Dict = None) -> str:
        """
        Generate comprehensive polypharmacy summary using DDI network data
        
        Args:
            interactions: All interactions from DDI network
            risk_assessment: Risk scores
            drugs: List of patient's drugs
            alternatives: Suggested alternatives if available
            
        Returns:
            Comprehensive clinical summary
        """
        # Extract key DDI descriptions
        critical_ddis = [i for i in interactions 
                        if i.get('severity_label') == 'Contraindicated interaction']
        major_ddis = [i for i in interactions 
                     if i.get('severity_label') == 'Major interaction']
        
        prompt = f"""Generate a comprehensive polypharmacy assessment based on DDI network analysis.

PATIENT MEDICATIONS: {', '.join(drugs)}

RISK SUMMARY:
- Overall Risk: {risk_assessment.get('risk_level', 'Unknown')} ({risk_assessment.get('overall_score', 0)}/100)
- Contraindicated Pairs: {len(critical_ddis)}
- Major Interactions: {len(major_ddis)}
- Total DDIs Found: {len(interactions)}

CRITICAL INTERACTIONS FROM DDI NETWORK:
"""
        for i, ddi in enumerate(critical_ddis[:3], 1):
            desc = ddi.get('description') or ddi.get('interaction_description', '')
            prompt += f"{i}. {ddi.get('drug_1', '?')} + {ddi.get('drug_2', '?')}: {desc}\n"
        
        if major_ddis:
            prompt += "\nMAJOR INTERACTIONS:\n"
            for i, ddi in enumerate(major_ddis[:3], 1):
                desc = ddi.get('description') or ddi.get('interaction_description', '')
                prompt += f"{i}. {ddi.get('drug_1', '?')} + {ddi.get('drug_2', '?')}: {desc}\n"
        
        if alternatives:
            prompt += "\nSUGGESTED ALTERNATIVES:\n"
            for drug, alt in list(alternatives.items())[:3]:
                if isinstance(alt, dict):
                    prompt += f\"- {drug} -> {alt.get('drug_name', 'Unknown')} (safety: {alt.get('safety_score', 'N/A')})\\n\"
                else:
                    prompt += f\"- {drug} -> {alt}\\n\"
        
        prompt += """
Provide a structured clinical summary with:
1. IMMEDIATE CONCERNS (based on contraindicated/major DDIs)
2. RECOMMENDED ACTIONS (prioritized list)
3. MONITORING PLAN
4. PATIENT COUNSELING POINTS

Reference the specific DDI descriptions in your recommendations."""

        response = self.generate(
            prompt=prompt,
            system_prompt=self.CLINICAL_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.content if response.success else ""
    
    def _build_clinical_prompt(self, 
                               interactions: List[Dict],
                               risk_assessment: Dict,
                               drugs: List[str]) -> str:
        """Build prompt for clinical explanation using DDI network severity descriptions"""
        
        # Group interactions by severity with full descriptions
        severity_order = ['Contraindicated interaction', 'Major interaction', 
                         'Moderate interaction', 'Minor interaction']
        severity_groups = {sev: [] for sev in severity_order}
        
        for inter in interactions:
            sev = inter.get('severity_label', 'Unknown')
            if sev in severity_groups:
                # Get the full interaction description from DDI network
                description = inter.get('description') or inter.get('interaction_description', '')
                drug1 = inter.get('drug_1') or inter.get('drug_name_1', '?')
                drug2 = inter.get('drug_2') or inter.get('drug_name_2', '?')
                confidence = inter.get('severity_confidence', 'N/A')
                
                severity_groups[sev].append({
                    'drug_pair': f"{drug1} + {drug2}",
                    'description': description,
                    'confidence': confidence
                })
        
        prompt = f"""Analyze these drug-drug interactions from the DDI network for a patient taking: {', '.join(drugs)}

OVERALL RISK ASSESSMENT:
- Risk Level: {risk_assessment.get('risk_level', 'Unknown')}
- Risk Score: {risk_assessment.get('overall_score', 0)}/100
- Maximum Severity: {risk_assessment.get('max_severity', 'Unknown')}
- Total Interactions Found: {len(interactions)}

DETAILED DDI NETWORK INTERACTIONS:
"""
        
        for severity in severity_order:
            if severity_groups[severity]:
                prompt += f"\n{'='*50}\n{severity.upper()} ({len(severity_groups[severity])} found):\n{'='*50}\n"
                for i, inter in enumerate(severity_groups[severity][:5], 1):  # Top 5 per severity
                    prompt += f"""
{i}. {inter['drug_pair']}
   Description: {inter['description']}
   Confidence: {inter['confidence']}
"""
                if len(severity_groups[severity]) > 5:
                    prompt += f"\n   ... and {len(severity_groups[severity]) - 5} more {severity.lower()}s\n"
        
        prompt += """
Based on the DDI network data above, provide:

1. CLINICAL SUMMARY
   - Most critical interaction(s) requiring immediate attention
   - Mechanism of interaction (based on descriptions)

2. MONITORING PARAMETERS
   - Specific labs or vitals to monitor
   - Signs/symptoms to watch for

3. PRESCRIBER RECOMMENDATIONS
   - Whether to continue, modify, or discontinue combinations
   - Timing adjustments if applicable
   - Alternative therapy suggestions

4. PATIENT SAFETY CONCERNS
   - Time-sensitive warnings
   - When to seek emergency care

Be specific and reference the actual DDI descriptions provided."""

        return prompt
    
    def _build_patient_prompt(self,
                              interactions: List[Dict],
                              risk_assessment: Dict,
                              drugs: List[str]) -> str:
        """Build prompt for patient explanation"""
        
        risk_level = risk_assessment.get('risk_level', 'Unknown')
        num_interactions = len(interactions)
        
        # Identify most serious interactions
        serious = [i for i in interactions 
                   if i.get('severity_label') in ['Contraindicated interaction', 'Major interaction']]
        
        prompt = f"""A patient is taking these medications: {', '.join(drugs)}

Our analysis found {num_interactions} potential interactions.
Risk level: {risk_level}
Serious interactions: {len(serious)}

Please write a brief, reassuring message to the patient that:
1. Explains what this means in simple terms
2. Tells them what to watch for
3. Encourages them to talk to their doctor/pharmacist
4. Does NOT list specific drug combinations (keep it general)

Keep it under 150 words and use a warm, supportive tone."""

        return prompt


def get_llm_client(prefer_local: bool = True) -> Optional[BaseLLMClient]:
    """
    Factory function to get the best available LLM client
    
    Args:
        prefer_local: Prefer local Ollama over cloud APIs
        
    Returns:
        LLM client or None if unavailable
    """
    if prefer_local:
        client = BioMistralClient()
        if client.is_available():
            return client
        else:
            print("Ollama not running. Start with: ollama serve")
            print("   Then pull a model: ollama pull biomistral")
            return None
    
    return None
