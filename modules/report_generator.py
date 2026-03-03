"""
Explanation Module - Generates human-readable reports and explanations using LLM (BioMistral-7B)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .base_module import BaseModule, Result, PipelineStatus


class ReportGenerator(BaseModule):
    """
    Explanation Module
    
    Responsible for:
    - Generating human-readable clinical reports
    - Creating patient-friendly explanations using BioMistral-7B
    - Producing structured recommendations
    
    Input: All analysis results
    Output: Human-readable reports
    
    Components:
    - BioMistral-7B for clinical text
    - Template fallback if LLM unavailable
    - Structured JSON output
    """
    
    RISK_LABELS = {
        'CRITICAL': '[CRITICAL]',
        'HIGH': '[HIGH]',
        'MODERATE': '[MODERATE]',
        'LOW': '[LOW]',
        'NO_RISK': '',
        'NO_INTERACTIONS': ''
    }
    
    SEVERITY_ICONS = {
        'Contraindicated interaction': '[STOP]',
        'Major interaction': '',
        'Moderate interaction': '[!]',
        'Minor interaction': ''
    }
    
    def __init__(self, use_llm: bool = True, llm_client: Any = None):
        super().__init__(
            name="ReportGenerator",
            description="Generates human-readable reports and explanations"
        )
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.llm_available = False
        
    def initialize(self, use_llm: bool = True) -> bool:
        """Initialize the explanation module with optional LLM"""
        print(f"[{self.name}] Initializing...")
        
        self.use_llm = use_llm
        
        if self.use_llm and self.llm_client is None:
            try:
                from .llm_client import BioMistralClient
                self.llm_client = BioMistralClient()
                self.llm_available = self.llm_client.is_available()
                
                if self.llm_available:
                    print(f"[{self.name}] BioMistral-7B connected via Ollama")
                else:
                    print(f"[{self.name}] LLM not available, using template fallback")
                    print(f"   To enable: ollama serve && ollama pull biomistral")
            except ImportError:
                print(f"[{self.name}] LLM client not found, using templates")
                self.llm_available = False
        elif self.llm_client is not None:
            self.llm_available = self.llm_client.is_available()
        
        self._initialized = True
        mode = "BioMistral-7B" if self.llm_available else "template"
        print(f"[{self.name}] Ready - Using {mode}-based generation")
        return True
    
    def validate_input(self, input_data: Dict[str, Any]) -> tuple:
        """Validate input contains analysis results"""
        # Accept any input - we'll work with whatever we have
        return True, ""
    
    def _generate_header(self, data: Dict) -> str:
        """Generate report header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
======================================================================
     POLYPHARMACY RISK ASSESSMENT REPORT
     AI-based Drug-Drug Interaction Analysis
----------------------------------------------------------------------
  Generated: {timestamp}
======================================================================
"""
    
    def _generate_drug_summary(self, validation: Dict) -> str:
        """Generate drug validation summary"""
        lines = ["\nMEDICATION LIST ANALYSIS", "-" * 50]
        
        valid_drugs = validation.get('validated', [])
        unrecognized = validation.get('unrecognized', [])
        
        if valid_drugs:
            lines.append(f"\nRecognized Medications ({len(valid_drugs)}):")
            for drug in valid_drugs:
                status = "[OK]" if drug.get('status') == 'valid' else "~"
                name = drug.get('input_name', drug.get('normalized_name', 'Unknown'))
                info = drug.get('info', {})
                cv = "💓" if info.get('is_cardiovascular') else ""
                lines.append(f"   {status} {name} {cv}")
        
        if unrecognized:
            lines.append(f"\n❓ Unrecognized Medications ({len(unrecognized)}):")
            for drug in unrecognized:
                lines.append(f"   ✗ {drug}")
        
        return "\n".join(lines)
    
    def _generate_risk_summary(self, risk: Dict) -> str:
        """Generate risk assessment summary"""
        risk_level = risk.get('risk_level', 'UNKNOWN')
        label = self.RISK_LABELS.get(risk_level, '[?]')
        score = risk.get('overall_score', 0)
        
        lines = [
            "\nOVERALL RISK ASSESSMENT",
            "-" * 50,
            f"\n   Risk Level:  {label} {risk_level}",
            f"   Risk Score:  {score}/100",
            f"   Maximum Severity: {risk.get('max_severity', 'N/A')}",
            f"   Total Interactions: {risk.get('interaction_count', 0)}"
        ]
        
        # Risk bar visualization
        filled = int(score / 10)
        bar = "#" * filled + "." * (10 - filled)
        lines.append(f"\n   Risk Scale: [{bar}] {score}%")
        
        return "\n".join(lines)
    
    def _generate_interaction_details(self, interactions: List[Dict]) -> str:
        """Generate detailed interaction list"""
        if not interactions:
            return "\nNO DRUG-DRUG INTERACTIONS DETECTED"
        
        lines = [
            "\nDRUG-DRUG INTERACTIONS DETECTED",
            "-" * 50
        ]
        
        # Group by severity
        severity_order = ['Contraindicated interaction', 'Major interaction', 
                         'Moderate interaction', 'Minor interaction']
        
        by_severity = {}
        for inter in interactions:
            sev = inter.get('severity_label', 'Unknown')
            if sev not in by_severity:
                by_severity[sev] = []
            by_severity[sev].append(inter)
        
        for severity in severity_order:
            if severity in by_severity:
                icon = self.SEVERITY_ICONS.get(severity, '*')
                lines.append(f"\n{icon} {severity.upper()} ({len(by_severity[severity])})")
                lines.append("   " + "-" * 45)
                
                for inter in by_severity[severity][:5]:  # Show top 5 per severity
                    drug1 = inter.get('drug_1', inter.get('drug_name_1', '?'))
                    drug2 = inter.get('drug_2', inter.get('drug_name_2', '?'))
                    desc = inter.get('description', inter.get('interaction_description', ''))
                    
                    lines.append(f"\n   * {drug1} <-> {drug2}")
                    if desc:
                        # Truncate long descriptions
                        desc_short = desc[:200] + "..." if len(desc) > 200 else desc
                        lines.append(f"     {desc_short}")
                
                if len(by_severity[severity]) > 5:
                    lines.append(f"\n   ... and {len(by_severity[severity]) - 5} more")
        
        return "\n".join(lines)
    
    def _generate_alternatives_section(self, alternatives: Dict) -> str:
        """Generate alternatives recommendations section"""
        if not alternatives:
            return ""
        
        all_alts = alternatives.get('alternatives', {})
        best_alts = alternatives.get('best_alternatives', {})
        
        if not all_alts and not best_alts:
            return ""
        
        lines = [
            "\nRECOMMENDED ALTERNATIVES",
            "-" * 50
        ]
        
        for drug, best in best_alts.items():
            lines.append(f"\n   Replace: {drug}")
            lines.append(f"   With:    {best.get('drug_name', 'Unknown')}")
            safety_score = best.get('safety_score', 0)
            stars = int(safety_score / 20)
            lines.append(f"   Safety:  {'*' * stars}{'-' * (5 - stars)} ({safety_score}/100)")
            
            conflicts = best.get('conflicts_with_current', [])
            if conflicts:
                lines.append(f"   Note: May interact with {', '.join(conflicts[:3])}")
        
        no_alts = alternatives.get('drugs_without_alternatives', [])
        if no_alts:
            lines.append(f"\n   [!] No alternatives found for: {', '.join(no_alts)}")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, data: Dict) -> str:
        """Generate clinical recommendations"""
        risk = data.get('risk_assessment', {})
        risk_level = risk.get('risk_level', 'UNKNOWN')
        
        lines = [
            "\nCLINICAL RECOMMENDATIONS",
            "-" * 50
        ]
        
        if risk_level == 'CRITICAL':
            lines.extend([
                "\n   [CRITICAL] URGENT ACTIONS REQUIRED:",
                "   1. Immediately review contraindicated drug combinations",
                "   2. Consider hospitalization for monitoring during changes",
                "   3. Consult clinical pharmacist before any modifications",
                "   4. Document rationale if continuing high-risk therapy",
                "   5. Establish frequent monitoring schedule"
            ])
        elif risk_level == 'HIGH':
            lines.extend([
                "\n   [HIGH] HIGH PRIORITY ACTIONS:",
                "   1. Review major interactions with prescribing physician",
                "   2. Consider therapeutic alternatives",
                "   3. Implement enhanced monitoring protocols",
                "   4. Educate patient on warning signs",
                "   5. Schedule follow-up within 1-2 weeks"
            ])
        elif risk_level == 'MODERATE':
            lines.extend([
                "\n   [MODERATE] RECOMMENDED ACTIONS:",
                "   1. Review interactions at next appointment",
                "   2. Monitor for adverse effects",
                "   3. Consider timing adjustments for medications",
                "   4. Update medication list documentation",
                "   5. Schedule routine follow-up"
            ])
        else:
            lines.extend([
                "\n   [LOW] MAINTENANCE ACTIONS:",
                "   1. Continue current regimen with standard monitoring",
                "   2. Maintain updated medication records",
                "   3. Routine follow-up as scheduled",
                "   4. Patient education on general drug safety"
            ])
        
        return "\n".join(lines)
    
    def _generate_footer(self) -> str:
        """Generate report footer"""
        return """
----------------------------------------------------------------------
 DISCLAIMER: This AI-generated report is for informational purposes
    only and should not replace professional clinical judgment. Always
    consult with qualified healthcare providers before making changes
    to medication regimens.
    
    Generated by: AI-based Polypharmacy Risk-aware Recommender System
----------------------------------------------------------------------
"""
    
    def generate_clinical_report(self, data: Dict) -> str:
        """Generate full clinical report with optional LLM enhancement"""
        sections = []
        
        # Header
        sections.append(self._generate_header(data))
        
        # Drug summary
        if 'validation' in data:
            sections.append(self._generate_drug_summary(data['validation']))
        
        # Risk assessment
        if 'risk_assessment' in data:
            sections.append(self._generate_risk_summary(data['risk_assessment']))
        
        # Interaction details
        interactions = data.get('analyzed_interactions', data.get('interactions', []))
        sections.append(self._generate_interaction_details(interactions))
        
        # LLM-enhanced clinical analysis (if available)
        if self.llm_available and self.llm_client and interactions:
            sections.append(self._generate_llm_clinical_analysis(data))
        
        # Alternatives
        if 'alternatives' in data or 'best_alternatives' in data:
            sections.append(self._generate_alternatives_section(data))
        
        # Recommendations
        sections.append(self._generate_recommendations(data))
        
        # Footer
        sections.append(self._generate_footer())
        
        return "\n".join(sections)
    
    def _generate_llm_clinical_analysis(self, data: Dict) -> str:
        """Generate LLM-powered clinical analysis using DDI network severity descriptions"""
        try:
            interactions = data.get('analyzed_interactions', data.get('interactions', []))
            risk_assessment = data.get('risk_assessment', {})
            validation = data.get('validation', {})
            drugs = [d.get('input_name', '') for d in validation.get('validated', [])]
            
            if not drugs:
                drugs = list(set(
                    [i.get('drug_1', '') for i in interactions] + 
                    [i.get('drug_2', '') for i in interactions]
                ))
            
            # Get alternatives if available
            alternatives = data.get('best_alternatives', {})
            
            # Use the comprehensive DDI-aware summary
            llm_analysis = self.llm_client.generate_polypharmacy_summary(
                interactions=interactions,
                risk_assessment=risk_assessment,
                drugs=drugs,
                alternatives=alternatives
            )
            
            if llm_analysis and not llm_analysis.startswith("[LLM unavailable"):
                return f"""
AI CLINICAL ANALYSIS (BioMistral-7B + DDI Network)
--------------------------------------------------

{llm_analysis}
"""
        except Exception as e:
            pass
        
        return ""

    def generate_patient_summary(self, data: Dict) -> str:
        """Generate patient-friendly summary with optional LLM enhancement"""
        risk = data.get('risk_assessment', {})
        risk_level = risk.get('risk_level', 'UNKNOWN')
        interactions = data.get('analyzed_interactions', data.get('interactions', []))
        
        # Simplified language mapping
        risk_simple = {
            'CRITICAL': 'Very High',
            'HIGH': 'High',
            'MODERATE': 'Medium',
            'LOW': 'Low',
            'NO_RISK': 'Very Low',
            'NO_INTERACTIONS': 'Very Low'
        }
        
        summary = f"""
═══════════════════════════════════════════════════════
        YOUR MEDICATION SAFETY SUMMARY
═══════════════════════════════════════════════════════

Dear Patient,

Based on our analysis of your medications, here's what 
you should know:

Overall Safety Level: {risk_simple.get(risk_level, 'Unknown')}

What This Means:
"""
        
        # Try LLM-generated patient explanation first
        if self.llm_available and self.llm_client and interactions:
            try:
                validation = data.get('validation', {})
                drugs = [d.get('input_name', '') for d in validation.get('validated', [])]
                
                llm_explanation = self.llm_client.generate_patient_explanation(
                    interactions=interactions,
                    risk_assessment=risk,
                    drugs=drugs
                )
                
                if llm_explanation and not llm_explanation.startswith("[LLM unavailable"):
                    summary += f"""
{llm_explanation}
"""
                    summary += """
═══════════════════════════════════════════════════════
Questions? Talk to your pharmacist or doctor.
═══════════════════════════════════════════════════════
"""
                    return summary
            except:
                pass
        
        # Fallback to template-based summary
        if risk_level in ['CRITICAL', 'HIGH']:
            summary += """
   Your medications have some combinations that need
   attention. Please talk to your doctor or pharmacist
   before taking them together.
   
   IMPORTANT: Do not stop any medication without
   talking to your healthcare provider first.
"""
        elif risk_level == 'MODERATE':
            summary += """
   Your medications have some minor interactions that
   your healthcare team should know about. They may
   want to monitor you more closely or adjust timing.
"""
        else:
            summary += """
   Your medications appear to work well together!
   Continue taking them as prescribed.
"""
        
        if interactions:
            summary += f"""
Interactions Found: {len(interactions)}

   Please bring this report to your next doctor's
   visit to discuss any concerns.
"""
        
        summary += """
═══════════════════════════════════════════════════════
Questions? Talk to your pharmacist or doctor.
═══════════════════════════════════════════════════════
"""
        return summary
    
    def generate_structured_json(self, data: Dict) -> Dict:
        """Generate structured JSON output for integration"""
        risk = data.get('risk_assessment', {})
        interactions = data.get('analyzed_interactions', data.get('interactions', []))
        validation = data.get('validation', {})
        alternatives = data.get('alternatives', {})
        
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0',
                'system': 'AI Polypharmacy Risk-aware Recommender'
            },
            'patient_summary': {
                'medications_analyzed': validation.get('valid_count', 0),
                'unrecognized_medications': validation.get('unrecognized_count', 0),
                'risk_level': risk.get('risk_level', 'UNKNOWN'),
                'risk_score': risk.get('overall_score', 0),
                'interaction_count': len(interactions)
            },
            'interactions': [
                {
                    'drugs': [inter.get('drug_1', ''), inter.get('drug_2', '')],
                    'severity': inter.get('severity_label', 'Unknown'),
                    'description': inter.get('description', inter.get('interaction_description', ''))
                }
                for inter in interactions
            ],
            'recommendations': {
                'alternatives': {
                    drug: {
                        'suggested': [alt.get('drug_name') for alt in alts[:3]],
                        'safety_scores': [alt.get('safety_score') for alt in alts[:3]]
                    }
                    for drug, alts in (alternatives.get('alternatives', {}) or {}).items()
                },
                'actions': self._get_recommended_actions(risk.get('risk_level', 'UNKNOWN'))
            }
        }
    
    def _get_recommended_actions(self, risk_level: str) -> List[str]:
        """Get list of recommended actions based on risk level"""
        actions = {
            'CRITICAL': [
                "Seek immediate medical review",
                "Do not start any new medications",
                "Contact prescriber urgently"
            ],
            'HIGH': [
                "Schedule appointment within 1 week",
                "Report any new symptoms immediately",
                "Review with pharmacist"
            ],
            'MODERATE': [
                "Discuss at next appointment",
                "Monitor for side effects",
                "Keep medication diary"
            ],
            'LOW': [
                "Continue current regimen",
                "Routine follow-up"
            ]
        }
        return actions.get(risk_level, ["Consult healthcare provider"])
    
    def process(self, input_data: Dict[str, Any]) -> Result:
        """
        Main processing: Generate all report formats
        """
        # Aggregate data from all sources
        aggregated_data = {}
        
        # From InteractionDetector
        if 'validation' in input_data:
            aggregated_data['validation'] = input_data['validation']
        if 'interactions' in input_data:
            aggregated_data['interactions'] = input_data['interactions']
        
        # From SeverityClassifier
        if 'analyzed_interactions' in input_data:
            aggregated_data['analyzed_interactions'] = input_data['analyzed_interactions']
        if 'risk_assessment' in input_data:
            aggregated_data['risk_assessment'] = input_data['risk_assessment']
        
        # From AlternativeFinder
        if 'alternatives' in input_data:
            aggregated_data['alternatives'] = input_data['alternatives']
        if 'best_alternatives' in input_data:
            aggregated_data['best_alternatives'] = input_data['best_alternatives']
        
        # Generate all report formats
        clinical_report = self.generate_clinical_report(aggregated_data)
        patient_summary = self.generate_patient_summary(aggregated_data)
        structured_json = self.generate_structured_json(aggregated_data)
        
        return Result(
            module_name=self.name,
            status=PipelineStatus.SUCCESS,
            data={
                'clinical_report': clinical_report,
                'patient_summary': patient_summary,
                'structured_output': structured_json
            },
            metadata={
                'report_types': ['clinical', 'patient', 'structured'],
                'generation_method': 'llm' if self.use_llm else 'template'
            }
        )
