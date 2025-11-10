"""
SOAP Note Generator for converting medical transcripts to structured SOAP notes.
"""

import json
import re
from typing import Dict, List
from medical_nlp import MedicalNLPPipeline
from utils import separate_speakers, clean_text

class SOAPNoteGenerator:
    """Generator for SOAP notes from medical transcripts."""
    
    def __init__(self):
        """Initialize SOAP note generator."""
        self.nlp_pipeline = MedicalNLPPipeline()
    
    def extract_subjective(self, text: str, medical_summary: Dict = None) -> Dict[str, str]:
        """Extract Subjective section (patient's reported symptoms and history)."""
        if medical_summary is None:
            medical_summary = self.nlp_pipeline.extract_medical_details(text)
        
        # Extract patient dialogue
        speakers = separate_speakers(text)
        patient_text = speakers.get('patient', text)
        
        # Chief Complaint
        chief_complaint = self._extract_chief_complaint(patient_text, medical_summary)
        
        # History of Present Illness
        hpi = self._extract_hpi(text, medical_summary)
        
        return {
            "Chief_Complaint": chief_complaint,
            "History_of_Present_Illness": hpi
        }
    
    def _extract_chief_complaint(self, patient_text: str, medical_summary: Dict) -> str:
        """Extract chief complaint from patient dialogue."""
        # Use symptoms from medical summary
        symptoms = medical_summary.get('Symptoms', [])
        
        if symptoms:
            # Combine symptoms into chief complaint
            if len(symptoms) == 1:
                return symptoms[0]
            elif len(symptoms) <= 3:
                return " and ".join(symptoms)
            else:
                return ", ".join(symptoms[:3]) + " and others"
        
        # Fallback: extract from first patient statement
        sentences = patient_text.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            # Look for pain or symptom mentions
            if any(word in first_sentence.lower() for word in ['pain', 'ache', 'discomfort', 'problem']):
                return first_sentence[:100]  # Limit length
        
        return "Not specified"
    
    def _extract_hpi(self, text: str, medical_summary: Dict) -> str:
        """Extract History of Present Illness."""
        hpi_parts = []
        
        # Extract timeline information
        timeline_patterns = [
            r'(September\s+\d+[^\.]*)',
            r'(\d+\s+weeks?\s+ago[^\.]*)',
            r'(last\s+[^\.]+)',
        ]
        
        timeline = None
        for pattern in timeline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                timeline = match.group(0)
                break
        
        # Build HPI from medical summary
        diagnosis = medical_summary.get('Diagnosis', '')
        symptoms = medical_summary.get('Symptoms', [])
        treatment = medical_summary.get('Treatment', [])
        current_status = medical_summary.get('Current_Status', '')
        
        # Construct HPI
        if timeline:
            hpi_parts.append(f"Patient reports {timeline.lower()}.")
        
        if diagnosis:
            hpi_parts.append(f"Diagnosed with {diagnosis.lower()}.")
        
        if symptoms:
            symptom_text = ", ".join(symptoms).lower()
            hpi_parts.append(f"Presented with {symptom_text}.")
        
        if treatment:
            treatment_text = ", ".join(treatment).lower()
            hpi_parts.append(f"Received {treatment_text}.")
        
        if current_status and current_status != "Not specified":
            hpi_parts.append(f"Current status: {current_status.lower()}.")
        
        # If no structured data, extract from text
        if not hpi_parts:
            # Look for patient's narrative
            speakers = separate_speakers(text)
            patient_text = speakers.get('patient', '')
            
            if patient_text:
                # Take first few sentences from patient
                sentences = patient_text.split('.')[:3]
                hpi_parts = [s.strip() + '.' for s in sentences if s.strip()]
        
        return " ".join(hpi_parts) if hpi_parts else "Not specified"
    
    def extract_objective(self, text: str) -> Dict[str, str]:
        """Extract Objective section (physical exam findings, observations)."""
        objective = {
            "Physical_Exam": "Not documented",
            "Observations": "Not documented"
        }
        
        # Look for physical examination mentions
        exam_patterns = [
            r'(physical\s+examination[^\.]+)',
            r'(exam[^\.]+)',
            r'(checked[^\.]+)',
            r'(range\s+of\s+movement[^\.]+)',
            r'(tenderness[^\.]+)',
        ]
        
        exam_findings = []
        for pattern in exam_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                exam_findings.append(match.group(0))
        
        if exam_findings:
            objective["Physical_Exam"] = ". ".join(exam_findings)
        
        # Look for specific exam findings
        specific_findings = []
        
        # Range of motion
        if 'range of movement' in text.lower() or 'range of motion' in text.lower():
            rom_match = re.search(r'(full\s+range\s+of\s+movement[^\.]+)', text, re.IGNORECASE)
            if rom_match:
                specific_findings.append(rom_match.group(0))
        
        # Tenderness
        if 'tenderness' in text.lower():
            tender_match = re.search(r'(no\s+tenderness[^\.]+|tenderness[^\.]+)', text, re.IGNORECASE)
            if tender_match:
                specific_findings.append(tender_match.group(0))
        
        # Muscle and spine condition
        if 'muscle' in text.lower() or 'spine' in text.lower():
            muscle_match = re.search(r'(muscle[^\.]+spine[^\.]+|spine[^\.]+muscle[^\.]+)', text, re.IGNORECASE)
            if muscle_match:
                specific_findings.append(muscle_match.group(0))
        
        if specific_findings:
            objective["Physical_Exam"] = ". ".join(specific_findings)
        
        # Observations
        observation_keywords = ['appears', 'normal', 'good condition', 'healthy', 'gait']
        observations = []
        
        for keyword in observation_keywords:
            pattern = f'([^\.]*{keyword}[^\.]*)'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.group(0)) < 200:  # Reasonable length
                    observations.append(match.group(0))
        
        if observations:
            objective["Observations"] = ". ".join(observations[:3])  # Limit to 3
        
        return objective
    
    def extract_assessment(self, text: str, medical_summary: Dict = None) -> Dict[str, str]:
        """Extract Assessment section (diagnosis and severity)."""
        if medical_summary is None:
            medical_summary = self.nlp_pipeline.extract_medical_details(text)
        
        diagnosis = medical_summary.get('Diagnosis', 'Not specified')
        
        # Determine severity
        severity = self._determine_severity(text, medical_summary)
        
        return {
            "Diagnosis": diagnosis,
            "Severity": severity
        }
    
    def _determine_severity(self, text: str, medical_summary: Dict) -> str:
        """Determine severity of condition."""
        text_lower = text.lower()
        current_status = medical_summary.get('Current_Status', '').lower()
        prognosis = medical_summary.get('Prognosis', '').lower()
        
        # Check for severity indicators
        if any(word in text_lower for word in ['severe', 'serious', 'critical', 'acute']):
            return "Severe"
        elif any(word in text_lower for word in ['mild', 'minor', 'slight', 'occasional']):
            return "Mild"
        elif any(word in current_status for word in ['improving', 'better', 'recovering']):
            return "Mild, improving"
        elif any(word in prognosis for word in ['full recovery', 'no long-term']):
            return "Mild, improving"
        elif any(word in text_lower for word in ['moderate', 'moderate']):
            return "Moderate"
        else:
            return "Mild to moderate"
    
    def extract_plan(self, text: str, medical_summary: Dict = None) -> Dict[str, str]:
        """Extract Plan section (treatment and follow-up)."""
        if medical_summary is None:
            medical_summary = self.nlp_pipeline.extract_medical_details(text)
        
        # Extract treatment recommendations
        treatment = medical_summary.get('Treatment', [])
        treatment_text = ", ".join(treatment).lower() if treatment else "Continue current management"
        
        # Extract follow-up instructions
        follow_up = self._extract_follow_up(text, medical_summary)
        
        return {
            "Treatment": treatment_text,
            "Follow-Up": follow_up
        }
    
    def _extract_follow_up(self, text: str, medical_summary: Dict) -> str:
        """Extract follow-up instructions."""
        follow_up_patterns = [
            r'(follow[^\.]*up[^\.]+)',
            r'(return\s+if[^\.]+)',
            r'(come\s+back[^\.]+)',
            r'(schedule[^\.]+)',
        ]
        
        follow_ups = []
        for pattern in follow_up_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                follow_ups.append(match.group(0))
        
        if follow_ups:
            return ". ".join(follow_ups)
        
        # Default follow-up based on prognosis
        prognosis = medical_summary.get('Prognosis', '')
        if 'full recovery' in prognosis.lower():
            return "Patient to return if symptoms worsen or persist beyond expected recovery period."
        else:
            return "Patient to return for follow-up as needed or if symptoms worsen."
    
    def generate_soap_note(self, text: str, medical_summary: Dict = None) -> Dict[str, Dict]:
        """Generate complete SOAP note from transcript."""
        text = clean_text(text)
        
        if medical_summary is None:
            medical_summary = self.nlp_pipeline.extract_medical_details(text)
        
        soap_note = {
            "Subjective": self.extract_subjective(text, medical_summary),
            "Objective": self.extract_objective(text),
            "Assessment": self.extract_assessment(text, medical_summary),
            "Plan": self.extract_plan(text, medical_summary)
        }
        
        return soap_note
    
    def generate_soap_json(self, text: str) -> str:
        """Generate JSON output for SOAP note."""
        soap_note = self.generate_soap_note(text)
        return json.dumps(soap_note, indent=2)

