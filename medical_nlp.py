"""
Medical NLP Pipeline for extracting medical details from transcripts.
"""

import json
import re
from typing import Dict, List, Set
import spacy
from spacy.matcher import Matcher

from utils import (
    extract_patient_name, extract_medical_phrases, extract_numbers_with_context,
    clean_text, extract_key_phrases, separate_speakers,
    SYMPTOM_KEYWORDS, TREATMENT_KEYWORDS, DIAGNOSIS_KEYWORDS, PROGNOSIS_KEYWORDS
)

class MedicalNLPPipeline:
    """Pipeline for medical NLP summarization."""
    
    def __init__(self):
        """Initialize the NLP pipeline."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found.")
            print("Please run: python -m spacy download en_core_web_sm")
            self.nlp = None
        except Exception as e:
            print(f"Warning: Could not load spaCy model: {e}")
            self.nlp = None
        
        self._setup_matchers()
    
    def _setup_matchers(self):
        """Setup spaCy matchers for medical entities."""
        if self.nlp is None:
            return
        
        self.matcher = Matcher(self.nlp.vocab)
        
        # Pattern for symptoms
        symptom_patterns = [
            [{"LOWER": {"IN": ["pain", "ache", "discomfort", "stiffness"]}}],
            [{"LOWER": {"IN": ["neck", "back", "head", "shoulder"]}}, {"LOWER": "pain"}],
            [{"LOWER": "trouble"}, {"LOWER": {"IN": ["sleeping", "concentrating"]}}],
        ]
        
        # Pattern for treatments
        treatment_patterns = [
            [{"LOWER": {"IN": ["physiotherapy", "therapy", "treatment"]}}],
            [{"LOWER": {"IN": ["painkiller", "medication", "medicine"]}}],
            [{"IS_DIGIT": True}, {"LOWER": {"IN": ["session", "sessions"]}}],
        ]
        
        # Pattern for diagnoses
        diagnosis_patterns = [
            [{"LOWER": {"IN": ["whiplash", "injury", "strain", "sprain"]}}],
            [{"LOWER": "diagnosed"}, {"LOWER": "with"}],
        ]
        
        self.matcher.add("SYMPTOM", symptom_patterns)
        self.matcher.add("TREATMENT", treatment_patterns)
        self.matcher.add("DIAGNOSIS", diagnosis_patterns)
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text."""
        symptoms = set()
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            
            for match_id, start, end in matches:
                if self.nlp.vocab.strings[match_id] == "SYMPTOM":
                    symptoms.add(doc[start:end].text)
        
        # Pattern-based extraction
        symptom_phrases = extract_medical_phrases(text, SYMPTOM_KEYWORDS)
        
        # Extract specific symptom mentions
        symptom_patterns = [
            r'(neck\s+pain|back\s+pain|head\s+pain|headache)',
            r'(discomfort|stiffness|tenderness)',
            r'(trouble\s+(?:sleeping|concentrating))',
            r'(hit\s+(?:my|the)\s+(?:head|neck|back))',
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                symptoms.add(match.group(0))
        
        # Extract from symptom phrases
        for phrase in symptom_phrases:
            # Look for pain-related mentions
            if any(kw in phrase.lower() for kw in ['pain', 'ache', 'discomfort']):
                # Extract the body part + pain
                body_parts = ['neck', 'back', 'head', 'shoulder', 'arm', 'leg']
                for part in body_parts:
                    if part in phrase.lower() and 'pain' in phrase.lower():
                        symptoms.add(f"{part.capitalize()} pain")
        
        return list(symptoms)
    
    def extract_diagnosis(self, text: str) -> str:
        """Extract diagnosis from text."""
        diagnosis = None
        
        # Look for explicit diagnosis mentions
        diagnosis_patterns = [
            r'(whiplash\s+injury)',
            r'(diagnosed\s+with\s+([^\.]+))',
            r'(it\s+was\s+(?:a|an)\s+([^\.]+))',
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                diagnosis = match.group(1) if len(match.groups()) == 1 else match.group(2)
                diagnosis = diagnosis.strip()
                break
        
        # If no explicit diagnosis, look for medical conditions
        if not diagnosis:
            condition_keywords = ['whiplash', 'injury', 'strain', 'sprain', 'fracture']
            for keyword in condition_keywords:
                if keyword in text.lower():
                    # Get context around the keyword
                    sentences = text.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            # Extract a short phrase
                            words = sentence.split()
                            if keyword in [w.lower() for w in words]:
                                idx = [w.lower() for w in words].index(keyword)
                                start = max(0, idx - 2)
                                end = min(len(words), idx + 3)
                                diagnosis = ' '.join(words[start:end])
                                break
                    if diagnosis:
                        break
        
        return diagnosis or "Not specified"
    
    def extract_treatment(self, text: str) -> List[str]:
        """Extract treatment information."""
        treatments = set()
        
        # Extract numbers with context
        number_contexts = extract_numbers_with_context(text)
        for item in number_contexts:
            if item['type'] == 'sessions':
                treatments.add(f"{item['number']} {item['context']}")
            elif item['type'] == 'treatment':
                treatments.add(f"{item['number']} {item['context']}")
        
        # Look for treatment mentions
        treatment_patterns = [
            r'(\d+)\s*(?:sessions?\s+of\s+)?(?:physiotherapy|therapy)',
            r'(physiotherapy|therapy|treatment)',
            r'(painkiller|medication|medicine|analgesic)',
        ]
        
        for pattern in treatment_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    treatments.add(match.group(0))
        
        # Extract from treatment phrases
        treatment_phrases = extract_medical_phrases(text, TREATMENT_KEYWORDS)
        for phrase in treatment_phrases:
            # Look for specific treatments
            if 'physiotherapy' in phrase.lower():
                # Extract number if present
                num_match = re.search(r'(\d+)', phrase)
                if num_match:
                    treatments.add(f"{num_match.group(1)} physiotherapy sessions")
                else:
                    treatments.add("Physiotherapy")
            if any(kw in phrase.lower() for kw in ['painkiller', 'medication', 'medicine']):
                treatments.add("Painkillers")
        
        return list(treatments)
    
    def extract_prognosis(self, text: str) -> str:
        """Extract prognosis information."""
        prognosis = None
        
        # Look for prognosis keywords
        prognosis_patterns = [
            r'(full\s+recovery\s+(?:expected|within|in)\s+[^\.]+)',
            r'(recovery\s+(?:expected|within|in)\s+[^\.]+)',
            r'(prognosis[^\.]+)',
            r'(expect\s+[^\.]+recovery[^\.]+)',
            r'(no\s+(?:signs?|indication)\s+of\s+[^\.]+)',
        ]
        
        for pattern in prognosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                prognosis = match.group(0).strip()
                break
        
        # If no explicit prognosis, infer from context
        if not prognosis:
            if 'full recovery' in text.lower():
                # Extract time frame
                time_match = re.search(r'(full\s+recovery[^\.]*(?:\d+\s*(?:month|months|week|weeks))[^\.]*)', text, re.IGNORECASE)
                if time_match:
                    prognosis = time_match.group(0)
                else:
                    prognosis = "Full recovery expected"
            elif 'improving' in text.lower() or 'better' in text.lower():
                prognosis = "Condition improving"
        
        return prognosis or "Not specified"
    
    def extract_current_status(self, text: str) -> str:
        """Extract current status of the patient."""
        status = None
        
        # Look for current status indicators
        status_patterns = [
            r'(still\s+(?:experiencing|having|feeling)\s+[^\.]+)',
            r'(occasional\s+[^\.]+)',
            r'(not\s+constant[^\.]+)',
            r'(doing\s+better[^\.]+)',
            r'(feeling\s+[^\.]+now)',
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                status = match.group(0).strip()
                break
        
        # Extract from patient's recent statements
        if not status:
            # Look for phrases like "I still have", "I get occasional"
            patient_phrases = [
                r'(I\s+(?:still|get|have)\s+[^\.]+)',
                r'(It\'?s\s+(?:not|nothing)\s+[^\.]+)',
            ]
            
            for pattern in patient_phrases:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    status = match.group(0).strip()
                    break
        
        return status or "Not specified"
    
    def extract_medical_details(self, text: str) -> Dict[str, any]:
        """Extract all medical details from transcript."""
        text = clean_text(text)
        
        # Extract patient name
        patient_name = extract_patient_name(text)
        
        # Extract symptoms
        symptoms = self.extract_symptoms(text)
        
        # Extract diagnosis
        diagnosis = self.extract_diagnosis(text)
        
        # Extract treatment
        treatment = self.extract_treatment(text)
        
        # Extract current status
        current_status = self.extract_current_status(text)
        
        # Extract prognosis
        prognosis = self.extract_prognosis(text)
        
        # Extract keywords
        keywords = extract_key_phrases(text, max_phrases=10)
        
        return {
            "Patient_Name": patient_name,
            "Symptoms": symptoms,
            "Diagnosis": diagnosis,
            "Treatment": treatment,
            "Current_Status": current_status,
            "Prognosis": prognosis,
            "Keywords": keywords
        }
    
    def generate_summary_json(self, text: str) -> str:
        """Generate JSON summary of medical details."""
        details = self.extract_medical_details(text)
        return json.dumps(details, indent=2)

