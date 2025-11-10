"""
Utility functions for text processing and medical entity extraction.
"""

import re
from typing import List, Dict, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Medical keywords and patterns
SYMPTOM_KEYWORDS = [
    'pain', 'ache', 'discomfort', 'stiffness', 'sore', 'tender', 'numbness',
    'tingling', 'headache', 'dizziness', 'nausea', 'fatigue', 'weakness',
    'swelling', 'inflammation', 'difficulty', 'trouble', 'problem'
]

TREATMENT_KEYWORDS = [
    'treatment', 'therapy', 'physiotherapy', 'medication', 'medicine',
    'painkiller', 'analgesic', 'surgery', 'operation', 'injection',
    'session', 'appointment', 'prescription', 'exercise', 'rehabilitation'
]

DIAGNOSIS_KEYWORDS = [
    'diagnosis', 'diagnosed', 'condition', 'injury', 'disease', 'disorder',
    'syndrome', 'fracture', 'strain', 'sprain', 'whiplash', 'concussion'
]

PROGNOSIS_KEYWORDS = [
    'recovery', 'prognosis', 'outcome', 'expect', 'forecast', 'improve',
    'heal', 'resolve', 'chronic', 'acute', 'long-term', 'short-term',
    'full recovery', 'partial recovery', 'recurrence'
]

TIME_PATTERNS = [
    r'\d+\s*(week|weeks|month|months|year|years|day|days)',
    r'\d+\s*(session|sessions)',
    r'\d+\s*(time|times)'
]

def extract_patient_name(text: str) -> str:
    """Extract patient name from conversation."""
    # Look for patterns like "Ms. Jones", "Mr. Smith", "Patient:"
    patterns = [
        r'(?:Ms\.|Mr\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+)',
        r'Patient[:\s]+([A-Z][a-z]+)',
        r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # Full name pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1) if '(' not in match.group(0) else match.group(1)
    
    return "Unknown"

def extract_medical_phrases(text: str, keywords: List[str]) -> List[str]:
    """Extract phrases containing medical keywords."""
    sentences = sent_tokenize(text)
    phrases = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for keyword in keywords:
            if keyword.lower() in sentence_lower:
                # Extract the sentence or a relevant phrase
                phrases.append(sentence.strip())
                break
    
    return list(set(phrases))  # Remove duplicates

def extract_numbers_with_context(text: str) -> List[Dict[str, str]]:
    """Extract numbers with their medical context."""
    results = []
    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        # Look for numbers followed by medical terms
        patterns = [
            (r'(\d+)\s*(session|sessions)', 'sessions'),
            (r'(\d+)\s*(week|weeks|month|months)', 'time_period'),
            (r'(\d+)\s*(physiotherapy|therapy|treatment)', 'treatment'),
        ]
        
        for pattern, context_type in patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                results.append({
                    'number': match.group(1),
                    'context': match.group(2),
                    'sentence': sentence.strip(),
                    'type': context_type
                })
    
    return results

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove speaker labels if needed
    text = re.sub(r'^(Physician|Patient|Doctor):\s*', '', text, flags=re.MULTILINE)
    return text.strip()

def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract key medical phrases using simple keyword matching."""
    sentences = sent_tokenize(text)
    key_phrases = []
    
    all_keywords = SYMPTOM_KEYWORDS + TREATMENT_KEYWORDS + DIAGNOSIS_KEYWORDS + PROGNOSIS_KEYWORDS
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if sentence contains medical keywords
        if any(keyword in sentence_lower for keyword in all_keywords):
            # Extract important phrases (remove stopwords for cleaner extraction)
            words = word_tokenize(sentence)
            stop_words = set(stopwords.words('english'))
            important_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
            
            if important_words:
                # Take first 8-10 important words as phrase
                phrase = ' '.join(important_words[:10])
                if len(phrase) > 10:  # Only add substantial phrases
                    key_phrases.append(phrase)
    
    return key_phrases[:max_phrases]

def separate_speakers(text: str) -> Dict[str, List[str]]:
    """Separate text by speaker (Physician vs Patient)."""
    physician_text = []
    patient_text = []
    
    lines = text.split('\n')
    current_speaker = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Physician:') or line.startswith('Doctor:'):
            current_speaker = 'physician'
            content = re.sub(r'^(Physician|Doctor):\s*', '', line)
            if content:
                physician_text.append(content)
        elif line.startswith('Patient:'):
            current_speaker = 'patient'
            content = re.sub(r'^Patient:\s*', '', line)
            if content:
                patient_text.append(content)
        elif current_speaker:
            # Continuation of previous speaker's dialogue
            if current_speaker == 'physician':
                physician_text.append(line)
            else:
                patient_text.append(line)
    
    return {
        'physician': ' '.join(physician_text),
        'patient': ' '.join(patient_text)
    }

