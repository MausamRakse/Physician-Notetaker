"""
Sentiment and Intent Analysis for medical conversations.
"""

import json
import re
from typing import Dict, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    """Analyzer for patient sentiment and intent."""
    
    def __init__(self):
        """Initialize sentiment analysis models."""
        self.sentiment_model = None
        self.tokenizer = None
        
        try:
            # Use DistilBERT for sentiment analysis (faster than BERT)
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Warning: Could not load sentiment model: {e}")
            print("Using rule-based sentiment analysis as fallback.")
            self.sentiment_pipeline = None
        
        # Intent keywords
        self.intent_keywords = {
            "Seeking reassurance": [
                "worry", "worried", "concerned", "anxious", "hope", "wondering",
                "afraid", "fear", "nervous", "apprehensive", "uncertain"
            ],
            "Reporting symptoms": [
                "pain", "ache", "discomfort", "feeling", "experiencing",
                "having", "symptoms", "problem", "issue", "trouble"
            ],
            "Expressing concern": [
                "concern", "worried about", "afraid of", "fear that",
                "not sure", "uncertain", "question"
            ],
            "Seeking information": [
                "what", "how", "why", "when", "where", "explain", "tell me",
                "understand", "mean", "question"
            ],
            "Expressing relief": [
                "relief", "relieved", "glad", "happy", "thankful", "appreciate",
                "good to hear", "great", "wonderful"
            ]
        }
    
    def classify_sentiment(self, text: str) -> str:
        """Classify sentiment as Anxious, Neutral, or Reassured."""
        text_lower = text.lower()
        
        # Use transformer model if available
        if self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text[:512])  # Limit length
                label = result[0]['label']
                score = result[0]['score']
                
                # Map to our categories
                if label == 'POSITIVE':
                    # Check if it's reassurance or just positive
                    if any(word in text_lower for word in ['relief', 'relieved', 'good to hear', 'thank']):
                        return "Reassured"
                    else:
                        return "Neutral"
                else:  # NEGATIVE
                    # Check if it's anxiety or just negative
                    if any(word in text_lower for word in ['worry', 'worried', 'concerned', 'anxious', 'afraid']):
                        return "Anxious"
                    else:
                        return "Neutral"
            except Exception as e:
                print(f"Error in sentiment classification: {e}")
                # Fall through to rule-based
        
        # Rule-based fallback
        anxious_keywords = ['worried', 'concerned', 'anxious', 'afraid', 'fear', 'nervous', 'apprehensive']
        reassured_keywords = ['relief', 'relieved', 'glad', 'thankful', 'appreciate', 'good to hear', 'great']
        
        anxious_count = sum(1 for word in anxious_keywords if word in text_lower)
        reassured_count = sum(1 for word in reassured_keywords if word in text_lower)
        
        if anxious_count > reassured_count and anxious_count > 0:
            return "Anxious"
        elif reassured_count > anxious_count and reassured_count > 0:
            return "Reassured"
        else:
            return "Neutral"
    
    def detect_intent(self, text: str) -> str:
        """Detect patient intent."""
        text_lower = text.lower()
        intent_scores = {}
        
        # Score each intent based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with highest score
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return "General inquiry"
    
    def analyze_patient_dialogue(self, text: str) -> Dict[str, str]:
        """Analyze patient's dialogue for sentiment and intent."""
        # Extract patient statements
        patient_text = self._extract_patient_text(text)
        
        if not patient_text:
            patient_text = text  # Use full text if can't separate
        
        sentiment = self.classify_sentiment(patient_text)
        intent = self.detect_intent(patient_text)
        
        return {
            "Sentiment": sentiment,
            "Intent": intent
        }
    
    def _extract_patient_text(self, text: str) -> str:
        """Extract only patient's dialogue from conversation."""
        patient_lines = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Patient:'):
                content = re.sub(r'^Patient:\s*', '', line)
                if content:
                    patient_lines.append(content)
            elif line and not line.startswith(('Physician:', 'Doctor:')):
                # Might be continuation of patient dialogue
                if patient_lines:  # If we've seen patient text before
                    patient_lines.append(line)
        
        return ' '.join(patient_lines)
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Main method to analyze sentiment and intent."""
        return self.analyze_patient_dialogue(text)
    
    def generate_sentiment_json(self, text: str) -> str:
        """Generate JSON output for sentiment analysis."""
        result = self.analyze_sentiment(text)
        return json.dumps(result, indent=2)

