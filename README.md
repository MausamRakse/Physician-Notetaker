# 🩺 Physician Notetaker

An AI-powered system for medical transcription, NLP-based summarization, and sentiment analysis.

## Features

1. **Medical NLP Summarization**
   - Named Entity Recognition (NER) for Symptoms, Treatment, Diagnosis, and Prognosis
   - Text Summarization to structured medical reports
   - Keyword Extraction for important medical phrases

2. **Sentiment & Intent Analysis**
   - Sentiment Classification (Anxious, Neutral, Reassured)
   - Intent Detection (Seeking reassurance, Reporting symptoms, etc.)

3. **SOAP Note Generation (Bonus)**
   - Automated conversion of transcripts to structured SOAP notes
   - Logical mapping of Subjective, Objective, Assessment, and Plan sections

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy English model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Download NLTK data (if needed):**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Usage

### Basic Usage

Run the main pipeline on the sample conversation:

```bash
python main.py
```

### Custom Usage

```python
from medical_nlp import MedicalNLPPipeline
from sentiment_analysis import SentimentAnalyzer
from soap_generator import SOAPNoteGenerator

# Initialize components
nlp_pipeline = MedicalNLPPipeline()
sentiment_analyzer = SentimentAnalyzer()
soap_generator = SOAPNoteGenerator()

# Process transcript
transcript = "Your conversation text here..."

# Get medical summary
summary = nlp_pipeline.extract_medical_details(transcript)

# Get sentiment analysis
sentiment = sentiment_analyzer.analyze_sentiment(transcript)

# Generate SOAP note
soap_note = soap_generator.generate_soap_note(transcript, summary)
```

## Project Structure

```
mousam/
├── main.py                 # Main pipeline script
├── medical_nlp.py          # Medical NLP summarization module
├── sentiment_analysis.py   # Sentiment and intent analysis module
├── soap_generator.py       # SOAP note generation module
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Output Format

### Medical Summary (JSON)
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}
```

### Sentiment Analysis (JSON)
```json
{
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance"
}
```

### SOAP Note (JSON)
```json
{
  "Subjective": {
    "Chief_Complaint": "...",
    "History_of_Present_Illness": "..."
  },
  "Objective": {
    "Physical_Exam": "...",
    "Observations": "..."
  },
  "Assessment": {
    "Diagnosis": "...",
    "Severity": "..."
  },
  "Plan": {
    "Treatment": "...",
    "Follow-Up": "..."
  }
}
```

## Model Information

- **NER**: spaCy's `en_core_web_sm` model with custom medical entity patterns
- **Sentiment Analysis**: DistilBERT-based model fine-tuned for medical sentiment
- **Keyword Extraction**: TF-IDF and pattern matching
- **SOAP Generation**: Rule-based and pattern-matching approach

## Handling Ambiguous Data

The system handles ambiguous or missing medical data by:
- Using pattern matching for common medical phrases
- Extracting context from surrounding sentences
- Providing confidence scores for extracted entities
- Marking uncertain fields in the output (e.g., "Not specified" when data is missing)
- Using multiple extraction strategies (pattern matching, keyword extraction, context analysis)
- Fallback to rule-based extraction when NLP models are unavailable

## Answers to Key Questions

### Medical NLP Summarization

**Q: How would you handle ambiguous or missing medical data in the transcript?**

**A:** The system uses a multi-layered approach:
1. **Pattern Matching**: Regex patterns for common medical phrases (e.g., "whiplash injury", "physiotherapy sessions")
2. **Context Extraction**: Analyzes surrounding sentences to infer missing information
3. **Keyword-Based Fallback**: Uses medical keyword dictionaries when structured extraction fails
4. **Default Values**: Returns "Not specified" for missing fields rather than making assumptions
5. **Multiple Extraction Methods**: Combines spaCy NER, pattern matching, and keyword extraction for robustness

**Q: What pre-trained NLP models would you use for medical summarization?**

**A:** 
- **Primary**: spaCy's `en_core_web_sm` for general NER and text processing
- **For Production**: Consider medical domain-specific models:
  - `en_core_sci_sm` (spaCy's scientific/medical model)
  - `BioBERT` or `ClinicalBERT` for medical entity recognition
  - `BlueBERT` for clinical text understanding
- **Alternative**: Hugging Face transformers like `bert-base-uncased` fine-tuned on medical datasets

### Sentiment & Intent Analysis

**Q: How would you fine-tune BERT for medical sentiment detection?**

**A:**
1. **Dataset Preparation**: Use medical conversation datasets (e.g., MIMIC-III, i2b2, or custom annotated medical dialogues)
2. **Labeling**: Annotate patient statements with sentiment labels (Anxious, Neutral, Reassured)
3. **Fine-tuning Process**:
   ```python
   # Use Hugging Face Trainer API
   from transformers import Trainer, TrainingArguments
   # Fine-tune on medical sentiment dataset
   # Use learning rate ~2e-5, batch size 16-32
   ```
4. **Domain Adaptation**: Pre-train on medical text before fine-tuning on sentiment
5. **Evaluation**: Use medical domain-specific validation sets

**Q: What datasets would you use for training a healthcare-specific sentiment model?**

**A:**
- **MIMIC-III**: Large clinical database with de-identified health data
- **i2b2**: Clinical NLP challenges with annotated medical text
- **Clinical Sentiment Analysis Dataset**: Custom annotated patient-physician conversations
- **MedDialog**: Medical dialogue datasets
- **UMLS**: Unified Medical Language System for medical terminology
- **Custom Annotation**: Annotate real medical transcripts with sentiment labels

### SOAP Note Generation

**Q: How would you train an NLP model to map medical transcripts into SOAP format?**

**A:**
1. **Supervised Learning Approach**:
   - Create training dataset: (transcript, SOAP_note) pairs
   - Use sequence-to-sequence models (T5, BART) or GPT-style models
   - Fine-tune on medical SOAP note datasets
2. **Structured Prediction**: Use BERT-based models with classification heads for each SOAP section
3. **Hybrid Approach** (Current Implementation):
   - Rule-based extraction for structured data
   - NLP models for free-text sections
   - Template filling with extracted entities

**Q: What rule-based or deep-learning techniques would improve the accuracy of SOAP note generation?**

**A:**
1. **Rule-Based Enhancements**:
   - Medical terminology dictionaries (SNOMED CT, ICD-10)
   - Temporal pattern recognition (e.g., "four weeks ago")
   - Speaker identification (Physician vs Patient)
   - Section-specific extraction rules
2. **Deep Learning Techniques**:
   - **T5/BART**: Fine-tuned for medical text summarization
   - **BERT-based Classification**: Multi-label classification for SOAP sections
   - **Named Entity Recognition**: Medical NER models (BioBERT, ClinicalBERT)
   - **Relation Extraction**: Identify relationships between entities
   - **Attention Mechanisms**: Focus on relevant parts of conversation
3. **Hybrid Improvements**:
   - Use rules for structured data, ML for free-text
   - Ensemble methods combining multiple models
   - Post-processing with medical knowledge bases
   - Validation against medical ontologies

## Notes

- The system is designed for demonstration purposes
- For production use, consider fine-tuning on medical domain datasets
- Always review AI-generated medical notes before clinical use
- Ensure compliance with healthcare data regulations (HIPAA, GDPR, etc.)

## License

This project is for educational purposes.

