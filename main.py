"""
Main pipeline script for Physician Notetaker.
Processes medical transcripts and generates summaries, sentiment analysis, and SOAP notes.
"""

from medical_nlp import MedicalNLPPipeline
from sentiment_analysis import SentimentAnalyzer
from soap_generator import SOAPNoteGenerator

# Sample conversation from the requirements
SAMPLE_CONVERSATION = """
Physician: Good morning, Ms. Jones. How are you feeling today?

Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.

Physician: I understand you were in a car accident last September. Can you walk me through what happened?

Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.

Physician: That sounds like a strong impact. Were you wearing your seatbelt?

Patient: Yes, I always do.

Physician: What did you feel immediately after the accident?

Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.

Physician: Did you seek medical attention at that time?

Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.

Physician: How did things progress after that?

Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.

Physician: That makes sense. Are you still experiencing pain now?

Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.

Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?

Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.

Physician: And how has this impacted your daily life? Work, hobbies, anything like that?

Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.

Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.

[Physical Examination Conducted]

Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.

Patient: That's a relief!

Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.

Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?

Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.

Patient: Thank you, doctor. I appreciate it.

Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
"""

def main():
    """Main function to run the complete pipeline."""
    print("=" * 80)
    print("Physician Notetaker - Medical NLP Pipeline")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing components...")
    nlp_pipeline = MedicalNLPPipeline()
    sentiment_analyzer = SentimentAnalyzer()
    soap_generator = SOAPNoteGenerator()
    print("[OK] Components initialized\n")
    
    # Process the sample conversation
    print("Processing sample conversation...")
    print("-" * 80)
    
    # 1. Medical NLP Summarization
    print("\n1. MEDICAL NLP SUMMARIZATION")
    print("=" * 80)
    medical_summary = nlp_pipeline.extract_medical_details(SAMPLE_CONVERSATION)
    summary_json = nlp_pipeline.generate_summary_json(SAMPLE_CONVERSATION)
    print(summary_json)
    
    # 2. Sentiment & Intent Analysis
    print("\n\n2. SENTIMENT & INTENT ANALYSIS")
    print("=" * 80)
    sentiment_result = sentiment_analyzer.analyze_sentiment(SAMPLE_CONVERSATION)
    sentiment_json = sentiment_analyzer.generate_sentiment_json(SAMPLE_CONVERSATION)
    print(sentiment_json)
    
    # 3. SOAP Note Generation
    print("\n\n3. SOAP NOTE GENERATION")
    print("=" * 80)
    soap_note = soap_generator.generate_soap_note(SAMPLE_CONVERSATION, medical_summary)
    soap_json = soap_generator.generate_soap_json(SAMPLE_CONVERSATION)
    print(soap_json)
    
    # Save outputs to files
    print("\n\nSaving outputs to files...")
    with open("medical_summary.json", "w") as f:
        f.write(summary_json)
    print("[OK] Saved: medical_summary.json")
    
    with open("sentiment_analysis.json", "w") as f:
        f.write(sentiment_json)
    print("[OK] Saved: sentiment_analysis.json")
    
    with open("soap_note.json", "w") as f:
        f.write(soap_json)
    print("[OK] Saved: soap_note.json")
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()

