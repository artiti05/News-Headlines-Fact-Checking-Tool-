import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)

import os

class EntailmentVerifier:
    def __init__(self):
        # Load the locally fine-tuned model
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.model_path = os.path.join(base_dir, 'models', 'marbert_factcheck_finetuned')
        
        # Fallback to base model if fine-tuned is not found
        if not os.path.exists(self.model_path):
            self.model_path = "UBC-NLP/MARBERTv2"
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Loading MARBERT Verifier from {self.model_path} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=3)
        self.model.to(self.device)
        self.model.eval()
        
    def verify(self, claim, evidence_text):
        if not evidence_text:
            return "Neutral", "غير مؤكد", 0.0
            
        inputs = self.tokenizer(claim, evidence_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        labels_map_ar = {
            0: "غير مؤكد",
            1: "صحيح",
            2: "خاطئ"
        }
        labels_map_en = {
            0: "Neutral",
            1: "True",
            2: "False"
        }
        
        # If confidence is extremely low, fall back to Neutral
        if confidence < 0.45:
            pred_idx = 0
            
        ar_verdict = labels_map_ar.get(pred_idx, "غير مؤكد")
        en_verdict = labels_map_en.get(pred_idx, "Neutral")
        
        return en_verdict, ar_verdict, confidence
