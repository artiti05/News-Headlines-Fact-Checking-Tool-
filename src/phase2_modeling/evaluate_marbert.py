import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_balanced_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'marbert_factcheck_best')

class NLIDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_model():
    print(f"Loading Dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}.")
        return

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['claim', 'evidence', 'nli_label'])
    
    claims = df['claim'].tolist()
    evidences = df['evidence'].tolist()
    labels = df['nli_label'].astype(int).tolist()

    print(f"Loading Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Fine-tuned model not found. Using base MARBERTv2 for evaluation.")
        model_name = "UBC-NLP/MARBERTv2"
    else:
        model_name = MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    print("Tokenizing data...")
    encodings = tokenizer(claims, evidences, truncation=True, padding=True, max_length=512)
    dataset = NLIDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=16)

    all_preds = []
    all_labels = []

    print("Starting Evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            batch_labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nClassification Report:")
    # Label mapping: 0=Neutral, 1=True, 2=False
    target_names = ['Neutral (0)', 'True (1)', 'False (2)']
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_model()
