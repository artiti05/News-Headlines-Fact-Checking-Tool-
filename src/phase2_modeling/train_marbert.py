import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_balanced_dataset.csv')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'marbert_factcheck_finetuned')

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

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

def train_model():
    print("Loading Dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Please run data prep first.")
        # Create dummy data for testing the pipeline if file is missing
        df = pd.DataFrame({
            'claim': ["الرئيس استقال", "قرار جديد للبرلمان", "لا يوجد قرار"],
            'evidence': ["قدم الرئيس استقالته اليوم", "البرلمان لم يصدر أي قرار", "تم تأجيل القرار"],
            'nli_label': [1, 2, 0]
        })
    else:
        df = pd.read_csv(DATASET_PATH)

    df = df.dropna(subset=['claim', 'evidence', 'nli_label'])
    
    claims = df['claim'].tolist()
    evidences = df['evidence'].tolist()
    labels = df['nli_label'].astype(int).tolist()

    # Split dataset
    train_c, val_c, train_e, val_e, train_l, val_l = train_test_split(
        claims, evidences, labels, test_size=0.2, random_state=42
    )

    print("Loading MARBERT tokenizer and model...")
    model_name = "UBC-NLP/MARBERTv2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We have 3 labels: 0=Neutral, 1=Entailment, 2=Contradiction
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Move to GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f"Model loaded on {device}")

    print("Tokenizing data...")
    train_encodings = tokenizer(train_c, train_e, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_c, val_e, truncation=True, padding=True, max_length=128)

    train_dataset = NLIDataset(train_encodings, train_l)
    val_dataset = NLIDataset(val_encodings, val_l)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("Starting Training...")
    trainer.train()

    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print("Training Complete!")

if __name__ == "__main__":
    train_model()
