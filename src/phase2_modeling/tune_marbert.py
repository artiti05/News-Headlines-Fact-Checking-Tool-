import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'nli_dataset_optimized.csv')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'marbert_factcheck_best')

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

def load_data():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['claim', 'evidence', 'nli_label'])
    
    claims = df['claim'].tolist()
    evidences = df['evidence'].tolist()
    labels = df['nli_label'].tolist()
    
    return train_test_split(claims, evidences, labels, test_size=0.1, random_state=42)

def main():
    print("Loading Dataset...")
    train_c, val_c, train_e, val_e, train_l, val_l = load_data()

    print("Loading Tokenizer...")
    model_name = "UBC-NLP/MARBERTv2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(train_c, train_e, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_c, val_e, truncation=True, padding=True, max_length=128)

    train_dataset = NLIDataset(train_encodings, train_l)
    val_dataset = NLIDataset(val_encodings, val_l)

    # Hyperparameters to test
    learning_rates = [2e-5, 3e-5, 5e-5]
    batch_sizes = [8, 16]
    
    best_loss = float('inf')
    best_config = {}

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n--- Testing LR: {lr}, Batch Size: {bs} ---")
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
            
            training_args = TrainingArguments(
                output_dir=f'./results_tune/lr_{lr}_bs_{bs}',
                num_train_epochs=3,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                warmup_steps=100,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr,
                load_best_model_at_end=True,
                logging_steps=50,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            trainer.train()
            eval_results = trainer.evaluate()
            current_loss = eval_results['eval_loss']
            
            print(f"Validation Loss for LR {lr}, BS {bs}: {current_loss}")
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_config = {'lr': lr, 'batch_size': bs}
                print("New best model found! Saving...")
                trainer.save_model(MODEL_OUTPUT_DIR)
                tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print(f"\n=== Hyperparameter Tuning Complete ===")
    print(f"Best Configuration: Learning Rate {best_config['lr']}, Batch Size {best_config['batch_size']}")
    print(f"Best Validation Loss: {best_loss}")
    print(f"Best model saved to {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
