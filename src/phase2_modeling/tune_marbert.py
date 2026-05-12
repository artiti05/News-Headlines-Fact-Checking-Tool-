import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_balanced_dataset.csv')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'marbert_factcheck_best')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}

def load_data():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['claim', 'evidence', 'nli_label'])
    
    claims = df['claim'].tolist()
    evidences = df['evidence'].tolist()
    labels = df['nli_label'].tolist()
    
    return train_test_split(claims, evidences, labels, test_size=0.1, random_state=42)

from transformers import TrainerCallback

class TrainingMetricsCallback(TrainerCallback):
    """Callback to evaluate on training set and log results to state.log_history."""
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer')
        if trainer is None:
            return control
        
        # Evaluate on a sample of training data for speed
        train_sub = torch.utils.data.Subset(self.train_dataset, range(min(500, len(self.train_dataset))))
        metrics = trainer.evaluate(eval_dataset=train_sub, metric_key_prefix="train")
        
        # IMPORTANT: Add epoch to these metrics so they show up in the history correctly
        metrics["epoch"] = state.epoch
        trainer.log(metrics)
        return control

def plot_training_curves(history, config_name):
    """Generates loss, f1, and acc curves for both training and validation."""
    train_epochs, t_loss = [], []
    eval_epochs, v_loss = [], []
    t_f1, v_f1, t_acc, v_acc = [], [], [], []
    
    for entry in history:
        if 'epoch' in entry:
            e = entry['epoch']
            # Training Loss (logged frequently by Trainer)
            if 'loss' in entry:
                train_epochs.append(e)
                t_loss.append(entry['loss'])
            
            # Validation Metrics (logged once per epoch)
            if 'eval_loss' in entry:
                eval_epochs.append(e)
                v_loss.append(entry['eval_loss'])
                v_f1.append(entry.get('eval_f1', 0))
                v_acc.append(entry.get('eval_accuracy', 0))
            
            # Training Metrics from our callback (logged once per epoch)
            if 'train_f1' in entry:
                t_f1.append(entry['train_f1'])
                t_acc.append(entry['train_accuracy'])
                # If for some reason eval_epochs wasn't populated yet
                if not any(abs(x - e) < 0.01 for x in eval_epochs):
                    pass # Just keep collecting, we'll align below

    plt.figure(figsize=(18, 5))
    
    # 1. Loss Curve
    plt.subplot(1, 3, 1)
    if t_loss: plt.plot(train_epochs, t_loss, label='Train Loss', color='blue', alpha=0.6)
    if v_loss: plt.plot(eval_epochs, v_loss, label='Val Loss', marker='o', color='red')
    plt.title(f'Loss: {config_name}')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

    # 2. F1 Score Curve
    plt.subplot(1, 3, 2)
    # Align training metrics with epochs
    t_epochs = eval_epochs[:len(t_f1)]
    if t_f1: plt.plot(t_epochs, t_f1, label='Train F1', linestyle='--', color='blue')
    if v_f1: plt.plot(eval_epochs, v_f1, label='Val F1', marker='s', color='green')
    plt.title(f'F1 Score: {config_name}')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

    # 3. Accuracy Curve
    plt.subplot(1, 3, 3)
    if t_acc: plt.plot(t_epochs, t_acc, label='Train Acc', linestyle='--', color='blue')
    if v_acc: plt.plot(eval_epochs, v_acc, label='Val Acc', marker='^', color='orange')
    plt.title(f'Accuracy: {config_name}')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{config_name}_performance.png'))
    plt.close()

def main():
    print(f"Loading Dataset from {DATASET_PATH}...")
    train_c, val_c, train_e, val_e, train_l, val_l = load_data()

    print("Loading Tokenizer...")
    model_name = "UBC-NLP/MARBERTv2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Hyperparameters - The "Smooth & Deep" Configuration
    learning_rates = [5e-6, 1e-6]
    batch_sizes = [32] 
    max_lengths = [256, 512] # Focusing on 256 for speed; set to [512] if evidence is long
    weight_decays = [0.1]
    
    # Calculate search size
    search_size = len(learning_rates) * len(batch_sizes) * len(max_lengths) * len(weight_decays)
    print(f"\n=== Starting Stable Training. Search Size: {search_size} Combinations ===")

    best_f1 = 0.0
    best_config = {}
    tuning_results = []

    for m_len in max_lengths:
        print(f"\n--- Tokenizing for Max Length: {m_len} ---")
        train_encodings = tokenizer(train_c, train_e, truncation=True, padding=True, max_length=m_len)
        val_encodings = tokenizer(val_c, val_e, truncation=True, padding=True, max_length=m_len)
        train_dataset = NLIDataset(train_encodings, train_l)
        val_dataset = NLIDataset(val_encodings, val_l)

        for lr in learning_rates:
            for bs in batch_sizes:
                for wd in weight_decays:
                    config_name = f"ml{m_len}_lr{lr}_bs{bs}_wd{wd}"
                    print(f"\n--- Testing {config_name} (Stable Mode) ---")
                    
                    try:
                        torch.cuda.empty_cache()
                        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
                        
                        training_args = TrainingArguments(
                            output_dir=f'./results_tune/{config_name}',
                            num_train_epochs=5, # Increased epochs for better Neutral learning
                            per_device_train_batch_size=bs,
                            per_device_eval_batch_size=bs,
                            gradient_accumulation_steps=1,
                            warmup_ratio=0.1,
                            weight_decay=wd,
                            label_smoothing_factor=0.1, 
                            bf16=torch.cuda.is_available(),
                            eval_strategy="epoch",
                            save_strategy="epoch",
                            learning_rate=lr,
                            lr_scheduler_type="cosine",
                            load_best_model_at_end=True,
                            metric_for_best_model="f1",
                            logging_steps=10,
                            save_total_limit=1,
                            report_to="none"
                        )

                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            compute_metrics=compute_metrics,
                            callbacks=[TrainingMetricsCallback(train_dataset)]
                        )

                        trainer.train()
                        eval_results = trainer.evaluate()
                        current_f1 = eval_results['eval_f1']
                        
                        plot_training_curves(trainer.state.log_history, config_name)
                        tuning_results.append({'config': config_name, 'f1': current_f1})
                        
                        if current_f1 > best_f1:
                            best_f1 = current_f1
                            best_config = {'max_len': m_len, 'lr': lr, 'batch_size': bs, 'weight_decay': wd, 'name': config_name}
                            print(f"New best model found! (F1: {current_f1:.4f}). Saving...")
                            trainer.save_model(MODEL_OUTPUT_DIR)
                            trainer.save_model(os.path.join(BASE_DIR, 'models', 'marbert_factcheck_finetuned'))
                            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
                            tokenizer.save_pretrained(os.path.join(BASE_DIR, 'models', 'marbert_factcheck_finetuned'))
                    
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            print(f"!!! OOM Error caught for {config_name}. Skipping.")
                            torch.cuda.empty_cache()
                        else:
                            raise e
    
    # Final Comparison Chart
    if tuning_results:
        res_df = pd.DataFrame(tuning_results).sort_values(by='f1', ascending=False)
        plt.figure(figsize=(10, 6))
        plt.barh(res_df['config'], res_df['f1'], color='skyblue')
        plt.xlabel('Validation F1 Score')
        plt.title('Hyperparameter Configuration Comparison')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'all_configs_comparison.png'))

    print(f"\n=== Hyperparameter Tuning Complete ===")
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"All visuals saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()

