# Arabic News Headlines Fact-Checking Tool

An advanced fact-checking platform for Arabic news headlines, combining **Live RAG (Retrieval-Augmented Generation)** with a **two-stage fine-tuned MARBERTv2** classifier. The tool automatically extracts factual claims from news articles, retrieves corroborating or contradicting evidence from whitelisted trusted news sources in real time, and outputs a verified status: **True (صحيح)**, **False (خاطئ)**, or **Neutral/Unverified (غير مؤكد)**.

---

## 🚀 Key Features

*   **Claim Extraction**: Automated parsing of full article text or headlines to extract a singular, clear claim sentence using a locally hosted `qwen2.5:7b` LLM.
*   **Live Web Retrieval (RAG)**: Real-time search query formulation and crawling of trusted Arabic news domains (filtered via whitelisted sources) with semantic passage alignment powered by the `multilingual-e5-small` embedding model.
*   **Deep NLI Classification**: State-of-the-art Natural Language Inference (NLI) classification utilizing a fine-tuned Arabic-specific transformer model (**MARBERTv2**).
*   **Interactive Web UI**: Premium, responsive dashboard interface built with **Flask** to input articles, inspect retrieved evidence snippets, view step-by-step verification progress, and review historic queries.

---

## 📊 Dataset & Class Distribution

The model is adapted to the political news domain using **3,420 domain-specific news records** extracted from the **Arabic Fake News Dataset (AFND)** (labeled using Misbar).

### Split Structure (Stratified 90/10)
*   **Training Set (Stratified)**: 3,078 records (90%)
*   **Validation Set (Stratified)**: 342 records (10%)

### Class Percentages
The class distribution within the domain dataset ([final_balanced_dataset.csv](data/processed/final_balanced_dataset.csv)) is as follows:

| NLI Label | Meaning | Record Count | Percentage |
| :--- | :--- | :---: | :---: |
| **`1`** | **True / Entails** (صحيح) | 1,673 | **48.92%** |
| **`0`** | **Neutral / Unverified** (غير مؤكد) | 900 | **26.32%** |
| **`2`** | **False / Contradicts** (خاطئ) | 847 | **24.77%** |

---

## 🧠 Methodology

Our training pipeline employs **Two-Stage Domain-Adaptive Fine-Tuning**:

1.  **Stage 1: NLI Task Learning (General Arabic NLI)**
    *   **Dataset**: The Arabic split of the **XNLI Dataset** (392,702 training pairs, 2,490 validation pairs, 5,010 test pairs).
    *   **Objective**: Introduce the raw `UBC-NLP/MARBERTv2` model to NLI structural formats (understanding premise-hypothesis relations).
    *   **Performance**: Stage 1 validation reached **Macro F1: 0.7551** and **Accuracy: 0.7554**.
2.  **Stage 2: Domain Adaptation (Political Fact-Checking)**
    *   **Dataset**: The 3,420 domain-specific claim-evidence pairs from AFND.
    *   **Objective**: Adapt the Stage-1 model checkpoint specifically to fact-checking and political terminology.
    *   **Performance**: Final Stage 2 validation reached **Macro F1: 0.7864** and **Accuracy: 0.8041**.

### Live Pipeline Flow
```
[User Input Article/Headline]
             │
             ▼
    [Extract Claim] (Qwen2.5:7b via Ollama)
             │
             ▼
    [Live Evidence Search] (DuckDuckGo + Whitelist Domain Filter)
             │
             ▼
    [Semantic Passage Alignment] (E5 Embeddings Similarity)
             │
             ▼
    [NLI Verification] (Fine-tuned MARBERTv2 on Claim + Evidence)
             │
             ▼
[Output: True (صحيح) / False (خاطئ) / Neutral (غير مؤكد)]
```

---

## 🛠️ Installation & Setup

### 1. Clone & Setup Environment
Ensure you have Python installed, then create and activate a virtual environment:
```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate

# Activate venv (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama (Local LLM)
Make sure Ollama is installed locally and pull the Qwen model:
```bash
# Start Ollama service, then run:
ollama run qwen2.5:7b
```

---

## ⚙️ Running the Application

### 1. Data Preparation & Model Training (Optional)
If you want to re-run data processing and fine-tuning scripts:
```bash
# Build the processed NLI dataset
python src/phase1_data_prep/build_nli_dataset.py

# Run baseline training
python src/phase2_modeling/train_marbert.py

# Run hyperparameter tuning (saves the best model to models/marbert_factcheck_best)
python src/phase2_modeling/tune_marbert.py

# Run model evaluation
python src/phase2_modeling/evaluate_marbert.py
```

### 2. Test the Live Pipeline (CLI)
You can run a quick command-line test using sample news statements:
```bash
python test_live_pipeline.py
```

### 3. Start the Web Dashboard
Launch the interactive web application:
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`** to access the premium verification user interface.