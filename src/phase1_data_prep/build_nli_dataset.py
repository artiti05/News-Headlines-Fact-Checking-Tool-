import os
import json
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Ensure GPU is used
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'claim_data', 'AFND', 'Dataset')
SOURCES_JSON = os.path.join(BASE_DIR, 'data', 'claim_data', 'AFND', 'sources.json')
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'nli_dataset_optimized.csv')
TRUSTED_URLS_FILE = os.path.join(BASE_DIR, 'src', 'rag', 'trusted_urls.json')

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Political keywords for filtering
POLITICAL_KEYWORDS = ['حكومة', 'رئيس', 'برلمان', 'انتخابات', 'سياسة', 'معارضة', 'حزب', 'وزير', 'الجيش', 'أمن']

print("Loading E5 embedding model...")
model = SentenceTransformer('intfloat/multilingual-e5-small', device=device)

# Load trusted domains
trusted_domains = []
if os.path.exists(TRUSTED_URLS_FILE):
    with open(TRUSTED_URLS_FILE, 'r', encoding='utf-8') as f:
        urls = json.load(f)
        trusted_domains = [u.split('/')[2].replace('www.', '') for u in urls if '/' in u]
print(f"Loaded {len(trusted_domains)} trusted domains.")

def chunk_text(text, chunk_size=100):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def scrape_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code >= 400 or "blocked" in response.text.lower() or "captcha" in response.text.lower():
            return "BROKEN_URL"
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
        return text if text.strip() else "BROKEN_URL"
    except:
        return "BROKEN_URL"

def search_and_retrieve(claim):
    print(f"Searching DuckDuckGo for: {claim}")
    
    try:
        with DDGS() as ddgs:
            organic_results = list(ddgs.text(claim, region='wt-wt', safesearch='off', max_results=15))
    except Exception as e:
        print(f"DDGS error: {e}")
        return None, None, "DDGS_Error"

    all_chunks = []
    chunk_to_url = {}
    chunk_to_status = {}
    
    for res in organic_results:
        url = res.get("href")
        if not url: continue
        
        # Domain Filtering
        if trusted_domains:
            domain = url.split('/')[2].replace('www.', '')
            if not any(trusted in domain for trusted in trusted_domains):
                continue
                
        scraped_text = scrape_url(url)
        url_status = "Active"
        if scraped_text == "BROKEN_URL":
            url_status = "Broken/Blocked"
            scraped_text = res.get("body", "") # Fallback to DuckDuckGo snippet
            
        if not scraped_text:
            scraped_text = res.get("body", "") # Fallback to snippet
            
        chunks = chunk_text(scraped_text)
        all_chunks.extend(chunks)
        for c in chunks:
            chunk_to_url[c] = url
            chunk_to_status[c] = url_status
            
    if not all_chunks:
        return None, None, "Unknown"
        
    # Embed and find best
    claim_emb = model.encode([f"query: {claim}"])[0]
    passage_embs = model.encode([f"passage: {c}" for c in all_chunks])
    
    similarities = np.dot(passage_embs, claim_emb) / (np.linalg.norm(passage_embs, axis=1) * np.linalg.norm(claim_emb))
    best_idx = np.argmax(similarities)
    best_chunk = all_chunks[best_idx]
    
    return best_chunk, chunk_to_url[best_chunk], chunk_to_status[best_chunk]

def extract_claim(title, text):
    sys_prompt = "أنت مساعد ذكي للتحقق من الأخبار. استخرج الادعاء الرئيسي (جملة واحدة فقط) من هذا النص. أخرج الادعاء فقط بدون أي مقدمات، لا تكتب 'الادعاء هو'."
    try:
        resp = ollama.chat(model='qwen2.5:7b', messages=[
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': f"العنوان: {title}\nالنص: {text}"}
        ])
        claim = resp['message']['content'].strip()
        # Clean up common prefixes if the model still outputs them
        prefixes_to_remove = ["الادعاء الرئيسي هو:", "الادعاء الرئيسي:", "الادعاء هو:", "الادعاء:"]
        for p in prefixes_to_remove:
            if claim.startswith(p):
                claim = claim.replace(p, "").strip()
        return claim
    except Exception as e:
        print("Ollama Error:", e)
        return ""

def generate_reasoning_and_label(claim, evidence):
    prompt = f"""أنت خبير في التحقق من الأخبار. بناءً على هذا الدليل فقط، هل الادعاء صحيح أم خاطئ أم لا يمكن تأكيده؟
الادعاء: {claim}
الدليل: {evidence}

قم بالرد بصيغة JSON تحتوي على:
"reasoning": "شرح سبب الدعم أو النفي",
"label": "1 للإثبات (صحيح)، 2 للنفي (خاطئ)، 0 لغير مؤكد"
"""
    try:
        resp = ollama.chat(model='qwen2.5:7b', messages=[
            {'role': 'user', 'content': prompt}
        ], format='json')
        # Robust parsing for JSON
        content = resp['message']['content'].strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        result = json.loads(content)
        return result.get('reasoning', ''), int(result.get('label', 0))
    except Exception as e:
        print("Error generating label:", e)
        return "خطأ في التحليل", 0

def process_afnd():
    print("Loading AFND Dataset...")
    if not os.path.exists(SOURCES_JSON):
        print(f"Cannot find {SOURCES_JSON}")
        return
        
    with open(SOURCES_JSON, 'r', encoding='utf-8') as f:
        sources_cred = json.load(f)
        
    political_articles = []
    
    for src, cred in sources_cred.items():
        src_dir = os.path.join(DATASET_DIR, src)
        json_path = os.path.join(src_dir, 'scraped_articles.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    articles = data.get('articles', []) if isinstance(data, dict) else data
                    for art in articles:
                        if not isinstance(art, dict): continue
                        text = art.get('text', '')
                        if any(kw in text for kw in POLITICAL_KEYWORDS):
                            art['source'] = src
                            art['credibility'] = cred
                            political_articles.append(art)
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
                        
    print(f"Found {len(political_articles)} political articles. Sampling 200 for NLI dataset generation...")
    credible = [a for a in political_articles if a['credibility'] == 'credible']
    not_credible = [a for a in political_articles if a['credibility'] == 'not credible']
    
    # Take 5000 true and 5000 fake news to create a 10,000 sample dataset
    sample = random.sample(credible, min(5000, len(credible))) + random.sample(not_credible, min(5000, len(not_credible)))
    
    dataset = []
    
    for i, art in enumerate(sample):
        print(f"\nProcessing {i+1}/{len(sample)}: {art['title']}")
        claim = extract_claim(art['title'], art['text'])
        if not claim: continue
        print(f"Claim: {claim}")
        
        evidence, url, url_status = search_and_retrieve(claim)
        if not evidence: 
            print("No evidence found.")
            continue
            
        print(f"Evidence: {evidence[:100]}...")
        
        # Combo A/B: Real Evidence Match
        reasoning, label = generate_reasoning_and_label(claim, evidence)
        
        dataset.append({
            'claim': claim,
            'evidence': evidence,
            'evidence_url': url,
            'url_status': url_status,
            'evidence_domain': url.split('/')[2].replace('www.', '') if url else 'unknown',
            'reasoning': reasoning,
            'nli_label': label,
            'original_credibility': art['credibility'],
            'article_source': art.get('source', 'unknown')
        })
        
        # Combo C: Hard Neutral
        neutral_evidence = art['text'][:200]
        dataset.append({
            'claim': claim,
            'evidence': neutral_evidence,
            'evidence_url': 'local',
            'url_status': 'Local Data',
            'evidence_domain': 'local',
            'reasoning': "النص لم يثبت أو ينف الادعاء بشكل قاطع.",
            'nli_label': 0,
            'original_credibility': art['credibility'],
            'article_source': art.get('source', 'unknown')
        })
        
    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} records to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_afnd()
