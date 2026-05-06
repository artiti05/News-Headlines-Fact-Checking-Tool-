import os
import json
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)

class LiveClaimRetriever:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load trusted domains
        self.trusted_domains = []
        trusted_file = os.path.join(os.path.dirname(__file__), 'trusted_urls.json')
        if os.path.exists(trusted_file):
            with open(trusted_file, 'r', encoding='utf-8') as f:
                urls = json.load(f)
                # Extract clean domains (e.g., 'www.bbc.com' from 'https://www.bbc.com/arabic')
                self.trusted_domains = [u.split('/')[2].replace('www.', '') for u in urls if '/' in u]
        
        logging.info(f"Loaded {len(self.trusted_domains)} trusted domains. Loading E5 model on {self.device}...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-small', device=self.device)

    def chunk_text(self, text, chunk_size=100):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def scrape_url(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
            return text
        except:
            return ""

    def retrieve_evidence(self, claim, top_k=1):
        logging.info(f"Live Searching DuckDuckGo for: {claim}")
        
        try:
            with DDGS() as ddgs:
                organic_results = list(ddgs.text(claim, region='wt-wt', safesearch='off', max_results=15))
        except Exception as e:
            logging.error(f"DDGS error: {e}")
            return []

        all_chunks = []
        chunk_to_meta = {}
        
        for res in organic_results:
            url = res.get("href")
            source = url.split('/')[2] if url else "Unknown"
            if not url: continue
            
            # Domain Filtering
            if self.trusted_domains:
                domain = url.split('/')[2].replace('www.', '')
                if not any(trusted in domain for trusted in self.trusted_domains):
                    continue # Skip untrusted websites
                    
            scraped_text = self.scrape_url(url)
            if len(scraped_text) < 100:
                scraped_text = res.get("body", "")
                
            chunks = self.chunk_text(scraped_text)
            all_chunks.extend(chunks)
            for c in chunks:
                chunk_to_meta[c] = {"url": url, "source": source, "credibility": "live_search"}
                
        if not all_chunks:
            return []
            
        # Semantic Search within scraped chunks
        claim_emb = self.model.encode([f"query: {claim}"])[0]
        passage_embs = self.model.encode([f"passage: {c}" for c in all_chunks])
        
        similarities = np.dot(passage_embs, claim_emb) / (np.linalg.norm(passage_embs, axis=1) * np.linalg.norm(claim_emb))
        
        best_indices = np.argsort(similarities)[::-1][:top_k]
        
        evidence = []
        for idx in best_indices:
            text = all_chunks[idx]
            evidence.append({
                "text": text,
                "metadata": chunk_to_meta[text],
                "score": float(similarities[idx])
            })
            
        return evidence
