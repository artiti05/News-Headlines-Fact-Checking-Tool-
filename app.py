from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
import json
import requests
from datetime import datetime

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag.extract_claim import extract_claim
from rag.live_retriever import LiveClaimRetriever
from rag.verify import EntailmentVerifier

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, 'chromadb_store')
HISTORY_FILE = os.path.join(BASE_DIR, 'data', 'history.json')

# Ensure data directory exists
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

# --- Persistent Initialization (Load once at startup) ---
logging.info("Initializing Live Retriever and MARBERT Verifier...")
retriever = LiveClaimRetriever()
verifier = EntailmentVerifier()

def unload_ollama_model():
    """Forces Ollama to unload the model to free VRAM."""
    try:
        requests.post("http://localhost:11434/api/generate", 
                      json={"model": "qwen2.5:7b", "keep_alive": 0})
        logging.info("Ollama model unloaded.")
    except:
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    title = data.get('title', '')
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "نص الخبر مطلوب"}), 400

    results = {
        "status": "success",
        "steps": [],
        "claim": "",
        "evidence": [],
        "verdict": "",
        "ar_verdict": "",
        "confidence": 0.0,
        "source_trust": "Unknown",
        "reasoning": "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        # Step 1: Extraction (Qwen)
        logging.info("Step 1: Extracting claim...")
        claim = extract_claim(title, text)
        results["claim"] = claim
        results["steps"].append("تم استخراج الادعاء الرئيسي بنجاح")

        # Free VRAM after Qwen so MARBERT has more room for high max_length
        unload_ollama_model()

        # Step 2: Retrieval (Live DuckDuckGo RAG + E5)
        logging.info("Step 2: Retrieving live evidence...")
        evidence_chunks = retriever.retrieve_evidence(claim, top_k=3)
        results["evidence"] = evidence_chunks
        results["steps"].append(f"تم العثور على {len(evidence_chunks)} من الأدلة المصدرية الحية")

        # Step 3: Verification (MARBERT)
        logging.info("Step 3: Verifying entailment...")
        
        # Combine evidence text for the verifier
        full_evidence = " ".join([c['text'] for c in evidence_chunks])
        en_label, ar_label, confidence = verifier.verify(claim, full_evidence)
        
        # Source Credibility Analysis
        credible_count = 0
        total_sources = len(evidence_chunks)
        
        for chunk in evidence_chunks:
            cred = chunk['metadata'].get('credibility', '').lower()
            if cred in ['credible', 'true', 'trusted']:
                credible_count += 1
        
        if total_sources > 0:
            trust_ratio = credible_count / total_sources
            if trust_ratio > 0.7: results["source_trust"] = "High"
            elif trust_ratio > 0.3: results["source_trust"] = "Medium"
            else: results["source_trust"] = "Low"

        results["verdict"] = en_label
        results["ar_verdict"] = ar_label
        results["confidence"] = round(float(confidence), 2) # Round for cleaner UI
        results["steps"].append(f"تم تحليل موثوقية المصادر ({results['source_trust']})")

        # Clean Evidence for UI
        clean_evidence = []
        unique_urls = set()
        for chunk in evidence_chunks:
            url = chunk['metadata'].get('url', '')
            if url and url not in unique_urls:
                unique_urls.add(url)
                source_name = chunk['metadata'].get('source', 'موقع إخباري')
                clean_evidence.append({
                    "source": source_name.replace('.com', '').replace('.net', '').capitalize(),
                    "url": url,
                    "snippet": chunk['text'][:150] + "..."
                })
        
        results["evidence"] = clean_evidence

        # Smart Reasoning Generation
        if not evidence_chunks:
            reasoning = f"لم أتمكن من العثور على أي تغطية إخبارية موثوقة للادعاء: '{claim}'. نظراً لعدم وجود أي مصادر رسمية تؤكد هذا الخبر، فإنه يُعتبر غير مؤكد."
        else:
            sources_list = list(set([e['source'] for e in clean_evidence]))
            sources_str = " و ".join(sources_list)
            if ar_label == "صحيح":
                reasoning = f"هذا الادعاء صحيح. تم تأكيده من خلال تقارير متطابقة في ({sources_str})."
            elif ar_label == "خاطئ":
                reasoning = f"هذا الادعاء خاطئ. التقارير في ({sources_str}) قدمت أدلة تتناقض تماماً مع هذا الخبر."
            else:
                reasoning = f"هذا الادعاء غير مؤكد حالياً، رغم وجود تغطية في ({sources_str})."
                
        results["reasoning"] = reasoning

        # Save to history
        save_to_history(title, claim, ar_label, results["timestamp"])

        return jsonify(results)

    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        return jsonify({"error": str(e)}), 500


def save_to_history(title, claim, verdict, timestamp):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    history.insert(0, {
        "title": title or claim[:50] + "...",
        "verdict": verdict,
        "timestamp": timestamp
    })
    
    # Keep last 10
    history = history[:10]
    
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

@app.route('/history', methods=['GET'])
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/admin/db', methods=['GET'])
def admin_db_view():
    # Basic security check
    auth_key = request.args.get('key')
    if auth_key != "admin123":
        return "غير مسموح بالدخول", 403
    
    try:
        from phase3_retrieval.retriever import ClaimRetriever
        manager = ClaimRetriever(CHROMA_DB_PATH)
        
        count = manager.collection.count()
        samples = manager.collection.peek(limit=10)
        
        db_data = {
            "total_count": count,
            "samples": []
        }
        
        for i in range(len(samples['ids'])):
            db_data["samples"].append({
                "id": samples['ids'][i],
                "metadata": samples['metadatas'][i],
                "text": samples['documents'][i]
            })
            
        return render_template('admin.html', data=db_data)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
