import ollama
import logging

logging.basicConfig(level=logging.INFO)

def extract_claim(title, text):
    sys_prompt = "أنت مساعد ذكي للتحقق من الأخبار. قم باستخراج الادعاء الرئيسي الوحيد (جملة واحدة فقط) من هذا النص الإخباري، بدون أي إضافات أو شروحات."
    try:
        logging.info("Extracting claim using Qwen...")
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
        logging.error(f"Error during claim extraction: {e}")
        return "تعذر استخراج الادعاء"
