import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, 'src'))

from rag.extract_claim import extract_claim
from rag.live_retriever import LiveClaimRetriever
from rag.verify import EntailmentVerifier

def run_test():
    print("=== بدء اختبار نظام التحقق المباشر (Live RAG) ===")
    
    # قائمة حالات الاختبار (اختر رقم الحالة من 0 إلى 4)
    test_cases = [
        "أعلنت حكزمة ايران ان امريكا هي حليف لنا وليست عدوة بل تسعى للسلام معانا وان اتفاقات قد تمت بخصوص المشروع النووي الايراني", # 0: False (News contradiction)
        "المملكة العربية السعودية تعلن عن إطلاق مشروع ذا لاين في مدينة نيوم كأول مدينة ذكية بلا شوارع.", # 1: True (Fact)
        "النادي الأهلي المصري يتوج بلقب دوري أبطال أفريقيا للمرة الثانية عشرة في تاريخه.", # 2: True (Recent Sport Fact)
        "رسمياً: وكالة ناسا تعلن أن الأرض مكعبة الشكل وليست كروية كما كان يعتقد سابقاً.", # 3: False (Scientific contradiction)
        "ليونيل ميسي يعلن رسمياً اعتزاله كرة القدم نهائياً والانضمام لتدريب نادي الهلال السعودي غداً." # 4: False (Sport News)
    ]
    
    case_index = 0 # <--- غير هذا الرقم لتجربة حالات أخرى
    test_text = test_cases[case_index]
    
    print(f"\n[اختبار حالة رقم {case_index}]")
    print("\n1. النص الأصلي:")
    print(test_text)
    
    print("\n2. جاري استخراج الادعاء باستخدام Qwen...")
    claim = extract_claim("خبر عاجل", test_text)
    print(f"الادعاء المستخرج: {claim}")
    
    print("\n3. جاري البحث المباشر عن الأدلة باستخدام DuckDuckGo...")
    retriever = LiveClaimRetriever()
    evidence_chunks = retriever.retrieve_evidence(claim, top_k=1)
    
    if not evidence_chunks:
        print("لم يتم العثور على أدلة (الخبر غير موجود في المواقع الموثوقة).")
        print("\nالنتيجة النهائية:")
        print("غير مؤكد (Unverified) - لا توجد مصادر موثوقة تدعم هذا الادعاء.")
        return
        
    best_evidence = evidence_chunks[0]
    print(f"أفضل دليل وجدناه (درجة {best_evidence['score']:.2f}):")
    print(best_evidence['text'])
    print(f"المصدر: {best_evidence['metadata']['url']}")
    
    print("\n4. جاري التحقق من صحة الادعاء باستخدام MARBERT...")
    # Using the local MARBERT model we just defined (it might use the base or the finetuned one depending on what verify.py defaults to)
    try:
        verifier = EntailmentVerifier()
        best_evidence_text = best_evidence['text']
        en_label, ar_label, confidence = verifier.verify(claim, best_evidence_text)
        print("\nالنتيجة النهائية:")
        print(f"النتيجة: {ar_label} ({en_label}) - ثقة النموذج: {confidence:.2f}")
    except Exception as e:
        print(f"خطأ في نموذج MARBERT: {e}")

if __name__ == "__main__":
    run_test()
