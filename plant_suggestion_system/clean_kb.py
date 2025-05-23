# clean_kb.py
import json
from collections import defaultdict

def clean_rules(rules, min_confidence=0.5, min_feedback=1):
    unique_signatures = set()
    cleaned = []

    for rule in rules:
        # Ayıklanacak kurallar: düşük güven, tekrar eden koşullar, negatif feedback
        if rule.get("confidence", 0) < min_confidence:
            continue
        if rule.get("feedback", 0) < min_feedback:
            continue

        # Koşulları stringleştirip hash’le
        sig = json.dumps(rule["conditions"], sort_keys=True)
        if sig not in unique_signatures:
            unique_signatures.add(sig)
            cleaned.append(rule)

    return cleaned

def clean_knowledge_base(input_path, output_path, min_conf=0.5, min_feedback=1):
    with open(input_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    print(f"Orijinal kural sayısı: {len(kb['rules'])}")
    cleaned_rules = clean_rules(kb["rules"], min_conf, min_feedback)
    print(f"Temizlenmiş kural sayısı: {len(cleaned_rules)}")

    kb['rules'] = cleaned_rules

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=2)
    print(f"Yeni bilgi tabanı '{output_path}' dosyasına yazıldı.")

if __name__ == "__main__":
    clean_knowledge_base("knowledge_base.json", "knowledge_base.cleaned.json")
