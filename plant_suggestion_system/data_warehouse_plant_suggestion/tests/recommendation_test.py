# recommendation_test.py (Final Version)
import json
import pandas as pd
import joblib

# 1. Örnek kullanıcı verisi
user_input = {
    "area_size": "Small",
    "sunlight_need": "Can live in shade",
    "environment_type": "Indoor",
    "climate_type": "All seasons",
    "watering_frequency": "Weekly",
    "fertilizer_frequency": "Monthly",
    "pesticide_frequency": "Never needed",
    "has_pet": "Yes",
    "has_child": "No"
}

# 2. RuleEngine benzeri öneri sistemi
def recommend_from_rules(user_input):
    with open("knowledge_base.json", encoding="utf-8") as f:
        rules = json.load(f).get("rules", [])
    
    for rule in rules:
        if rule["conditions"].items() <= user_input.items():
            explanation = f"Kural eşleşti: {rule['conditions']}"
            return rule["suggested_plant"], explanation
    return None, "Hiçbir kural eşleşmedi."

# 3. ML tahmini (kullanıcı beğenir mi?)
def predict_feedback(user_input, suggested_plant):
    user_input["suggested_plant"] = suggested_plant
    x = pd.DataFrame([user_input])
    x_encoded = pd.get_dummies(x)

    # Modeli ve özellikleri yükle
    bundle = joblib.load("feedback_model.pkl")
    model = bundle["model"]
    features = bundle["features"]

    # Eksik sütunları sıfırla (tek adımda)
    missing_cols = [col for col in features if col not in x_encoded.columns]
    full_input = pd.concat([x_encoded, pd.DataFrame([{col: 0 for col in missing_cols}])], axis=1)
    full_input = full_input[features].copy()

    prediction = model.predict(full_input)[0]
    prob = model.predict_proba(full_input)[0][1]
    result = "Beğenilir ✅" if prediction == 1 else "Beğenilmez ❌"
    return result, f"Tahmin: {result} (Güven: {prob:.2f})"

# 4. Ana test fonksiyonu
def main():
    plant, explanation = recommend_from_rules(user_input)
    print("📌 Kullanıcı bilgileri:", user_input)
    print("🔍 Açıklama:", explanation)

    if plant:
        print(f"🌿 Önerilen Bitki: {plant}")
        result, score = predict_feedback(user_input, plant)
        print(f"🧠 ML Değerlendirmesi: {score}")
    else:
        print("❌ Kural tabanlı sistemden öneri çıkmadı.")

if __name__ == "__main__":
    main()
