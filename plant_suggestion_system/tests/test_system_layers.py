import json
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# MSSQL bağlantı
def sql_connect():
    import pyodbc
    return pyodbc.connect(
        r"DRIVER={ODBC Driver 17 for SQL Server};"
        r"SERVER=LAPTOP-7GK6MUOG\SQLEXPRESS;"
        r"DATABASE=Smart_Plant_Recomandation_System;"
        r"Trusted_Connection=yes;"
    )

# RuleEngine sınıfı (güncellenmiş eşleşme)
class RuleEngine:
    def __init__(self, plants_df, kb_path="knowledge_base.json"):
        self.plants = plants_df
        with open(kb_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f).get("rules", [])

    def recommend(self, user_input):
        for rule in self.rules:
            if rule["conditions"].items() <= user_input.items():
                match = self.plants[
                    self.plants["plant_name"].str.lower() == rule["suggested_plant"].lower()
                ]
                if not match.empty:
                    return match
        return pd.DataFrame()

# Test 1: DB Bağlantısı
def test_database_connection():
    try:
        conn = sql_connect()
        df = pd.read_sql("SELECT TOP 1 * FROM Feedback", conn)
        conn.close()
        return "✅ Veritabanı bağlantısı çalışıyor."
    except Exception as e:
        return f"❌ Veritabanı bağlantı hatası: {e}"

# Test 2: RuleEngine çalışıyor mu?
def test_rule_engine():
    try:
        conn = sql_connect()
        df = pd.read_sql("SELECT TOP 100 * FROM Plants", conn)
        conn.close()

        rule_engine = RuleEngine(df)
        dummy_input = {
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

        result = rule_engine.recommend(dummy_input)
        print("📌 Kullanılan dummy input:", dummy_input)

        if not result.empty:
            print("🔍 Eşleşen Bitki:", result["plant_name"].values[0])
            return "✅ RuleEngine öneri üretebiliyor."
        else:
            return "⚠️ RuleEngine çalışıyor ama eşleşen bitki bulamadı."
    except Exception as e:
        return f"❌ RuleEngine hatası: {e}"

# Test 3: Binary ML Model Eğitimi
def test_learning_engine_v2_model():
    try:
        conn = sql_connect()
        df = pd.read_sql("SELECT * FROM Feedback", conn)
        conn.close()
        X = df.drop(columns=["user_feedback"])
        y = df["user_feedback"]
        X_encoded = pd.get_dummies(X)
        model = DecisionTreeClassifier(max_depth=5).fit(X_encoded, y)
        y_pred = model.predict(X_encoded)
        acc = accuracy_score(y, y_pred)
        return f"✅ Binary model eğitimi başarılı. Accuracy: {acc:.2f}"
    except Exception as e:
        return f"❌ Binary model eğitimi hatası: {e}"

# Test 4: Model Dosyası Yüklenebilir mi?
def test_model_pickle_load():
    try:
        if not os.path.exists("feedback_model.pkl"):
            return "❌ feedback_model.pkl dosyası bulunamadı."
        bundle = joblib.load("feedback_model.pkl")
        if "model" in bundle and "features" in bundle:
            return "✅ feedback_model.pkl başarıyla yüklendi."
        return "⚠️ feedback_model.pkl eksik içerik içeriyor."
    except Exception as e:
        return f"❌ feedback_model.pkl yüklenemedi: {e}"

# Test 5: KB Integrity
def test_kb_json_integrity():
    try:
        with open("knowledge_base.json", encoding="utf-8") as f:
            kb = json.load(f)
        rules = kb.get("rules", [])
        assert isinstance(rules, list)
        auto_rules = [r for r in rules if r["suggested_plant"] in ["", "AUTO-GENERATED"]]
        if auto_rules:
            return f"⚠️ {len(auto_rules)} kuralın suggested_plant alanı boş."
        return "✅ knowledge_base.json dosyası geçerli ve eksiksiz."
    except Exception as e:
        return f"❌ knowledge_base.json hatası: {e}"

# Ana test çalıştırıcı
def run_all_tests():
    tests = [
        test_database_connection(),
        test_rule_engine(),
        test_learning_engine_v2_model(),
        test_model_pickle_load(),
        test_kb_json_integrity()
    ]
    print("\n🧪 Sistem Katman Test Sonuçları:")
    for i, result in enumerate(tests, 1):
        print(f"{i}. {result}")

if __name__ == "__main__":
    run_all_tests()
