# model_debug.py
import joblib
import pandas as pd

# Model dosyasını yükle
model_bundle = joblib.load("model.pkl")
model = model_bundle["model"]
encoders = model_bundle["encoders"]
target_encoder = model_bundle["target_encoder"]

print("\n✅ Model file successfully loaded.")

# 1. Tahmin edilebilir sınıf etiketlerini yazdır
print("\n🔍 Model Target Classes:")
print(list(target_encoder.classes_))

# 2. Tüm encoder'ların içeriklerini göster
print("\n📦 Encoders for Input Features:")
for feature, encoder in encoders.items():
    print(f"\n- {feature}:")
    classes = list(encoder.classes_)
    print(classes)

# 3. Örnek tahmin testi (isteğe bağlı)
example_input = {
    "area_size": "Small",
    "sunlight_need": "Bright indirect light",
    "environment_type": "Indoor",
    "climate_type": "All seasons",
    "fertilizer_frequency": "1-2 times a year",
    "pesticide_frequency": "Never Needed",
    "has_pet": "Yes",
    "has_child": "No"
}

print("\n🧪 Test Prediction Input:")
print(example_input)

try:
    df_input = pd.DataFrame([example_input])
    for col in df_input.columns:
        df_input[col] = encoders[col].transform(df_input[col])

    pred_encoded = model.predict(df_input)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]

    print(f"\n✅ Prediction successful: {pred_label}")

except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
