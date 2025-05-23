# model_test.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Test verisi (CSV formatından simüle edilmiş)
data = [
    ["Medium", "Can live in shade", "Indoor", "All seasons", "1-2 times a year", "Never needed", "Yes", "No", "Snake Plant (Sansevieria)", 0],
    ["Small", "1-2 hours daily", "Indoor", "All seasons", "1-2 times a year", "Never needed", "Yes", "Yes", "Jade Plant", 1],
    ["Small", "Bright indirect light", "Indoor", "All seasons", "1-2 times a year", "Never needed", "No", "No", "Aloe Vera", 0],
    ["Medium", "Bright indirect light", "Indoor", "Spring", "Monthly", "1-2 times a year", "No", "No", "Boston Fern", 0],
    ["Medium", "Can live in shade", "Indoor", "All seasons", "1-2 times a year", "Never needed", "Yes", "No", "Snake Plant (Sansevieria)", 0],
]

columns = [
    "area_size", "sunlight_need", "environment_type", "climate_type",
    "fertilizer_frequency", "pesticide_frequency",
    "has_pet", "has_child", "suggested_plant", "user_feedback"
]

df = pd.DataFrame(data, columns=columns)

# Model dosyasını yükle
model_bundle = joblib.load("model.pkl")
model = model_bundle["model"]
encoders = model_bundle["encoders"]
target_encoder = model_bundle["target_encoder"]

# Özellik ve hedef değişkenleri ayır
feature_cols = [
    "area_size", "sunlight_need", "environment_type", "climate_type",
    "fertilizer_frequency", "pesticide_frequency",
    "has_pet", "has_child"
]

X = df[feature_cols].copy()
y = df["suggested_plant"].copy()

# Encode işlemi
for col in X.columns:
    if col in encoders:
        X[col] = encoders[col].transform(X[col])
    else:
        raise ValueError(f"Encoder missing for column: {col}")

# y encode
y_encoded = target_encoder.transform(y)

# Tahmin
y_pred = model.predict(X)

# Başarı oranı
accuracy = accuracy_score(y_encoded, y_pred)
print(f"\n✅ Model test accuracy on test data: {accuracy:.2f}")

# Örnek çıktı
for i in range(len(X)):
    predicted = target_encoder.inverse_transform([y_pred[i]])[0]
    print(f"Sample {i+1} | Predicted: {predicted} | Actual: {y.iloc[i]}")
