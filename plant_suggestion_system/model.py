# model.py
import pandas as pd
import pyodbc
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import logging

# Logger ayarı
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MSSQL bağlantısı
def sql_connect():
    try:
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=DESKTOP-572FS4D\SQLEXPRESS;"
            "DATABASE=Smart_Plant_Recomandation_System;"
            "Trusted_Connection=yes;"
        )
        logging.info("✅ Database connection established.")
        return conn
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")
        raise

# Model eğitimi ve kaydetme
def train_and_save_model(min_samples=10):
    conn = None
    try:
        conn = sql_connect()
        df = pd.read_sql("SELECT * FROM Feedback WHERE user_feedback = 1", conn)

        if df.shape[0] < min_samples:
            logging.warning(f"⚠️ Not enough data to train the model. Minimum {min_samples} required, got {df.shape[0]}.")
            return

        required_columns = [
            "area_size", "sunlight_need", "environment_type", "climate_type",
            "fertilizer_frequency", "pesticide_frequency",
            "has_pet", "has_child", "suggested_plant", "user_feedback"
        ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column in Feedback table: {col}")

        # Özellikler ve etiketleri ayır
        feature_cols = required_columns[:-2]
        X = df[feature_cols].copy()
        y = df["suggested_plant"].copy()

        # Encode özellikler
        encoders = {}
        for col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        # Encode hedef sütun
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)

        # Model eğitimi
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y_encoded)

        # Model + dönüştürücüler birlikte kaydedilir
        model_bundle = {
            "model": model,
            "encoders": encoders,
            "target_encoder": target_encoder
        }

        joblib.dump(model_bundle, "model.pkl")
        logging.info("✅ Model trained and saved as 'model.pkl'.")
        logging.info(f"📊 Trained on {df.shape[0]} samples and {len(set(y))} unique plant classes.")

    except Exception as e:
        logging.error(f"❌ Model training failed: {e}")

    finally:
        if conn:
            conn.close()
            logging.info("🛑 Database connection closed.")

# Doğrudan çalıştırıldığında
if __name__ == "__main__":
    train_and_save_model()
