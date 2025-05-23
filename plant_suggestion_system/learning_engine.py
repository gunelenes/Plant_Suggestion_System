# learning_engine.py
import os
import logging
import joblib
import pandas as pd
import pyodbc
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import json
 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --------------------------------------------------------------
# Config & Logging
# --------------------------------------------------------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# --------------------------------------------------------------
# Database Connection
# --------------------------------------------------------------
def sql_connect():
    """
    Establish connection to the SQL Server database.
    """
    return pyodbc.connect(
        r"DRIVER={ODBC Driver 17 for SQL Server};"
        r"SERVER=LAPTOP-7GK6MUOG\SQLEXPRESS;"
        r"DATABASE=Smart_Plant_Recomandation_System;"
        r"Trusted_Connection=yes;"
    )

# --------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------

def fetch_feedback_data():
    """
    Fetch all feedback records with user inputs and chosen plant.
    """
    conn = sql_connect()
    query = '''
    SELECT
        area_size,
        sunlight_need,
        environment_type,
        climate_type,
        watering_frequency,
        fertilizer_frequency,
        pesticide_frequency,
        has_pet,
        has_child,
        suggested_plant,
        user_feedback
    FROM Feedback
    '''
    df = pd.read_sql(query, conn)
    conn.close()
    logging.info(f"Fetched {len(df)} feedback records.")
    return df



def preprocess_data(df):
    X = df.drop(columns=["user_feedback"])
    y = df["user_feedback"]
    return X, y


def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"📊 Confusion matrix saved to {filename}")

def main():
   

    # 1. Model klasörü oluştur
    os.makedirs("models", exist_ok=True)

    # 2. Geri bildirim verisini yükle
    df = fetch_feedback_data()
    logging.info(f"✅ {len(df)} feedback records loaded.")

    # 3. Özellikleri ve hedef sütunu ayır
    X_raw, y = preprocess_data(df)

    # 4. OneHotEncoder + ColumnTransformer ile encode et
    categorical_cols = [
    "area_size", "sunlight_need", "environment_type", "watering_frequency"
]

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    X_encoded = column_transformer.fit_transform(X_raw)

    # 5. Eğitim/test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    negative_count = sum(y_train == 0)
    positive_count = sum(y_train == 1)
    scale_ratio = negative_count / positive_count

    # 6. Model eğitimi (XGBoost)
    model = XGBClassifier(
                        max_depth=7,
                        learning_rate=0.05,
                        n_estimators=200,
                        objective='binary:logistic',
                        eval_metric='logloss',
                        scale_pos_weight=scale_ratio,  # 🔥 Buraya eklenmeli
                        random_state=42
                    )

    model.fit(X_train, y_train)
    logging.info("✅ Model training completed.")

    # 7. Tahmin ve değerlendirme
    y_pred = model.predict(X_test)
    report_text = classification_report(y_test, y_pred)
    logging.info("📊 Classification report:\n" + report_text)

    # 8. Confusion matrix görseli oluştur
    save_confusion_matrix(y_test, y_pred)

    # 9. Rapor dosyasına yaz
    with open("last_feedback_model_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info("📝 Report saved to 'last_feedback_model_report.txt'")

    # 10. Modeli ve encode ediciyi kaydet
    joblib.dump(model, "models/feedback_model.pkl")
    joblib.dump(column_transformer, "models/feedback_vec.pkl")
    logging.info("💾 Model saved to 'models/feedback_model.pkl'")
    logging.info("💾 Features saved to 'models/feedback_vec.pkl'")

    # 11. Özellik adlarını JSON dosyasına kaydet
    with open("models/feature_names.json", "w") as f:
        feature_names = column_transformer.get_feature_names_out()
        json.dump(feature_names.tolist(), f)
    logging.info("🧠 Feature names saved to 'feature_names.json'")
    print(df["user_feedback"].value_counts())


if __name__ == "__main__":
    main()
