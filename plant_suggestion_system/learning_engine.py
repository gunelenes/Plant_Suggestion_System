# learning_engine.py
import os
import logging
import joblib
import pandas as pd
import pyodbc
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
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
    X_encoded = pd.get_dummies(X)
    return X_encoded, y


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
    X, y = preprocess_data(df)

    # 4. Eğitim/test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 5. Model eğitimi
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    logging.info("✅ Model training completed.")

    # 6. Tahmin ve değerlendirme
    y_pred = model.predict(X_test)
    report_text = classification_report(y_test, y_pred)
    logging.info("📊 Classification report:\n" + report_text)

    # 7. Confusion matrix görseli oluştur
    save_confusion_matrix(y_test, y_pred)

    # 8. Rapor dosyasına yaz
    with open("last_feedback_model_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info("📝 Report saved to 'last_feedback_model_report.txt'")

    # 9. Modeli ve özellik listesini kaydet
    joblib.dump(model, "models/feedback_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feedback_vec.pkl")
    logging.info("💾 Model saved to 'models/feedback_model.pkl'")
    logging.info("💾 Features saved to 'models/feedback_vec.pkl'")

if __name__ == "__main__":
    main()
