import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pyodbc
from sklearn.metrics import roc_curve, roc_auc_score

# -------------------------------
# MSSQL bağlantısı
# -------------------------------
def sql_connect():
    return pyodbc.connect(
        r"DRIVER={ODBC Driver 17 for SQL Server};"
        r"SERVER=localhost\SQLEXPRESS;"
        r"DATABASE=Smart_Plant_Recomandation_System;"
        r"Trusted_Connection=yes;"
    )

def load_feedback_data():
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
    return df

# -------------------------------
# ROC-AUC çizimi
# -------------------------------
def plot_roc_auc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_auc_curve.png")
    plt.show()

# -------------------------------
# Ana akış
# -------------------------------
if __name__ == "__main__":
    # Model ve vectorizer yükle
    model = joblib.load("models/feedback_model.pkl")
    vectorizer = joblib.load("models/feedback_vec.pkl")

    # Veriyi yükle ve vektörle
    df = load_feedback_data()
    X = df.drop(columns=["user_feedback"])
    y = df["user_feedback"]
    X_encoded = vectorizer.transform(X.to_dict(orient="records"))
    # Eğitimdeki feature sayısıyla testteki aynı mı? Kontrol et
    import json
    with open("models/feature_names.json", "r") as f:
        trained_features = json.load(f)

    print("🎯 Eğitimdeki feature sayısı:", len(trained_features))
    print("🧪 Testteki feature sayısı:", X_encoded.shape[1])

    if len(trained_features) != X_encoded.shape[1]:
        print("🚨 UYARI: Özellik sayısı uyuşmuyor! Model hatalı çalışabilir.")


    # (1) Vektör çeşitliliği kontrolü
    import numpy as np
    unique_vectors = np.unique(X_encoded, axis=0)
    print("Farklı vektör sayısı:", unique_vectors.shape[0])

    # (2) Olasılık çıktıları kontrolü
    y_proba = model.predict_proba(X_encoded)[:, 1]
    print("İlk 10 tahmin edilen olasılık:", y_proba[:10])

    # ROC-AUC görselleştir
    plot_roc_auc(y, y_proba)
