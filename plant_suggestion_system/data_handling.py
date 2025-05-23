# data_handling.py
import pandas as pd
import pyodbc
import logging

# Logger konfigürasyonu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SQL Server bağlantısı
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

# Tüm bitki verilerini getir
def load_plants():
    conn = None
    try:
        conn = sql_connect()
        query = "SELECT * FROM plants"
        df = pd.read_sql(query, conn)
        logging.info(f"✅ {len(df)} plant records loaded.")
        return df
    except Exception as e:
        logging.error(f"❌ Failed to load plant data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
            logging.info("🛑 Database connection closed after loading plants.")

def add_feedback(user_input, plant_name, feedback):
    try:
        print("🚀 add_feedback() fonksiyonu çağrıldı.") 
        conn = sql_connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO Feedback (
            area_size, sunlight_need, environment_type, climate_type,
            fertilizer_frequency, pesticide_frequency,
            has_pet, has_child, suggested_plant, user_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            user_input.get("area_size"),
            user_input.get("sunlight_need"),
            user_input.get("environment_type"),
            user_input.get("climate_type"),
            user_input.get("fertilizer_frequency"),
            user_input.get("pesticide_frequency"),
            user_input.get("has_pet"),
            user_input.get("has_child"),
            plant_name,
            feedback
        )

        print("📥 Executing SQL INSERT with values:", values)  # LOG
        cursor.execute(query, values)
        conn.commit()
        print("✅ Feedback successfully inserted.")

    except Exception as e:
        print("❌ Failed to insert feedback:", e)
        import traceback
        traceback.print_exc()  # ← buraya

    

    finally:
        if conn:
            conn.close()
            logging.info("🛑 Database connection closed after inserting feedback.")
