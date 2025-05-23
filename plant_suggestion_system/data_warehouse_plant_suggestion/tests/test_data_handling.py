# tests/test_data_handling.py

import sqlite3
import pandas as pd
import pytest
from data_handling import load_plants, clean_feedback_data, add_time_features

@pytest.fixture
def sqlite_connection(tmp_path):
    # In-memory SQLite veritabanı
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    # Sahte plants tablosu
    cursor.execute("""
        CREATE TABLE plants (
            plant_name TEXT
        )
    """)
    # Boş string, whitespace, None ve geçerli kayıt
    plants = [
        ("",),
        (" ",),
        (None,),
        (" aloe ",)
    ]
    cursor.executemany("INSERT INTO plants (plant_name) VALUES (?)", plants)
    conn.commit()
    yield conn
    conn.close()

def test_load_plants_cleaning(monkeypatch, sqlite_connection):
    # sql_connect() çağrısını bizim SQLite bağlantısına yönlendir
    monkeypatch.setattr("data_handling.sql_connect", lambda: sqlite_connection)
    df = load_plants()
    # 4 satır gelmeli
    assert len(df) == 4
    # Kolon adları lowercase
    assert all(col == col.lower() for col in df.columns)
    # plant_name normalize edilmiş olmalı
    assert df["plant_name"].tolist() == ["Unknown", "Unknown", "Unknown", "Aloe"]

def test_clean_feedback_nulls():
    raw_df = pd.DataFrame({
        "area_size": [None, "", "Large"],
        "sunlight_need": ["Bright", "  ", None],
        "environment_type": ["Indoor", "Outdoor", None],
        "climate_type": [None, "Summer", ""],
        "watering_frequency": ["Daily", None, ""],
        "fertilizer_frequency": [None, "Monthly", ""],
        "pesticide_frequency": ["", None, "Never"],
        "has_pet": [None, "", "Yes"],
        "has_child": ["No", None, ""],
        "suggested_plant": [None, " aloe ", ""],
        "user_feedback": ["1", "invalid", None]
    })
    clean_df = clean_feedback_data(raw_df)
    # Kategorik sütunlar dolu, string tipinde
    cat_cols = [
        "area_size", "sunlight_need", "environment_type", "climate_type",
        "watering_frequency", "fertilizer_frequency", "pesticide_frequency",
        "has_pet", "has_child", "suggested_plant"
    ]
    for col in cat_cols:
        assert all(isinstance(v, str) for v in clean_df[col])
        assert all(v != "" for v in clean_df[col])
    # user_feedback yalnızca 0 veya 1
    assert set(clean_df["user_feedback"].unique()) <= {0, 1}

def test_add_time_features_weekend():
    df = pd.DataFrame({
        "created_at": [
            "2025-05-17 10:00:00",  # Cumartesi sabah
            "2025-05-15 15:30:00",  # Perşembe öğleden sonra
            "2025-05-17 20:00:00",  # Cumartesi akşam
            "2025-05-18 02:00:00"   # Pazar gece
        ]
    })
    df_feat = add_time_features(df)
    # is_weekend doğruluğu
    assert df_feat.loc[0, "is_weekend"] == 1
    assert df_feat.loc[1, "is_weekend"] == 0
    assert df_feat.loc[2, "is_weekend"] == 1
    assert df_feat.loc[3, "is_weekend"] == 1
    # time_of_day kategorileri
    assert df_feat.loc[0, "time_of_day"] == "Morning"
    assert df_feat.loc[1, "time_of_day"] == "Afternoon"
    assert df_feat.loc[2, "time_of_day"] == "Evening"
    assert df_feat.loc[3, "time_of_day"] == "Night"
