# learning_engine_v2.py – Association‑Rule Miner (KB‑ready, DB/CSV flexible)
# --------------------------------------------------------------
# • Madencilik: Apriori + association_rules (mlxtend)
# • Çıktı: parsed_rules.json → RuleEngine/KbUpdater şemasında
# • Yeni: DB bağlantısı opsiyonel; --csv ile offline çalışır.
# --------------------------------------------------------------

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules


try:
    import pyodbc
except ImportError:  # DB bağlantısı şart değil
    pyodbc = None

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Veri kaynakları
# --------------------------------------------------------------

def sql_connect(conn_str: str | None = None):
    """ODBC connection helper. conn_str None → env → default."""
    if not pyodbc:
        raise RuntimeError("pyodbc is not installed – install or use --csv path instead.")

    conn_str = (
        conn_str
        or os.getenv("SQLSERVER_CONN")
        or r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=LAPTOP-7GK6MUOG\\SQLEXPRESS;DATABASE=Smart_Plant_Recomandation_System;Trusted_Connection=yes;"
    )
    logger.info("ODBC connecting → %s", conn_str.split("SERVER=")[1].split(";")[0])
    return pyodbc.connect(conn_str, timeout=5)


def fetch_feedback_from_db() -> pd.DataFrame:
    conn = sql_connect()
    df = pd.read_sql("SELECT * FROM Feedback", conn)

    # Sadece feedback=1 olan kayıtları al
    df = df[df["user_feedback"] == 1]

    # Eğer 3000'den fazlaysa örnekle, değilse olduğu gibi al
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)

    conn.close()
    logger.info("✅ Fetched %d positive feedback records from DB.", len(df))
    return df


# --------------------------------------------------------------
# Canonical kategorik sütunlar
# --------------------------------------------------------------
CAT_COLS = ["area_size", "sunlight_need", "environment_type", "watering_frequency"]


# --------------------------------------------------------------
# Yardımcı fonksiyonlar
# --------------------------------------------------------------

def _split_item(item: str, cat_cols: List[str]) -> Tuple[str, str]:
    for col in cat_cols:
        prefix = f"{col}_"
        if item.startswith(prefix):
            return col, item[len(prefix) :].replace("_", " ")
    idx = item.find("_")
    return item[:idx], item[idx + 1 :].replace("_", " ")


def _parse_rules(rules_df: pd.DataFrame, feedback_flag: int) -> List[Dict]:
    parsed: List[Dict] = []
    for _, row in rules_df.iterrows():
        conds: Dict[str, List[str]] = {}
        for item in row["antecedents"]:
            col, val = _split_item(str(item), CAT_COLS)
            conds.setdefault(col, []).append(val)
        plant_item = next(x for x in row["consequents"] if str(x).startswith("suggested_plant_"))
        _, plant_name = _split_item(str(plant_item), ["suggested_plant"])
        parsed.append(
            {
                "conditions": conds,
                "suggested_plant": plant_name,
                "feedback": feedback_flag,
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            }
        )
    return parsed

# --------------------------------------------------------------
# Ana madencilik rutini
# --------------------------------------------------------------

def mine_association_rules(
    df: pd.DataFrame,
    *,
    min_support: float = 0.005,
    min_confidence: float = 0.1,
    output_path: str = "parsed_rules.json",
) -> None:
    parsed: List[Dict] = []

    def _mine(df_sub: pd.DataFrame, flag: int):
        if df_sub.empty:
            logger.info("No records for feedback=%d", flag)
            return
        trans = pd.get_dummies(df_sub[CAT_COLS].astype(str))
            # --- Apriori: en fazla 2 koşullu item-set + daha güçlü kural seçimi
        freq = fpgrowth(
            trans,
            min_support=min_support,
            use_colnames=True,
        )

     
        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=min_confidence
        )

       

        rules = rules[
            rules["consequents"].apply(lambda idx: any(str(x).startswith("suggested_plant_") for x in idx))
        ]

         # Lift’e göre sırala, ilk 500 kuralı tut
        rules = rules.sort_values("lift", ascending=False).head(100)

        logger.info("feedback=%d → %d rules after filter", flag, len(rules))
        parsed.extend(_parse_rules(rules, flag))

    _mine(df[df["user_feedback"] == 1], 1)
    _mine(df[df["user_feedback"] == 0], 0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d parsed rules → %s", len(parsed), output_path)

# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine association rules for KB")
    parser.add_argument("--csv", help="Optional CSV path instead of DB query")
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--output", default="parsed_rules.json")
    args = parser.parse_args()

    if args.csv:
        df_feedback = pd.read_csv(args.csv)
        logger.info("Loaded %d records from CSV %s", len(df_feedback), args.csv)
    else:
        try:
            df_feedback = fetch_feedback_from_db()
        except Exception as exc:
            logger.error("DB connection failed (%s). Tip: set SQLSERVER_CONN env or use --csv.", exc)
            raise SystemExit(1)

    mine_association_rules(
        df_feedback,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        output_path=args.output,
    )
