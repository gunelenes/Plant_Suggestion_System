# app.py – Smart Plant Recommendation System (REDESIGNED VERSION)
# --------------------------------------------------------------
# This version features:
# 1. Two-column layout for better space utilization
# 2. Enhanced UI with custom styling and visual improvements
# 3. Maintains ALL original backend logic and functionality
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
import numpy as np
import random  
import streamlit as st
from typing import Tuple
from data_handling import load_plants, add_feedback, sql_connect
from rule_engine import RuleEngine

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Feedback modeli ve vectorizer modül yüklemesi (bir kez çalışır)
feedback_model = joblib.load("models/feedback_model.pkl")
feedback_vec   = joblib.load("models/feedback_vec.pkl")


# --------------------------------------------------------------
# 🔧 Page / general config
# --------------------------------------------------------------
st.set_page_config(
    page_title="Smart Plant Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8fef8;
    }
    
    /* Header styling */
    h1 {
        color: #2e7d32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    h3 {
        text-align: center;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.75rem;
        text-transform: uppercase;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #2e7d32;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Form elements */
    .stSelectbox label, .stRadio label {
        font-weight: 500;
        color: #1b5e20;
    }
    
    /* Plant card */
    .plant-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
    }
    
    /* Feedback section */
    .feedback-section {
        border-top: 1px solid #e0e0e0;
        padding-top: 1rem;
        margin-top: 1rem;
    }
    
    /* Divider */
    .divider {
        background-color: #4CAF50;
        height: 3px;
        margin: 1rem 0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Header with decorative elements
st.title("🌿 Smart Plant Recommendation System")
st.markdown("### 🌱 Fill the form to get your best‑matching plant:")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------------------
# 🔁 Automatic retrain helper - KEEPING ORIGINAL CODE
# --------------------------------------------------------------

def check_and_retrain_if_needed(threshold: int = 3) -> bool:
    """Retrain model when accepted‑feedback count is divisible by *threshold*."""
    logger.debug("Retrain ihtiyacı kontrol ediliyor (threshold: %d)", threshold)
    try:
        conn = sql_connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Feedback WHERE user_feedback = 1")
        count = cur.fetchone()[0]
        conn.close()
        logger.info("Toplam pozitif feedback: %s", count)

        if count and count % threshold == 0:
            st.info(f"🔁 Retraining ML model… (total positive feedback: {count})")
            logger.warning("Retrain başlatıldı.")
            
    
             
            subprocess.run([sys.executable, "learning_engine.py"], check=True)
            logger.info("learning_engine.py çalıştırıldı.")
            
            subprocess.run([sys.executable, "learning_engine_v2.py"], check=True)
            logger.info("learning_engine_v2.py çalıştırıldı.")

           


               # 4. KB güncelle
            from kb_updater import update_knowledge_base
            update_knowledge_base("parsed_rules.json", "knowledge_base.json")
            logger.info("Bilgi tabanı güncellendi.")

            st.success("✅ Model retrained & rules updated.")
            return True


            

            
    except Exception as exc:  # pragma: no cover
        logger.error("Retrain sırasında hata: %s", exc)
        st.error(f"❌ Retrain check failed: {exc}")
    return False

# --------------------------------------------------------------
# 📝 User input in two-column layout
# --------------------------------------------------------------

def render_preference_form():
    """Return a dict with the user selections in a two-column layout."""
    
    # Create two columns for the form fields
    col1, col2 = st.columns(2)
    
    with col1:
        area_size = st.selectbox("Space Size", ["Mini", "Small", "Medium", "Large"])
        sunlight_need = st.selectbox(
            "Sunlight Requirement",
            ["Can live in shade", "1-2 hours daily", "Bright indirect light", "6+ hours"],
        )
        environment_type = st.selectbox("Environment Type", ["Indoor", "Outdoor", "Semi-outdoor"])
        climate_type = st.selectbox("Climate Type", ["All seasons", "Spring", "Summer", "Winter"])
        watering_frequency = st.selectbox(
            "Watering Frequency",
            ["Daily", "Weekly", "Bi-weekly", "Every 2-3 days", "Monthly"]
        )
    
    with col2:
        fertilizer_frequency = st.selectbox(
            "Fertilizer Frequency", ["Monthly", "1-2 times a year", "Never needed"]
        )
        pesticide_frequency = st.selectbox(
            "Pesticide Frequency", ["Monthly", "1-2 times a year", "Never needed"]
        )
        
        has_pet = st.radio("Do you have pets?", ["Yes", "No"])
        has_child = st.radio("Do you have children?", ["Yes", "No"])
        

        
    return {
        "area_size": area_size,
        "sunlight_need": sunlight_need,
        "environment_type": environment_type,
        "climate_type": climate_type,
        "fertilizer_frequency": fertilizer_frequency,
        "pesticide_frequency": pesticide_frequency,
        "has_pet": has_pet,
        "has_child": has_child,
        "watering_frequency": watering_frequency
    }

user_input = render_preference_form()
logger.info("📥 Kullanıcı inputu alındı: %s", user_input)
# Centered button with distinct styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_clicked = st.button("🌿 Recommend Plant")

# Eğer kullanıcı öner butonuna bastıysa
if recommend_clicked:
    logger.info(" Öner butonuna tıklandı.")
    df = load_plants()
    if df.empty:
        logger.error(" Bitki verisi yüklenemedi.")
        st.error(" Could not load plant data — check DB connection.")
        st.stop()

    rule_engine = RuleEngine(df)
    candidates = rule_engine.get_candidates(user_input, top_n=200)
    logger.info("🎯 RuleEngine aday bitkiler: %s", candidates)

    logger.info("🧠 ML skorlama başlatılıyor. %d aday değerlendirilecek.", len(candidates))

    # ML skorlama yapılacak bitki listesi (fallback destekli)
    scores = []

    if not candidates:
        logger.warning("⚠️ Kural tabanlı eşleşme bulunamadı, ML fallback başlatılıyor.")
        st.info("🔍 No rule-based match found. Trying best guess with ML...")
        for _, row in df.iterrows():
            record = {**user_input, "suggested_plant": row["plant_name"]}
            feature_values = [record.get(feat, 0) for feat in feedback_vec]
            x_vec = np.array([feature_values])
            proba = feedback_model.predict_proba(x_vec)[0, 1]
            scores.append((row["plant_name"], proba))

            logger.debug("🔢 ML skor: %s → %.3f",  proba)

    else:
        for plant in candidates:
            record = {**user_input, "suggested_plant": plant}
            feature_values = [record.get(feat, 0) for feat in feedback_vec]
            x_vec = np.array([feature_values])
            proba = feedback_model.predict_proba(x_vec)[0, 1]
            scores.append((plant, proba))
            logger.debug("🔢 ML skor: %s → %.3f", plant, proba)


        # Eşleşme veritabanında yoksa fallback için tekrar tüm df taranır
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        fallback_row = None
        for plant, score in scores:
            match = df[df["plant_name"] == plant]
            if not match.empty:
                fallback_row = match.iloc[0]
                best_plant = plant
                best_score = score
                logger.info("✅ Kural + ML ile bulunan öneri: %s (skor: %.3f)", best_plant, best_score)
                break
        if fallback_row is None:
            st.info("⚠️ Candidates found but not in DB. ML fallback triggered.")
            scores.clear()
            for _, row in df.iterrows():
                record = {**user_input, "suggested_plant": row["plant_name"]}
                feature_values = [record.get(feat, 0) for feat in feedback_vec]
                x_vec = np.array([feature_values])
                proba = feedback_model.predict_proba(x_vec)[0, 1]
                scores.append((row["plant_name"], proba))

    # En iyi sonucu bul ve göster
    if scores:
                

        TOP_K = 5  # ilk 5 yüksek skorlu bitki arasından seçeceğiz

        top_k = scores[:TOP_K] if len(scores) >= TOP_K else scores

        # Geçmiş önerilen bitkiler tutulur (session bazlı)
        past = st.session_state.get("past_recommendations", [])

        # Daha önce önerilmemiş olanı rastgele seç
        suggestion = None
        for plant, score in random.sample(top_k, len(top_k)):
            if plant not in past:
                suggestion = (plant, score)
                past.append(plant)
                break

        st.session_state["past_recommendations"] = past

        if suggestion is None:
            st.warning("🔁 All top suggestions already shown. Try different input.")
            st.stop()

        best_plant, best_score = suggestion

        logger.info("✅ En iyi öneri: %s (Skor: %.3f)", best_plant, best_score)

        row = df[df["plant_name"] == best_plant].iloc[0]
        logger.debug("📄 Bitki veri satırı bulundu: %s", row.to_dict())

        img_url = row["image_url"]

        st.session_state["recommended_plant"] = {
            "plant_name": best_plant,
            "description": row["description"],
            "image_url": img_url,
        }
        st.session_state["user_input"] = user_input
    else:
        st.error("❌ Unable to generate a recommendation.")
        st.stop()

# Eğer tavsiye varsa göster
plant_dict = st.session_state.get("recommended_plant")
if plant_dict:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"<div class='plant-card'>", unsafe_allow_html=True)
        st.markdown(f"### {plant_dict['plant_name']}")
        if pd.notna(plant_dict.get("image_url")):
            st.image(plant_dict["image_url"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='plant-card'>", unsafe_allow_html=True)
        if not recommend_clicked:
            st.success(f"✅ Recommended Plant: **{plant_dict['plant_name']}**")

        st.markdown("### 📄 Description:")
        description_html = f"<div style='font-size:20px; line-height:1.6'>{plant_dict['description']}</div>"
        st.markdown(description_html, unsafe_allow_html=True)


        # Geri bildirim formu
        with st.form("feedback_form"):
            fb_choice = st.radio("💬 Was this recommendation suitable for you?", ["Yes", "No"])
            submit = st.form_submit_button("📩 Submit Feedback")

        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        try:
            feedback_val = 1 if fb_choice == "Yes" else 0
            add_feedback(
                st.session_state["user_input"],
                plant_dict["plant_name"],
                feedback_val,
            )
            st.success("🎉 Feedback saved. Thank you!")
            check_and_retrain_if_needed()
            st.session_state.pop("recommended_plant", None)
            st.session_state.pop("user_input", None)
        except Exception as exc:
            st.error(f"❌ Failed to save feedback: {exc}")
