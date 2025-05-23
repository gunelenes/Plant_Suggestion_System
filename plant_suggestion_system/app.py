# app.py – Smart Plant Recommendation System (FULL REVISED VERSION)
# --------------------------------------------------------------
# This version fixes the feedback‑not‑saved issue by
# 1. Using st.session_state to persist user_input & plant across reruns
# 2. Wrapping the feedback UI in st.form so a single submit event is captured
# 3. Keeping a clear separation of "recommend" vs "feedback" phases
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
from data_handling import load_plants, add_feedback, sql_connect

# --------------------------------------------------------------
# 🔧 Page / general config
# --------------------------------------------------------------
st.set_page_config(page_title="Smart Plant Recommender", layout="centered")
st.title("🌿 Smart Plant Recommendation System")
st.markdown("### 🌱 Fill the form to get your best‑matching plant:")

# --------------------------------------------------------------
# 🔁 Automatic retrain helper
# --------------------------------------------------------------

def check_and_retrain_if_needed(threshold: int = 3) -> bool:
    """Retrain model when accepted‑feedback count is divisible by *threshold*."""
    try:
        conn = sql_connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Feedback WHERE user_feedback = 1")
        count = cur.fetchone()[0]
        conn.close()

        if count and count % threshold == 0:
            st.info(f"🔁 Retraining ML model… (total positive feedback: {count})")
            subprocess.run([".venv\\Scripts\\python.exe", "model.py"], check=True)
            

            st.success("✅ Model retrained & model.pkl updated.")
            return True
    except Exception as exc:  # pragma: no cover
        st.error(f"❌ Retrain check failed: {exc}")
    return False

# --------------------------------------------------------------
# 📝 User input (always visible)
# --------------------------------------------------------------

def render_preference_form():
    """Return a dict with the user selections."""
    return {
        "area_size": st.selectbox("Space Size", ["Mini", "Small", "Medium", "Large"]),
        "sunlight_need": st.selectbox(
            "Sunlight Requirement",
            ["Can live in shade", "1-2 hours daily", "Bright indirect light", "6+ hours"],
        ),
        "environment_type": st.selectbox("Environment Type", ["Indoor", "Outdoor", "Semi-outdoor"]),
        "climate_type": st.selectbox("Climate Type", ["All seasons", "Spring", "Summer", "Winter"]),
        "fertilizer_frequency": st.selectbox(
            "Fertilizer Frequency", ["Monthly", "1-2 times a year", "Never needed"]
        ),
        "pesticide_frequency": st.selectbox(
            "Pesticide Frequency", ["Monthly", "1-2 times a year", "Never needed"]
        ),
        "has_pet": st.radio("Do you have pets?", ["Yes", "No"]),
        "has_child": st.radio("Do you have children?", ["Yes", "No"]),
    }

user_input = render_preference_form()
recommend_clicked = st.button("🌿 Recommend Plant")

# --------------------------------------------------------------
# 🎯 Recommendation phase
# --------------------------------------------------------------
if recommend_clicked:
    df = load_plants()
    if df.empty:
        st.error("❌ Could not load plant data — check DB connection.")
        st.stop()

    # 1️⃣ Rule‑based exact match
    mask = (
        (df["area_size"] == user_input["area_size"]) &
        (df["sunlight_need"] == user_input["sunlight_need"]) &
        (df["environment_type"] == user_input["environment_type"]) &
        (df["climate_type"] == user_input["climate_type"]) &
        (df["fertilizer_frequency"] == user_input["fertilizer_frequency"]) &
        (df["pesticide_frequency"] == user_input["pesticide_frequency"]) &
        (df["has_pet"] == user_input["has_pet"]) &
        (df["has_child"] == user_input["has_child"])  # noqa: E501
    )

    if df[mask].shape[0] > 0:
        plant_row = df[mask].sample(1).iloc[0]
        st.info("🎯 Exact match found — rule‑based filter.")
    else:
        st.warning("⚠️ No exact match — falling back to ML prediction…")
        if not os.path.exists("model.pkl"):
            st.error("❌ model.pkl not found — run model.py first.")
            st.stop()
        try:
            bundle = joblib.load("model.pkl")
            model = bundle["model"]
            encoders = bundle["encoders"]
            tgt_enc = bundle["target_encoder"]

            x = pd.DataFrame([user_input])
            for col in x.columns:
                x[col] = encoders[col].transform(x[col])
            pred_encoded = model.predict(x)[0]
            pred_label = pred_encoded if isinstance(pred_encoded, str) else tgt_enc.inverse_transform([pred_encoded])[0]
            if df[df["plant_name"] == pred_label].empty:
                st.error(f"❌ Predicted plant '{pred_label}' not in DB.")
                st.stop()
            plant_row = df[df["plant_name"] == pred_label].iloc[0]
            st.info("🧠 Recommendation provided by ML model.")
        except Exception as exc:  # pragma: no cover
            st.error(f"❌ ML prediction failed: {exc}")
            st.stop()

    # Store to session so next rerun (after form submit) still has context
    st.session_state["recommended_plant"] = plant_row.to_dict()
    st.session_state["user_input"] = user_input

# --------------------------------------------------------------
# 📩 Feedback phase (only if we have a plant in session)
# --------------------------------------------------------------
plant_dict = st.session_state.get("recommended_plant")
if plant_dict:
    # Show recommendation card (only when not just shown in same run)
    if not recommend_clicked:
        st.success(f"✅ Recommended Plant: **{plant_dict['plant_name']}**")
        if pd.notna(plant_dict.get("image_url")):
            st.image(plant_dict["image_url"], width=250)
        st.write("📄 Description:")
        st.markdown(f"> {plant_dict['description']}")

    # ---- Feedback Form ----
    with st.form("feedback_form"):
        fb_choice = st.radio("💬 Was this recommendation suitable for you?", ["Yes", "No"])
        fb_submit = st.form_submit_button("📩 Submit Feedback")

    if fb_submit:
        try:
            print("🚨 Feedback form submitted → saving to DB…")
            feedback_val = 1 if fb_choice == "Yes" else 0
            add_feedback(
                st.session_state["user_input"],
                plant_dict["plant_name"],
                feedback_val,
            )
            st.success("🎉 Feedback saved. Thank you!")
            # Check retraining condition
            check_and_retrain_if_needed()
            # Clear stored plant so user can search again cleanly
            st.session_state.pop("recommended_plant", None)
            st.session_state.pop("user_input", None)
        except Exception as exc:  # pragma: no cover
            st.error(f"❌ Failed to save feedback: {exc}")
