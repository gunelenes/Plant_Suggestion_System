import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_handling import add_feedback

def simulate_feedback_submission():
    print("🚀 Starting test: Submitting simulated feedback...")

    user_input = {
        "area_size": "Medium",
        "sunlight_need": "Bright indirect light",
        "environment_type": "Indoor",
        "climate_type": "Spring",
        "fertilizer_frequency": "Monthly",
        "pesticide_frequency": "1-2 times a year",
        "has_pet": "Yes",
        "has_child": "No",
    }

    plant_name = "Aloe Vera"
    feedback_value = 1

    print("📤 Submitting simulated feedback...")
    add_feedback(user_input, plant_name, feedback_value)
    print("✅ Feedback successfully submitted to database.")

simulate_feedback_submission()
