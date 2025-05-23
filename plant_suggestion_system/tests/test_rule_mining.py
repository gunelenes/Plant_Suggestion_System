# tests/test_rule_mining.py

import json
import pandas as pd
import pytest
from pathlib import Path
from learning_engine_v2 import mine_association_rules

@pytest.fixture
def small_feedback_df():
    # 5 kayıtlı küçük bir DataFrame, 2 pozitif, 3 negatif geri bildirim
    return pd.DataFrame({
        "area_size":            ["Small","Small","Large","Large","Small"],
        "sunlight_need":        ["Bright indirect","Bright indirect","Shade","Shade","Shade"],
        "environment_type":     ["Indoor","Indoor","Indoor","Indoor","Indoor"],
        "climate_type":         ["All seasons"]*5,
        "watering_frequency":   ["Weekly","Weekly","Monthly","Monthly","Weekly"],
        "fertilizer_frequency": ["Monthly"]*5,
        "pesticide_frequency":  ["Never needed"]*5,
        "has_pet":              ["No","No","Yes","Yes","Yes"],
        "has_child":            ["No","No","No","No","No"],
        "suggested_plant":      ["Aloe Vera","Aloe Vera","ZZ Plant","ZZ Plant","ZZ Plant"],
        "user_feedback":        [1,1,0,0,0]
    })

def test_apriori_positive_rules(tmp_path, small_feedback_df):
    out = tmp_path / "parsed_rules.json"
    mine_association_rules(
        small_feedback_df,
        min_support=0.4,      # 40% destek (ilk iki Aloe Vera kaydı)
        min_confidence=0.5,   # 50% güven
        output_path=str(out)
    )
    parsed = json.loads(out.read_text(encoding="utf-8"))
    # En az bir pozitif kural olsun
    assert any(r["feedback"] == 1 for r in parsed), "Pozitif kural bulunamadı."

def test_apriori_negative_rules(tmp_path, small_feedback_df):
    out = tmp_path / "parsed_rules.json"
    mine_association_rules(
        small_feedback_df,
        min_support=0.4,      # 40% destek (3 negatif ZZ Plant kaydı)
        min_confidence=0.5,   # 50% güven
        output_path=str(out)
    )
    parsed = json.loads(out.read_text(encoding="utf-8"))
    # En az bir negatif kural olsun
    assert any(r["feedback"] == 0 for r in parsed), "Negatif kural bulunamadı."

