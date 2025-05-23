# tests/test_rule_engine.py

import sys
from pathlib import Path
import json
import pandas as pd
import pytest

# Proje kök dizinini path'e ekle (tests/ klasöründeyseniz bir üst klasör)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rule_engine import RuleEngine

@pytest.fixture
def sample_plants_df():
    """Sample plants DataFrame for testing."""
    return pd.DataFrame([
        {"plant_name": "Aloe Vera",   "description": "Desc AV", "image_url": None},
        {"plant_name": "ZZ Plant",    "description": "Desc ZZ", "image_url": None},
        {"plant_name": "Snake Plant", "description": "Desc SP", "image_url": None},
    ])

@pytest.fixture
def sample_kb(tmp_path):
    """Create a temporary knowledge_base.json matching current RuleEngine."""
    kb = {
        "rules": [
            {"conditions": {"area_size": "Small"}, "suggested_plant": "Aloe Vera"},
            {"conditions": {"sunlight_need": "Shade"}, "suggested_plant": "ZZ Plant"},
        ],
        "meta_rules": [
            {
                "condition": {"environment_type": "Indoor"},
                "suggested_types": ["Succulent"],
                "excluded_types": []
            }
        ],
        "frames": {
            "Succulent": ["Snake Plant"]
        }
    }
    kb_path = tmp_path / "knowledge_base.json"
    kb_path.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(kb_path)

def test_direct_rule_match(sample_plants_df, sample_kb):
    """Exact rule in 'rules' should return only that suggested plant."""
    engine = RuleEngine(sample_plants_df, kb_path=sample_kb)
    candidates = engine.get_candidates({"area_size": "Small"}, top_n=5)
    assert candidates == ["Aloe Vera"]

def test_meta_rule_expansion(sample_plants_df, sample_kb):
    """When no direct rule, meta_rules should expand frames into candidates."""
    engine = RuleEngine(sample_plants_df, kb_path=sample_kb)
    candidates = engine.get_candidates({"environment_type": "Indoor"}, top_n=5)
    assert candidates == ["Snake Plant"]

def test_fallback_top_n(sample_plants_df, sample_kb):
    """When no rules or meta_rules match, fallback should yield top_n from all plants."""
    engine = RuleEngine(sample_plants_df, kb_path=sample_kb)
    candidates = engine.get_candidates({"unknown_feature": "value"}, top_n=2)
    expected = list(sample_plants_df["plant_name"].head(2))
    assert candidates == expected
