import json
import pytest
from pathlib import Path
import kb_updater

@pytest.fixture
def tmp_parsed_and_kb(tmp_path):
    # 1) Create a parsed_rules.json with mixed feedback
    parsed = [
        {"conditions": {"foo": ["bar"]}, "suggested_plant": "PlantA", "feedback": 1},
        {"conditions": {"baz": ["qux"]}, "suggested_plant": "PlantB", "feedback": 0},
        {"conditions": {"alpha": ["beta"]}, "suggested_plant": "PlantC", "feedback": 1},
    ]
    parsed_path = tmp_path / "parsed_rules.json"
    parsed_path.write_text(json.dumps(parsed), encoding="utf-8")

    # 2) Create a minimal knowledge_base.json
    initial_kb = {
        "meta_rules": [],
        "frames": {},
        "rules": []
    }
    kb_path = tmp_path / "knowledge_base.json"
    kb_path.write_text(json.dumps(initial_kb), encoding="utf-8")

    return str(parsed_path), str(kb_path)

def test_update_kb_rules_split(tmp_parsed_and_kb):
    parsed_path, kb_path = tmp_parsed_and_kb

    # Call the updater function
    kb_updater.update_knowledge_base(parsed_path=parsed_path, kb_path=kb_path)

    # Reload the updated KB
    kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))

    # It must now have separate lists
    assert "positive_rules" in kb and isinstance(kb["positive_rules"], list)
    assert "negative_rules" in kb and isinstance(kb["negative_rules"], list)

    # All feedback=1 should be in positive_rules
    pos = kb["positive_rules"]
    assert all(r.get("feedback") == 1 for r in pos)
    assert len(pos) == 2

    # All feedback=0 should be in negative_rules
    neg = kb["negative_rules"]
    assert all(r.get("feedback") == 0 for r in neg)
    assert len(neg) == 1
