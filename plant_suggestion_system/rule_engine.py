# rule_engine.py – clean, modular rewrite
# --------------------------------------------------------------
# A) CORE IDEA
#    A *rule engine* isolates domain knowledge (kurallar) from procedural code.
#    Each rule is an IF–THEN statement encoded in JSON; the engine evaluates
#    these declarative rules against user input to build a candidate list of
#    plants before the ML‑ranking step.
#
#    Positive rules  → "ÖNER"          (feedback == 1)
#    Negative rules  → "ÖNERME"        (feedback == 0)
#    Meta‑rules      → frame ekleme/çıkarma (örn. "succulent" tipi bitkileri ekle)
#
# B) DESIGN GOALS
#    • 💡 Şeffaflık – clear & testable matching logic
#    • 🧩 Modülerlik – Rule / KnowledgeBase / RuleEngine katmanları
#    • 🐛 Debug‑ability – rich logging with rule ids & traces
# --------------------------------------------------------------

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Set

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------
# 📐 Data structures
# --------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """Immutable representation of a single IF–THEN rule."""

    conditions: Dict[str, str]
    suggested_plant: str
    feedback: int = 1
    confidence: float = 1.0
    lift: float = 1.0
    support: float = 0.0

    def matches(self, user_input: Dict[str, str]) -> bool:
        """Return True if *all* condition key/values exist in the user input."""
        return self.conditions.items() <= user_input.items()


class KnowledgeBase:
    """Load & organise rules / meta‑rules / frames from JSON."""

    def __init__(self, kb_path: str = "knowledge_base.json") -> None:
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        def _strip(rule_dict: dict) -> dict:
            return {
                "conditions":      rule_dict["conditions"],
                "suggested_plant": rule_dict["suggested_plant"],
                "feedback":        rule_dict.get("feedback", 1),
                "confidence":      rule_dict.get("confidence", 1.0),
                "lift":            rule_dict.get("lift", 1.0),
                "support":         rule_dict.get("support", 0.0),
            }


        self.positive_rules: List[Rule] = [
            Rule(**_strip(r)) for r in kb.get("positive_rules", [])
        ]
        self.negative_rules: List[Rule] = [
            Rule(**_strip(r)) for r in kb.get("negative_rules", [])
        ]

        self.meta_rules: List[dict] = kb.get("meta_rules", [])
        self.frames: Dict[str, List[str]] = kb.get("frames", {})

        logger.info(
            "KB loaded – % d positive, % d negative, % d meta‑rules, % d frames",
            len(self.positive_rules), len(self.negative_rules), len(self.meta_rules), len(self.frames)
        )


# --------------------------------------------------------------
# 🧠 Rule Engine
# --------------------------------------------------------------
class RuleEngine:
    """Kural tabanlı aday üretici katman."""

    def __init__(self, plants_df: pd.DataFrame, kb_path: str = "knowledge_base.json") -> None:
        self.plants_df = plants_df.copy()
        self.kb = KnowledgeBase(kb_path)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def get_candidates(self, user_input: Dict[str, str], top_n: int = 5) -> List[str]:
        """Return up to *top_n* plant names matching the rule logic."""

        # Step 1 – hard negative veto
        #if self._is_forbidden(user_input):
         #   logger.info("❌ User input hit a negative veto – no suggestions.")
          #  return []

        # Step 2 – exact positive match first (highest precision)
        for rule in self.kb.positive_rules:
            if rule.matches(user_input):
                logger.info("✅ Exact positive rule match → %s", rule.suggested_plant)
                return [rule.suggested_plant]

        # Step 3 – collect partial positive matches (recall)
        matches = []
        for rule in self.kb.positive_rules:
            if rule.matches(user_input):
                matches.append(rule)

        # ⚠ confidence + lift'e göre sırala
        matches.sort(key=lambda r: (r.confidence, r.lift), reverse=True)

        # ⚠ aynı bitkiden tekrar olmaması için öneri listesi oluştur
        candidates = []
        seen = set()
        for rule in matches:
            if rule.suggested_plant not in seen:
                candidates.append(rule.suggested_plant)
                seen.add(rule.suggested_plant)
            if len(candidates) >= top_n:
                break


        # Step 4 – meta‑rules (frames)
        self._apply_meta_rules(user_input, candidates, top_n)

        # Step 5 – fill‑up from the full plant list if still < top_n
        if len(candidates) < top_n:
            for plant in self.plants_df["plant_name"].unique():
                if plant not in candidates:
                    candidates.append(plant)
                if len(candidates) >= top_n:
                    break

        logger.info("Final candidate list (%d): %s", len(candidates), candidates[:top_n])
        return candidates[:top_n]

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    def _is_forbidden(self, user_input: Dict[str, str]) -> bool:
        """Return True if ANY negative rule fully matches the profile."""
        return any(r.matches(user_input) for r in self.kb.negative_rules)

    def _collect_partial_matches(self, user_input: Dict[str, str]) -> List[str]:
        """Add suggested_plant for every positive rule whose *subset* matches."""
        cands: List[str] = []
        for rule in self.kb.positive_rules:
            if rule.conditions.items() <= user_input.items():  # subset match
                if rule.suggested_plant not in cands:
                    cands.append(rule.suggested_plant)
        return cands

    def _apply_meta_rules(self, user_input: Dict[str, str], cands: List[str], top_n: int) -> None:
        """Expand / prune candidate list according to meta‑rules."""
        suggested_frames: Set[str] = set()
        excluded_frames: Set[str] = set()

        for meta in self.kb.meta_rules:
            cond = meta.get("conditions", {})
            if cond.items() <= user_input.items():
                suggested_frames.update(meta.get("suggested_types", []))
                excluded_frames.update(meta.get("excluded_types", []))

        # add suggested frame plants
        for frame in suggested_frames:
            for plant in self.kb.frames.get(frame, []):
                if plant not in cands:
                    cands.append(plant)
                if len(cands) >= top_n:
                    return

        # remove excluded frame plants
        for frame in excluded_frames:
            forbidden = set(self.kb.frames.get(frame, []))
            cands[:] = [p for p in cands if p not in forbidden]


# --------------------------------------------------------------
# 👉 Quick CLI demo
# --------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="RuleEngine demo runner")
    parser.add_argument("--kb", default="knowledge_base.json", help="Path to KB JSON")
    parser.add_argument("--csv", required=True, help="plants.csv (must have plant_name column)")
    args = parser.parse_args()

    df_plants = pd.read_csv(args.csv)
    engine = RuleEngine(df_plants, kb_path=args.kb)

    sample_profile = {
        "area_size": "Small",
        "sunlight_need": "Can live in shade",
        "environment_type": "Indoor",
        "climate_type": "All seasons",
        "watering_frequency": "Weekly",
        "fertilizer_frequency": "Monthly",
        "pesticide_frequency": "Never needed",
        "has_pet": "Yes",
        "has_child": "No",
    }
    print(engine.get_candidates(sample_profile))
