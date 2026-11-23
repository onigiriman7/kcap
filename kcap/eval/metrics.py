"""Metrics on the pilot's grades format: {rule_id: {facet: first turn captured | None}}."""
import re

from kcap.schemas import PlantedContradiction, Turn

FACETS = ("action", "reason", "scope")


def first_turn_any(rule: dict):
    turns = [t for t in rule.values() if t is not None]
    return min(turns) if turns else None


def recall_curve(grades: dict, n_turns: int) -> list[float]:
    n = max(1, len(grades))
    return [sum(1 for r in grades.values() if (t := first_turn_any(r)) is not None and t <= k) / n
            for k in range(1, n_turns + 1)]


def summarize(grades: dict, n_turns: int) -> dict:
    rules = list(grades)
    n = max(1, len(rules))
    hit = [r for r in rules if first_turn_any(grades[r]) is not None]
    full = [r for r in rules if all(grades[r].get(f) is not None for f in FACETS)]
    return {
        "rule_recall": len(hit) / n,
        "full_rule_recall": len(full) / n,
        "facet_recall": {f: sum(1 for r in rules if grades[r].get(f) is not None) / n for f in FACETS},
        "rules_recovered": len(hit),
        "questions_per_rule": n_turns / len(hit) if hit else None,
        "recall_curve": recall_curve(grades, n_turns),
    }


def from_disclosure_log(rows: list[dict], rule_ids: list[str]) -> dict:
    g = {r: {f: None for f in FACETS} for r in rule_ids}
    for e in rows:
        for d in e.get("disclosed", []):
            if d.get("rule_id") in g:
                for f in d.get("facets", []):
                    if f in g[d["rule_id"]] and g[d["rule_id"]][f] is None:
                        g[d["rule_id"]][f] = e["turn"]
    return g


def contradiction_catch_rate(contras: list[PlantedContradiction], grades: dict, turns: list[Turn]) -> float | None:
    """Caught = some question cites one of the contradiction's documents AND the rule's action was captured."""
    if not contras:
        return None
    cited = set()
    for t in turns:
        cited |= set(re.findall(r"\b[A-Z]{1,4}-?\d{2,5}\b", t.question)) | set(t.meta.get("doc_ids", []))
    caught = sum(1 for k in contras
                 if set(k.doc_ids) & cited and grades.get(k.rule_id, {}).get("action") is not None)
    return caught / len(contras)


def gap_card_precision(card_map: dict) -> float | None:
    return sum(1 for v in card_map.values() if v) / len(card_map) if card_map else None


def gap_card_coverage(card_map: dict, n_rules: int) -> float:
    return len({v for v in card_map.values() if v}) / max(1, n_rules)
