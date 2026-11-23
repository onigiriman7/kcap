from kcap.sakana_adapters.review_adapter import perform_review
from kcap.schemas import Scenario


def grade(llm, scenario: Scenario, candidate: str, cfg: dict) -> dict:
    j = cfg.get("judge", {})
    item = f"SCENARIO: {scenario.prompt}\n\nREFERENCE: {scenario.reference_answer}\n\nCANDIDATE: {candidate}"
    rev = perform_review(llm, "scenario_answer", item, "eval.grade", j.get("ensemble", 3), j.get("num_reflections", 1))
    try:
        score = max(0.0, min(2.0, float(rev.get("score", 0))))
    except (TypeError, ValueError):
        score = 0.0
    return {"score": score, "justification": rev.get("justification", ""), "item": item}
