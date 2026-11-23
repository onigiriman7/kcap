# Adapted from SakanaAI/AI-Scientist-v2 ai_scientist/perform_llm_review.py@96bd516
# (perform_review: few-shot form -> ensemble of reviews -> aggregate -> reflection rounds).
# Kept: structure, ensemble + aggregation, reflection prompt with "I am done" early stop.
# Changed: paper-review form -> our rubrics; meta-review LLM call -> deterministic majority/mean;
#          few-shot examples come from config/judge_examples/<rubric>.json (owner-written).
import json
from collections import Counter

from kcap.io import CONFIG_DIR
from kcap.llm.client import parse_json

RUBRICS = {
    "scenario_answer": {
        "system": "You are a strict grader of answers to realistic workplace scenarios.",
        "form": (
            "Grade the CANDIDATE answer against the REFERENCE answer.\n"
            "score 0 = wrong, harmful, or misses the key point; 1 = partly right (right direction, missing a key "
            "condition/exception); 2 = matches the reference in substance (wording may differ).\n"
            'Return JSON {"score": 0|1|2, "justification": "<one sentence>"}.'
        ),
        "mean": ["score"], "majority": [],
    },
    "rule_match": {
        "system": "You check whether a hidden expert rule was captured in a list of knowledge units (KUs).",
        "form": (
            "For EACH facet of the HIDDEN RULE (action, reason, scope), decide whether the KUs capture it.\n"
            "captured = a KU states the same content (paraphrase OK, key specifics present); partially = related "
            "but missing a key specific; not = absent or contradicted.\n"
            'Return JSON {"action": {"verdict": "captured|partially|not", "ku_id": "<id or null>"}, '
            '"reason": {...}, "scope": {...}}.'
        ),
        "mean": [], "majority": ["action.verdict", "reason.verdict", "scope.verdict"],
    },
    "gap_card_match": {
        "system": "You check whether a hypothesis about undocumented practice corresponds to a known hidden rule.",
        "form": (
            "Given a HYPOTHESIS and a numbered list of RULES, return the id of the rule the hypothesis is about "
            "(same situation and same kind of action; it may be partly wrong in details), or null if none.\n"
            'Return JSON {"rule_id": "<id or null>", "justification": "<one sentence>"}.'
        ),
        "mean": [], "majority": ["rule_id"],
    },
}

REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the accuracy of the judgment you just made against the rubric.
Do not make things overly complicated. Stick to the original judgment unless there are glaring issues.
Respond with THOUGHT: <your thought> then the JSON.
If there is nothing to improve, repeat the previous JSON EXACTLY and include "I am done" in the thought."""


def fewshot(rubric: str, k: int = 5) -> str:
    p = CONFIG_DIR / "judge_examples" / f"{rubric}.json"
    if not p.exists():
        return ""
    ex = json.loads(p.read_text(encoding="utf-8"))[:k]
    return "\n\nExamples:\n" + "\n\n".join(
        f"INPUT:\n{e['input']}\nOUTPUT:\n{json.dumps(e['output'])}" for e in ex)


def _get(d, path):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _set(d, path, v):
    ks = path.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = v


def _aggregate(rubric: str, reviews: list[dict]) -> dict:
    spec = RUBRICS[rubric]
    out = json.loads(json.dumps(reviews[0]))
    for f in spec["majority"]:
        votes = [json.dumps(_get(r, f)) for r in reviews]
        winner = json.loads(Counter(votes).most_common(1)[0][0])
        _set(out, f, winner)
        # keep the ku_id from a review that agrees with the majority verdict
        if f.endswith(".verdict"):
            base = f.rsplit(".", 1)[0]
            for r in reviews:
                if _get(r, f) == winner:
                    _set(out, base + ".ku_id", _get(r, base + ".ku_id"))
                    break
    for f in spec["mean"]:
        vals = [float(_get(r, f)) for r in reviews if isinstance(_get(r, f), (int, float))]
        if vals:
            _set(out, f, sum(vals) / len(vals))
    out["_votes"] = reviews
    return out


def perform_review(llm, rubric: str, item_text: str, task: str, ensemble: int = 3,
                   num_reflections: int = 1, seed: int = 0) -> dict:
    """num_reflections = extra reflection rounds after aggregation (0 = none)."""
    spec = RUBRICS[rubric]
    base_prompt = spec["form"] + fewshot(rubric) + f"\n\nHere is the item to judge:\n```\n{item_text}\n```"
    reviews = []
    for i in range(max(1, ensemble)):
        try:
            reviews.append(llm.json("judge", task, base_prompt, system=spec["system"], seed=seed + i,
                                    temperature=0.75 if ensemble > 1 else None))
        except ValueError:
            continue
    if not reviews:
        raise ValueError(f"all {ensemble} reviews failed for {task}")
    review = _aggregate(rubric, reviews)
    votes = review.pop("_votes")
    history = [{"role": "user", "content": base_prompt},
               {"role": "assistant", "content": f"THOUGHT: I aggregated {len(reviews)} judgments.\n{json.dumps(review)}"}]
    for j in range(num_reflections):
        history.append({"role": "user", "content": REFLECTION_PROMPT.format(
            current_round=j + 2, num_reflections=num_reflections + 1)})
        text = llm.chat("judge", task + ".reflect", history, system=spec["system"], seed=seed)
        history.append({"role": "assistant", "content": text})
        try:
            review = parse_json(text[text.find("{"):]) if "{" in text else review
        except (ValueError, json.JSONDecodeError):
            pass
        if "I am done" in text:
            break
    review["_votes"] = votes
    return review
