import numpy as np
import pytest

from kcap.eval import metrics
from kcap.eval.judge_validation import cohen_kappa
from kcap.eval.stats import bootstrap_ci, paired_bootstrap
from kcap.schemas import PlantedContradiction, Turn

GRADES = {
    "R1": {"action": 2, "reason": 3, "scope": 5},
    "R2": {"action": 4, "reason": None, "scope": None},
    "R3": {"action": None, "reason": None, "scope": None},
    "R4": {"action": None, "reason": 1, "scope": None},
}


def test_metrics():
    s = metrics.summarize(GRADES, 6)
    assert s["rule_recall"] == 0.75
    assert s["full_rule_recall"] == 0.25
    assert s["facet_recall"] == {"action": 0.5, "reason": 0.5, "scope": 0.25}
    assert s["questions_per_rule"] == 2.0
    assert s["recall_curve"] == [0.25, 0.5, 0.5, 0.75, 0.75, 0.75]


def test_disclosure_log():
    rows = [{"turn": 1, "disclosed": []}, {"turn": 2, "disclosed": [{"rule_id": "R1", "facets": ["action"]}]},
            {"turn": 3, "disclosed": [{"rule_id": "R1", "facets": ["action", "reason"]}]}]
    g = metrics.from_disclosure_log(rows, ["R1", "R2"])
    assert g["R1"] == {"action": 2, "reason": 3, "scope": None} and metrics.first_turn_any(g["R2"]) is None


def test_contradiction_catch_rate():
    k = [PlantedContradiction(id="K1", rule_id="R1", doc_ids=["D002", "D016"]),
         PlantedContradiction(id="K2", rule_id="R3", doc_ids=["D003"])]
    turns = [Turn(idx=1, strategy="D", question="D016 shows Tri-State. Why?", question_kind="contradiction"),
             Turn(idx=2, strategy="D", question="And D003?", question_kind="widen")]
    assert metrics.contradiction_catch_rate(k, GRADES, turns) == 0.5


def test_gap_card_precision():
    m = {"G1": "R1", "G2": None, "G3": "R1", "G4": "R2"}
    assert metrics.gap_card_precision(m) == 0.75
    assert metrics.gap_card_coverage(m, 4) == 0.5


def test_bootstrap():
    rng = np.random.default_rng(1)
    covered = 0
    trials = 200
    for i in range(trials):
        x = rng.normal(0.3, 1.0, size=25)
        _, lo, hi = bootstrap_ci(x, n_boot=2000, seed=i)
        covered += lo <= 0.3 <= hi
    assert 0.88 <= covered / trials <= 0.99
    d, lo, hi = paired_bootstrap([0, 0, 0, 0], [1, 1, 1, 1], n_boot=500)
    assert d == 1 and lo == 1 and hi == 1


def test_kappa():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1
    assert cohen_kappa(list("abab"), list("aabb")) == pytest.approx(0.0)
