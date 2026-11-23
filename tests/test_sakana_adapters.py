import json
import socket
from pathlib import Path

import pytest

from kcap.components.corpus import PublicCorpus
from kcap.components.gap_common import finalize
from kcap.sakana_adapters import ideation_adapter
from kcap.sakana_adapters.review_adapter import perform_review


def test_ideation_adapter_offline(make_llm, docs, cfg, monkeypatch):
    def no_net(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    corpus = PublicCorpus(docs, cfg)
    cards = ideation_adapter.generate_temp_free_gap_cards(make_llm(), corpus, "workshop", 2, 2)
    assert cards
    assert "SemanticScholar" not in ideation_adapter.SYSTEM_PROMPT
    src = Path(ideation_adapter.__file__).read_text(encoding="utf-8")
    for forbidden in ("exec(", "eval(", "subprocess", "requests", "urllib", "semantic_scholar"):
        assert forbidden not in src.replace("perform_ideation", "")


def test_ideation_adapter_schema(make_llm, docs, cfg):
    corpus = PublicCorpus(docs, cfg)
    raw = ideation_adapter.generate_temp_free_gap_cards(make_llm(), corpus, "workshop", 1, 2)
    cards = finalize(raw, corpus, cfg, 10)
    assert cards and cards[0].id == "G01"
    for c in cards:
        assert c.evidence_doc_ids and all(d in corpus.by_id for d in c.evidence_doc_ids)


def test_review_adapter_mean_and_majority(make_llm):
    scores = iter([2, 2, 0])

    def h(task, system, messages):
        if task.endswith(".reflect"):
            return "THOUGHT: fine. I am done\n" + json.dumps({"score": 4 / 3, "justification": "x"})
        return json.dumps({"score": next(scores), "justification": "x"})
    rev = perform_review(make_llm(h), "scenario_answer", "item", "eval.grade", ensemble=3, num_reflections=1)
    assert rev["score"] == pytest.approx(4 / 3)

    verdicts = iter(["captured", "not", "captured"])

    def h2(task, system, messages):
        if task.endswith(".reflect"):
            return "I am done"
        v = next(verdicts)
        return json.dumps({f: {"verdict": v, "ku_id": "KU001" if v == "captured" else None}
                           for f in ("action", "reason", "scope")})
    rev = perform_review(make_llm(h2), "rule_match", "item", "eval.rule_match", ensemble=3, num_reflections=1)
    assert rev["action"]["verdict"] == "captured" and rev["action"]["ku_id"] == "KU001"


def test_review_adapter_obvious_cases(make_llm):
    """Fixture examples with obvious answers produce the expected grades (fake judge stands in for the model)."""
    def h(task, system, messages):
        item = messages[0]["content"].split("Here is the item to judge:")[-1]
        right = "Replace the capacitor free" in item
        return json.dumps({"score": 2 if right else 0, "justification": "x"}) + ("\nI am done" if len(messages) > 1 else "")
    good = "SCENARIO: s\nREFERENCE: Replace the capacitor free\nCANDIDATE: Replace the capacitor free"
    bad = "SCENARIO: s\nREFERENCE: r\nCANDIDATE: leave it"
    assert perform_review(make_llm(h), "scenario_answer", good, "eval.grade")["score"] == 2
    assert perform_review(make_llm(h), "scenario_answer", bad, "eval.grade")["score"] == 0
