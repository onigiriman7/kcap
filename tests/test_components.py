import pytest

from kcap.components.contradiction import ContradictionChecker
from kcap.components.corpus import PublicCorpus
from kcap.components.interviewer.base import run_interview
from kcap.components.interviewer.strategy_d_adaptive import AdaptiveInterviewer
from kcap.llm.client import BudgetExceeded
from kcap.persona.synthetic_persona import SyntheticPersona
from kcap.runners.common import build_interviewer
from kcap.schemas import GapCard
from kcap.synth.leak_check import leak_check


@pytest.fixture
def cards():
    return [GapCard(id="G01", hypothesis="Capacitor swapped with fan motor", evidence_doc_ids=["D003"], importance=0.9,
                    undocumentedness=0.9, opening_question="In D003 you swapped the capacitor at $0. Why?"),
            GapCard(id="G02", hypothesis="Compressors from Tri-State", evidence_doc_ids=["D002"], importance=0.7,
                    undocumentedness=0.8, opening_question="D002 shows Tri-State, not Allied. Why?")]


@pytest.mark.parametrize("strategy", ["A", "B", "C", "D", "C+K", "C+S"])
def test_budget_equal(strategy, make_llm, docs, cfg, cards, rules, domain):
    llm = make_llm()
    corpus = PublicCorpus(docs, cfg)
    persona = SyntheticPersona(llm, cfg, domain, rules)
    iv = build_interviewer(strategy, llm, corpus, cards, cfg, 0)
    turns = run_interview(iv, persona.answer, 7)
    assert len(turns) == 7 and [t.idx for t in turns] == list(range(1, 8))
    assert all(t.question for t in turns)


def test_persona_gating(make_llm, cfg, rules, domain, tmp_path):
    p = SyntheticPersona(make_llm(), cfg, domain, rules, log_path=tmp_path / "d.jsonl")
    p.answer("What should your successor know?")
    assert p.disclosed == {"R01": set(), "R02": set()}
    p.answer("On CX15 jobs the capacitor line is $0. What do you do there?")
    assert p.disclosed["R01"] == {"action"}
    p.answer("Why the capacitor?")
    assert p.disclosed["R01"] == {"action", "reason"}
    p.answer("Any exceptions for the capacitor?")
    assert p.disclosed["R01"] == {"action", "reason", "scope"}
    assert p.disclosed["R02"] == set()


def test_persona_answer_sees_only_released_facets(cfg, rules, domain, make_llm):
    seen = []

    def h(task, system, messages):
        if task.endswith(".answer"):
            seen.append(system)
            return "ok"
        from conftest import default_handler
        return default_handler(task, system, messages)
    p = SyntheticPersona(make_llm(h), cfg, domain, rules)
    p.answer("Tell me about capacitor jobs")
    assert rules[0].action in seen[-1] and rules[0].reason not in seen[-1] and rules[0].scope not in seen[-1]


def test_leak_check(make_llm, docs, rules, domain):
    leaky = docs + [docs[2].model_copy(update={"id": "D006", "text": "We swap caps because the 2014 batch fails."})]
    out, report = leak_check(make_llm(), domain, leaky, rules, 3)
    assert report["initial_leaking"] == 1
    assert "because" not in next(d for d in out if d.id == "D006").text


def test_cache_determinism(make_llm, docs, cfg, cards, rules, domain):
    def run(llm):
        corpus = PublicCorpus(docs, cfg)
        persona = SyntheticPersona(llm, cfg, domain, rules)
        iv = build_interviewer("D", llm, corpus, cards, cfg, 0)
        return [t.model_dump(exclude={"answer_seconds"}) for t in run_interview(iv, persona.answer, 4)]
    first = run(make_llm())

    def boom(*a):
        raise AssertionError("cache miss")
    assert run(make_llm(boom)) == first


def test_treequest_wiring(make_llm, docs, cfg, cards):
    llm = make_llm()
    corpus = PublicCorpus(docs, cfg)
    checker = ContradictionChecker(llm, corpus)
    checker.pre_interview()
    log = []
    iv = AdaptiveInterviewer(llm, corpus, cards, cfg, checker, 0, log)
    from kcap.schemas import Turn
    t1 = iv.next_question([], 5)
    t1.idx, t1.answer = 1, "We swap it."
    iv._turns = [t1]
    assert set(iv.actions()) == {"widen", "deepen", "contradict"}
    t2 = iv.next_question([t1], 5)
    assert t2.question and t2.question_kind in ("widen", "deepen", "contradiction")
    assert log[-1]["nodes"] == cfg["strategy_d"]["search_steps"] and t2.search_cost_tokens > 0
    assert isinstance(t2, Turn)


def test_budget_cap(cfg, tmp_path):
    from kcap.llm.client import LLM
    c = dict(cfg, budget={"max_usd": 0.001}, pricing={m["model"]: [1000.0, 1000.0] for m in cfg["roles"].values()})
    llm = LLM(c, cache_dir=tmp_path, cost_log=tmp_path / "c.jsonl", fake=lambda *a: "OK")
    with pytest.raises(BudgetExceeded):
        llm.chat("judge", "x", "hello")
