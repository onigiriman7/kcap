import json
import os
import re
from pathlib import Path

import pytest

from kcap.config import load_config
from kcap.llm.client import LLM
from kcap.schemas import Document, Domain, HiddenRule

PILOT = Path(os.environ.get("KCAP_PILOT_DIR", "kcap_pilot"))


def _last_user(messages):
    return [m for m in messages if m["role"] == "user"][-1]["content"]


def default_handler(task: str, system: str, messages: list[dict]) -> str:
    p = _last_user(messages)
    if task.endswith(".gate"):
        q = p.split("NEW QUESTION:")[-1].splitlines()[0].lower()
        if "capacitor" in q:
            ask = "why" if "why" in q else "exceptions" if "except" in q else "what"
            return json.dumps({"targets": [{"rule_id": "R01", "ask": ask}]})
        return json.dumps({"targets": []})
    if task.endswith(".answer"):
        return "Well, it depends. We follow the procedures."
    if task == "A.followup":
        return "Can you give an example?"
    if task == "B.summary":
        return "A service company."
    if task == "B.plan":
        k = int(re.search(r"Write (\d+) questions", p).group(1))
        return json.dumps({"questions": [f"Staged question {i}? (For example: ...)" for i in range(k)]})
    if task == "C.followup":
        return json.dumps({"rejected": False, "question": "Why do you do that?"})
    if task in ("contradiction.pre", "contradiction.judge"):
        ids = re.findall(r"\[([DH]\d{3,4}) \|", p)
        return json.dumps({"conflicts": [{"claim": "c", "doc_ids": ids[:1], "question": f"Records {ids[:1]} differ. Why?"}]} if ids else {"conflicts": []})
    if task == "contradiction.claims":
        return json.dumps({"claims": ["We always use the preferred vendor."]})
    if task in ("D.deepen", "D.refine"):
        return "What exceptions are there?"
    if task == "D.simulate":
        return "Probably we do it differently for older units."
    if task == "D.score":
        return json.dumps({"novelty": 0.6, "specificity": 0.4, "rationale": "r"})
    if task == "synth.extract":
        return json.dumps({"units": [{"condition": "fan motor replaced", "action": "swap capacitor", "reason": None, "scope": None}]})
    if task == "synth.merge":
        n = len(re.findall(r"^\d+: WHEN", p, re.M))
        return json.dumps({"merged": [{"condition": "fan motor replaced", "action": "swap capacitor", "reason": None,
                                       "scope": None, "from": list(range(n))}]})
    if task == "synth.confidence":
        return json.dumps({"checks": [{"unit": 0, "confidence": "confirmed_by_records", "doc_ids": []}]})
    if task.startswith("eval.rule_match"):
        ku = re.search(r"(KU\d{3})", " ".join(m["content"] for m in messages).split("Here is the item to judge")[-1])
        k = ku.group(1) if ku else None
        return json.dumps({"action": {"verdict": "captured", "ku_id": k}, "reason": {"verdict": "not", "ku_id": None},
                           "scope": {"verdict": "not", "ku_id": None}}) + ("\nI am done" if "Round" in p else "")
    if task.startswith("eval.gap_card_match"):
        return json.dumps({"rule_id": "R01", "justification": "x"}) + ("\nI am done" if "Round" in p else "")
    if task.startswith("eval.grade"):
        return "THOUGHT: I am done\n" + json.dumps({"score": 1, "justification": "x"}) if "Round" in p else json.dumps({"score": 1, "justification": "x"})
    if task.startswith("apprentice"):
        return "Do the usual thing."
    if task == "gap_miner.cluster":
        ids = re.findall(r"\[([DH]\d{3,4})\]", p)
        return json.dumps({"assignments": {i: ("Corvane CX" if int(i[1:]) % 2 else "Allied") for i in ids}})
    if task == "exp3.topic":
        return "Equipment topic"
    if task == "gap_miner.ideation":
        if "Round" not in p:
            return 'ACTION: CorpusSearch\nARGUMENTS: {"query": "capacitor fan motor"}'
        ids = re.findall(r"\[([DH]\d{3,4}) \|", p) or ["D001"]
        card = {"hypothesis": f"Capacitor swapped with fan motor ({ids[0]})", "evidence_doc_ids": ids[:2] + ["NOPE"],
                "importance": 0.8, "undocumentedness": 0.9, "opening_question": f"In {ids[0]} you swapped the capacitor. Why?"}
        return "ACTION: FinalizeGapCard\nARGUMENTS: " + json.dumps({"gap_card": card})
    if task == "gap_miner.simple":
        return json.dumps({"gap_cards": [{"hypothesis": "Compressors from Tri-State", "evidence_doc_ids": ["D002"],
                                          "importance": 0.7, "undocumentedness": 0.8, "opening_question": "Why Tri-State?"}]})
    if task == "synth.leak_check":
        ids = re.findall(r"^\[(D\d{3})\] .*because", p, re.M)
        return json.dumps({"leaks": [{"doc_id": i, "rule_id": "R01", "facet": "reason"} for i in ids]})
    if task == "synth.leak_regen":
        return json.dumps({"text": "Swapped capacitor, $0."})
    if task == "synth.scenarios":
        k = int(re.search(r"Write (\d+) realistic", p).group(1))
        return json.dumps({"scenarios": [{"prompt": f"Scenario {i}", "reference_answer": "Swap the cap.", "rule_ids": ["R01"],
                                          "depends_on_scope": True} for i in range(k)]})
    if task == "check":
        return "OK"
    raise AssertionError(f"unhandled fake task {task}")


@pytest.fixture
def cfg():
    c = load_config()
    c["judge"] = {"ensemble": 3, "num_reflections": 1}
    c["strategy_d"]["search_steps"] = 4
    c["gap_miner"].update({"generations_per_cluster": 1, "num_reflections": 1, "top_k": 10})
    c["persona"] = {"misremember_prob": 0.0, "volunteer_prob": 0.0}
    c["budget"] = {"max_usd": 1000}
    return c


@pytest.fixture
def make_llm(cfg, tmp_path):
    def _make(handler=default_handler, **kw):
        return LLM(cfg, cache_dir=tmp_path / "cache", cost_log=tmp_path / "cost.jsonl", fake=handler, **kw)
    return _make


@pytest.fixture
def rules():
    return [HiddenRule(id="R01", trigger="Replacing a condenser fan motor on a Corvane CX unit",
                       action="Also replace the run capacitor at no charge", reason="2014-16 batch capacitors fail",
                       scope="Only serials CX14-CX16", importance=2, visible_effects=["$0 capacitor line"]),
            HiddenRule(id="R02", trigger="Quoting Pellham schools", action="Add 15% escort allowance",
                       reason="Escorted access", scope="Not the bus garage", importance=2)]


@pytest.fixture
def docs():
    return [Document(id="D001", doc_type="sop", date="2022-01-01", text="SOP-PUR-02: order all components from Allied."),
            Document(id="D002", doc_type="work_order", date="2022-02-01", text="Compressor bought from Tri-State.", related_rule_ids=["R03"]),
            Document(id="D003", doc_type="work_order", date="2022-03-01", text="CX15 fan motor replaced, capacitor $0 per JM.", related_rule_ids=["R01"]),
            Document(id="D004", doc_type="ticket", date="2022-04-01", text="CX19 fan motor replaced. Capacitor tested good.", related_rule_ids=["R01"]),
            Document(id="D005", doc_type="email", date="2022-05-01", text="Pellham school quote with escort allowance line.", related_rule_ids=["R02"])]


@pytest.fixture
def domain():
    return Domain(name="HVAC service", description="Regional HVAC field service.", roles=["lead technician"])
