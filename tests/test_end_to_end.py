"""End-to-end dry runs with a fake LLM: Exp 1 on the imported pilot world, and the Exp 3 pipeline with dummy data."""
import builtins

import pytest

from conftest import PILOT
from kcap.eval import judge_validation
from kcap.io import read_json, write_json
from kcap.runners import exp1, exp3


@pytest.mark.skipif(not (PILOT / "private" / "rulebook.json").exists(), reason="pilot folder not present")
def test_exp1_end_to_end(make_llm, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(exp1, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(exp1, "WORLDS", tmp_path / "runs" / "worlds")
    cfg["exp1"]["n_scenarios"] = 3
    cfg["judge"]["ensemble"] = 1
    llm = make_llm()
    w = exp1.import_pilot(PILOT, exp1.WORLDS / "pilot-hvac_s0", llm, cfg)
    assert len(w.rules) == 15 and len(w.scenarios) == 3
    pub = (w.dir / "public" / "corpus.jsonl").read_text(encoding="utf-8")
    assert "related_rule_ids" not in pub
    run_dir = exp1.run(llm, cfg, ["A", "C", "D"], 4, "t", None, pilot=True)
    rep = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "D-A" in rep and "rule_recall" in rep
    ev = read_json(run_dir / w.id / "C" / "eval.json")
    assert ev["summary"]["rule_recall"] > 0
    csv = judge_validation.export(run_dir)
    assert csv.exists()


def test_exp3_dry_run(make_llm, cfg, tmp_path, monkeypatch, docs):
    root = tmp_path / "exp3"
    monkeypatch.setattr(exp3, "EXP3", root)
    monkeypatch.setattr(exp3, "CORPUS", root / "public" / "corpus.jsonl")
    folder = tmp_path / "records" / "work_order"
    folder.mkdir(parents=True)
    for d in docs:
        (folder / f"{d.id}.txt").write_text(d.text, encoding="utf-8")
    exp3.ingest(tmp_path / "records", "My old HVAC project.")
    llm = make_llm()
    cfg["gap_miner"]["min_cluster_size"] = 1
    exp3.split(llm, cfg)
    s = read_json(root / "split.json")
    assert set(s["half"].values()) <= set(exp3.ARMS)
    assert sorted(s["session_order"]) == sorted(exp3.ARMS)
    from kcap.components.interviewer.base import run_interview
    for arm in exp3.ARMS:
        iv, _ = exp3.build_arm(llm, cfg, arm, [])
        turns = run_interview(iv, lambda q: "Always swap caps on CX15.", 2, root / "sessions" / f"{arm}.jsonl")
        assert len(turns) == 2
