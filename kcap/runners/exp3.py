"""Experiment 3: the owner as expert. ingest -> seal -> split -> interviews -> synthesize -> tag -> apprentice -> grade -> report."""
import random
from pathlib import Path

import numpy as np

from kcap.components import gap_miner
from kcap.components.corpus import PublicCorpus
from kcap.components.synthesizer import distill, synthesize
from kcap.config import load_prereg
from kcap.eval import apprentice, grader
from kcap.human import blind_grade_cli, tag_units_cli
from kcap.human.scenarios_cli import load_sealed
from kcap.io import RUNS, read_json, read_jsonl, write_json, write_jsonl
from kcap.retrieval.index import HybridIndex
from kcap.schemas import Document, GapCard, KnowledgeUnit, Turn

EXP3 = RUNS / "exp3"
CORPUS = EXP3 / "public" / "corpus.jsonl"
TEXT_EXT = {".txt", ".md", ".eml", ".csv", ".log", ".rst", ".json"}
ARMS = ("B", "C", "D")


def ingest(folder: Path, description: str, log=print):
    """Turn a folder of your own text records into the Exp 3 corpus. Subfolder name = doc_type (e.g. sop/, ticket/)."""
    from datetime import date
    docs, n = [], 0
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_EXT:
            text = p.read_text(encoding="utf-8", errors="ignore")
            dtype = p.parent.name.lower() if p.parent != Path(folder) else "doc"
            for i in range(0, len(text), 2000):
                n += 1
                docs.append(Document(id=f"H{n:04d}", doc_type=dtype, text=f"{p.name}: {text[i:i + 2000]}",
                                     date=date.fromtimestamp(p.stat().st_mtime).isoformat()))
    write_jsonl(CORPUS, docs)
    write_json(EXP3 / "public" / "domain.json", {"name": "owner project", "description": description})
    log(f"{len(docs)} documents ingested into {CORPUS}")


def _corpus(cfg):
    return PublicCorpus.load(CORPUS, cfg, read_json(EXP3 / "public" / "domain.json")["description"])


def gen_scenarios(llm, cfg: dict, n: int = 20, seed: int = 0, log=print):
    """Generate hidden test scenarios from random corpus docs and seal them unseen. Independent of the gap miner."""
    from kcap.human.scenarios_cli import seal
    from kcap.schemas import Scenario
    corpus = _corpus(cfg)
    pool = [d for d in corpus.docs if not d.doc_type.startswith("_")]
    picks = random.Random(seed).sample(pool, len(pool))
    scen = []
    for i, d in enumerate(picks):
        if len(scen) >= n:
            break
        try:
            out = llm.json("judge", "exp3.scenario_gen", f"""You write test questions for a successor taking over this work.
{corpus.description}

SOURCE DOCUMENT (for the area only):
{d.text[:1500]}

Write ONE realistic situation (2-4 sentences, English) in this document's area where the successor must make a
decision or judgment call: a trade-off, an exception, a priority, or what to do when something goes wrong.
It should need practical experience, not just restating a fact from the document. Do not include the answer.
Return JSON {{"topic": "<3-6 word area label>", "situation": "..."}}.""", seed=seed * 1000 + i)
        except ValueError:
            continue
        if out.get("situation"):
            scen.append(Scenario(id=f"H{len(scen) + 1:02d}", prompt=out["situation"].strip(), reference_answer="",
                                 topic=str(out.get("topic", "")).strip()))
    h = seal(scen)
    log("Scenarios are hidden on purpose. Do not open sealed/scenarios.json until grading.")
    return h


def _scenario_prompt(description: str, area: str, docs_text: str) -> str:
    return f"""You write test questions for a successor taking over this work.
{description}

AREA: {area}
RELATED DOCUMENTS (for context only):
{docs_text}

Write ONE realistic situation (2-4 sentences, English) in this area where the successor must make a
decision or judgment call: a trade-off, an exception, a priority, or what to do when something goes wrong.
It should need practical experience, not just restating a fact from the documents. Use project names, not
internal codes, where the documents allow. Do not include the answer.
Return JSON {{"topic": "<3-6 word area label>", "situation": "..."}}."""


def gen_scenarios_balanced(llm, cfg: dict, per_arm: int = 7, seed: int = 1, log=print):
    """Equal hidden scenarios per arm, built from each arm's neutral topic LABELS only (no hypotheses, no transcripts)."""
    from kcap.human.scenarios_cli import seal
    from kcap.schemas import Scenario
    corpus = _corpus(cfg)
    _, half, topics = _split()
    rng = random.Random(seed)
    scen = []
    for arm in ARMS:
        labels = [topics[c] for c in sorted(half) if half[c] == arm]
        rng.shuffle(labels)
        made, tries = 0, 0
        while made < per_arm and tries < per_arm * 3:
            label = labels[tries % len(labels)]
            tries += 1
            docs = corpus.search(label, 3)
            try:
                out = llm.json("judge", "exp3.scenario_gen_balanced",
                               _scenario_prompt(corpus.description, label, corpus.render(docs, 600)),
                               seed=seed * 1000 + len(scen) * 10 + tries)
            except ValueError:
                continue
            if out.get("situation"):
                made += 1
                scen.append(Scenario(id=f"H{len(scen) + 1:02d}", prompt=out["situation"].strip(), reference_answer="",
                                     topic=str(out.get("topic", "")).strip(), arm=arm))
    rng.shuffle(scen)
    scen = [s.model_copy(update={"id": f"H{i + 1:02d}"}) for i, s in enumerate(scen)]
    h = seal(scen)
    log(f"{len(scen)} scenarios sealed ({per_arm} per interview). Do not open sealed/scenarios.json until grading.")
    return h


def split(llm, cfg: dict, seed: int = 0, log=print):
    corpus = _corpus(cfg)
    log("mining gap cards from your corpus")
    cards = gap_miner.mine(llm, corpus, cfg, EXP3 / "gap_miner", seed)
    topics = {}
    for c in cards:
        topics[c.id] = llm.chat("interviewer", "exp3.topic", f"""Give a neutral 3-6 word topic label for this
hypothesis that names the area but does NOT reveal the hypothesized practice:
{c.hypothesis}
Output only the label.""", seed=seed).strip().strip('"')
    ids = [c.id for c in cards]
    write_json(EXP3 / "split.json", {"cards": cards, "topics": topics})
    resplit(seed, log)


def resplit(seed: int = 0, log=print):
    """Randomly assign the mined cards to arms B/C/D and a blinded session order. Reuses split.json cards."""
    s = read_json(EXP3 / "split.json")
    ids = [c["id"] for c in s["cards"]]
    rng = random.Random(seed)
    rng.shuffle(ids)
    s["half"] = {cid: ARMS[i % len(ARMS)] for i, cid in enumerate(ids)}
    order = list(ARMS)
    rng.shuffle(order)
    s["session_order"] = order
    write_json(EXP3 / "split.json", s)
    for a in ARMS:
        log(f"arm {a}: {sum(1 for v in s['half'].values() if v == a)} topics")


def _split():
    s = read_json(EXP3 / "split.json")
    return [GapCard(**c) for c in s["cards"]], s["half"], s["topics"]


def build_arm(llm, cfg: dict, arm: str, d_log: list):
    """Interviewer for one arm, restricted to that arm's topics. Returns (interviewer, max_questions)."""
    from kcap.runners.common import build_interviewer
    cards, half, topics = _split()
    mine = [c for c in cards if half[c.id] == arm]
    corpus = _corpus(cfg)
    if arm == "B":
        from kcap.components.interviewer.strategy_b_staged import StagedInterviewer
        # B has no gap cards; it gets only the neutral labels of its topics as focus areas
        return StagedInterviewer(llm, corpus, 0, focus=[topics[c.id] for c in mine]), 30
    if arm == "D" and "d_search_steps" in cfg["exp3"]:
        cfg = cfg | {"strategy_d": cfg["strategy_d"] | {"search_steps": cfg["exp3"]["d_search_steps"]}}
    return build_interviewer(arm, llm, corpus, mine, cfg, 0, d_log), 10_000


def interview(llm, cfg: dict, arm: str, minutes: float | None):
    from kcap.components.interviewer.base import run_interview
    from kcap.human.term import read_multiline
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {ARMS}")
    sess = EXP3 / "sessions"
    sess.mkdir(parents=True, exist_ok=True)
    log: list = []
    iv, n = build_arm(llm, cfg, arm, log)
    run_interview(iv, lambda q: (print(f"\nQ: {q}"), read_multiline("A:"))[1], n, sess / f"{arm}.jsonl",
                  time_limit_s=(minutes or cfg["exp3"]["session_minutes"]) * 60)
    write_json(sess / f"search_log_{arm}.json", log)


def synth(llm, cfg: dict, log=print):
    corpus = _corpus(cfg)
    for h in ARMS:
        p = EXP3 / "sessions" / f"{h}.jsonl"
        if p.exists():
            turns = [Turn(**t) for t in read_jsonl(p)]
            notes = synthesize(llm, turns, corpus)
            write_json(EXP3 / f"notes_{h}.json", notes)
            # the apprentice and tagging use distilled principles, not per-answer restatements
            kus = distill(llm, turns, corpus.description)
            write_json(EXP3 / f"kus_{h}.json", kus)
            log(f"Arm {h}: {len(notes)} raw notes -> {len(kus)} principles")


def tag():
    return tag_units_cli.tag(EXP3, ARMS)


def _assign_halves(scenarios, cards, half, cfg):
    idx = HybridIndex.from_cfg([(c.id, c.hypothesis + " " + c.cluster) for c in cards], cfg)
    return {s.id: half[idx.search(f"{s.topic} {s.prompt}", 1)[0][0]] for s in scenarios}


def run_apprentice(llm, cfg: dict, log=print):
    prereg = load_prereg() or {}
    scen = load_sealed(prereg.get("exp3", {}).get("sealed_scenarios_sha256"))
    cards, half, _ = _split()
    corpus = _corpus(cfg)
    assign = {s.id: s.arm for s in scen} if all(s.arm for s in scen) else _assign_halves(scen, cards, half, cfg)
    kus = {h: [KnowledgeUnit(**k) for k in read_json(EXP3 / f"kus_{h}.json")]
           for h in ARMS if (EXP3 / f"kus_{h}.json").exists()}
    rows = []
    for s in scen:
        h = assign[s.id]
        for cond, ku in (("corpus_only", None), (f"with_{h}", kus[h])):
            ans = apprentice.answer(llm, s, corpus, ku, cfg, cond)
            # without a reference answer the owner's blind grade is the only score
            g = grader.grade(llm, s, ans, cfg) if s.reference_answer.strip() else {"score": None}
            rows.append({"scenario_id": s.id, "half": h, "condition": cond, "answer": ans} | g)
    write_json(EXP3 / "apprentice.json", rows)
    log(f"{len(rows)} apprentice answers written; grade them with `kcap exp3 grade`")


def grade():
    prereg = load_prereg() or {}
    scen = {s.id: s for s in load_sealed(prereg.get("exp3", {}).get("sealed_scenarios_sha256"))}
    return blind_grade_cli.blind_grade(EXP3, scen)


def report(log=print) -> Path:
    prereg = (load_prereg() or {}).get("exp3", {}).get("pass", {})
    rows = read_json(EXP3 / "apprentice.json")
    owner = read_json(EXP3 / "owner_grades.json") if (EXP3 / "owner_grades.json").exists() else {}
    tags = read_json(EXP3 / "tags.json") if (EXP3 / "tags.json").exists() else {}

    def score(r):  # owner grade is primary when present
        return owner.get(f"{r['scenario_id']}|{r['condition']}", r["score"])

    gain = {}
    for h in ARMS:
        base = [s for r in rows if r["half"] == h and r["condition"] == "corpus_only" and (s := score(r)) is not None]
        withk = [s for r in rows if r["half"] == h and r["condition"] == f"with_{h}" and (s := score(r)) is not None]
        gain[h] = (np.mean(withk) - np.mean(base)) / 2 if base and withk else float("nan")
    surprises = {h: sum(1 for k, v in tags.items() if k.startswith(h + ":") and v.get("surprise")) for h in ARMS}
    burden = {h: read_json(EXP3 / "sessions" / f"burden_{h}.json")["rating"]
              for h in ARMS if (EXP3 / "sessions" / f"burden_{h}.json").exists()}
    sessions = {h: read_jsonl(EXP3 / "sessions" / f"{h}.jsonl") for h in ARMS
                if (EXP3 / "sessions" / f"{h}.jsonl").exists()}
    checks = []
    for other in ("B", "C"):
        key = f"apprentice_gain_D_vs_{other}_min"
        if key in prereg:
            diff = gain["D"] - gain[other]
            checks.append((f"apprentice gain D - {other}", f"{diff:+.3f}", diff >= prereg[key]))
    if "surprise_units_min" in prereg:
        checks.append(("surprise units (D)", str(surprises["D"]), surprises["D"] >= prereg["surprise_units_min"]))
    if "burden_max" in prereg and "D" in burden:
        checks.append(("tool burden (D)", str(burden["D"]), burden["D"] <= prereg["burden_max"]))
    agree = read_json(EXP3 / "owner_vs_judge.json") if (EXP3 / "owner_vs_judge.json").exists() else {}
    L = ["# Experiment 3 report: self as expert (interviewers B vs C vs D)", "",
         "| Threshold | Observed | Result |", "|---|---|---|"]
    L += [f"| {a} | {b} | **{'PASS' if c else 'FAIL'}** |" for a, b, c in checks]
    L += ["", "| Arm | Apprentice gain (0-1) | Surprise units | Principles true | Importance of true (sum / mean) | Burden | Questions | Minutes |",
          "|---|---|---|---|---|---|---|---|"]
    for h in ARMS:
        mine = [v for k, v in tags.items() if k.startswith(h + ":")]
        n_true = sum(1 for v in mine if v.get("true"))
        imp = [v["importance"] for v in mine if v.get("true") and v.get("importance")]
        secs = sum(t.get("answer_seconds") or 0 for t in sessions.get(h, []))
        L.append(f"| {h} | {gain[h]:.3f} | {surprises[h]} | {n_true} | {sum(imp)} / {np.mean(imp) if imp else 0:.1f} | "
                 f"{burden.get(h, '—')} | {len(sessions.get(h, []))} | {secs / 60:.0f} |")
    L += ["", f"Owner vs LLM judge on apprentice answers: agreement {agree.get('agreement')}, kappa {agree.get('kappa')}"]
    p = EXP3 / "report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {p}")
    return p
