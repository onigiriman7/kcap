"""Experiment 1: synthetic expert. Generate worlds, interview the persona with each strategy, evaluate, report."""
import time
from pathlib import Path

import numpy as np

from kcap.components import gap_miner, gap_miner_simple
from kcap.components.corpus import PublicCorpus
from kcap.components.gap_common import cluster_corpus
from kcap.components.interviewer.base import run_interview
from kcap.components.synthesizer import synthesize
from kcap.config import config_hash, load_prereg, require_committed_prereg
from kcap.eval import apprentice, grader, metrics, rule_matcher
from kcap.eval.report import write_report
from kcap.io import RUNS, read_json, read_jsonl, write_json
from kcap.persona.synthetic_persona import SyntheticPersona
from kcap.runners.common import build_interviewer, safe_name, strat_dir, token_split
from kcap.schemas import GapCard, KnowledgeUnit, Turn
from kcap.synth.world import World, generate_world, import_pilot, load_world, slug

WORLDS = RUNS / "worlds"


def generate(llm, cfg, n_domains: int, n_seeds: int, log=print) -> list[Path]:
    out = []
    for d in cfg["exp1"]["domains"][:n_domains]:
        for s in range(n_seeds):
            wdir = WORLDS / f"{slug(d)}_s{s}"
            if (wdir / "private" / "scenarios.json").exists():
                log(f"[{wdir.name}] exists, skipping")
            else:
                generate_world(llm, cfg, d, s, wdir, log)
            out.append(wdir)
    return out


def worlds(filter_: str | None = None) -> list[World]:
    dirs = sorted(p.parent.parent for p in WORLDS.glob("*/private/rulebook.json"))
    if filter_:
        keep = set(filter_.split(","))
        dirs = [d for d in dirs if d.name in keep or any(d.name.startswith(k) for k in keep)]
    return [load_world(d) for d in dirs]


def _cards(llm, cfg, world: World, corpus: PublicCorpus, wout: Path, seed: int):
    if (wout / "gap_cards.json").exists():
        return ([GapCard(**c) for c in read_json(wout / "gap_cards.json")],
                [GapCard(**c) for c in read_json(wout / "gap_cards_simple.json")])
    clusters = cluster_corpus(llm, corpus, cfg["gap_miner"]["min_cluster_size"])
    cards = gap_miner.mine(llm, corpus, cfg, wout / "gap_miner", seed, clusters)
    simple = gap_miner_simple.mine_simple(llm, corpus, cfg, seed, clusters)
    write_json(wout / "gap_cards.json", cards)
    write_json(wout / "gap_cards_simple.json", simple)
    return cards, simple


def interview_one(llm, cfg, world: World, corpus: PublicCorpus, cards, strategy: str, n: int, sdir: Path, seed: int):
    if (sdir / "kus.json").exists():
        return
    for f in ("transcript.jsonl", "disclosure.jsonl"):
        (sdir / f).unlink(missing_ok=True)
    sops = [d.text for d in corpus.sops()]
    persona = SyntheticPersona(llm, cfg, world.domain, world.rules, sops, seed, sdir / "disclosure.jsonl")
    d_log: list = []
    before = dict(llm.tokens_by_task)
    t0 = time.time()
    iv = build_interviewer(strategy, llm, corpus, cards, cfg, seed, d_log)
    turns = run_interview(iv, persona.answer, n, sdir / "transcript.jsonl")
    kus = synthesize(llm, turns, corpus, seed)
    write_json(sdir / "search_log.json", d_log)
    write_json(sdir / "cost.json", token_split(before, llm.tokens_by_task) | {"seconds": round(time.time() - t0, 1)})
    write_json(sdir / "kus.json", kus)


def _baseline(llm, cfg, world: World, corpus: PublicCorpus, wout: Path):
    p = wout / "apprentice_baseline.json"
    if p.exists():
        return read_json(p)
    rows = []
    for sc in world.scenarios:
        ans = apprentice.answer(llm, sc, corpus, None, cfg, "corpus_only")
        g = grader.grade(llm, sc, ans, cfg)
        rows.append({"scenario_id": sc.id, "condition": "corpus_only", "answer": ans} | g)
    write_json(p, rows)
    return rows


def evaluate_one(llm, cfg, world: World, corpus: PublicCorpus, sdir: Path, n: int, baseline: list):
    turns = [Turn(**t) for t in read_jsonl(sdir / "transcript.jsonl")]
    kus = [KnowledgeUnit(**k) for k in read_json(sdir / "kus.json")]
    matches = rule_matcher.match_rules(llm, world.rules, kus, cfg)
    grades = rule_matcher.to_grades(matches)
    rids = [r.id for r in world.rules]
    disc = metrics.summarize(metrics.from_disclosure_log(read_jsonl(sdir / "disclosure.jsonl"), rids), n) \
        if (sdir / "disclosure.jsonl").exists() else None
    app = []
    for sc in world.scenarios:
        ans = apprentice.answer(llm, sc, corpus, kus, cfg, "with_kus")
        app.append({"scenario_id": sc.id, "condition": "with_kus", "answer": ans} | grader.grade(llm, sc, ans, cfg))
    gain = (np.mean([a["score"] for a in app]) - np.mean([b["score"] for b in baseline])) / 2 if app and baseline else None
    write_json(sdir / "eval.json", {
        "matches": matches, "grades": grades,
        "summary": metrics.summarize(grades, n),
        "summary_lenient": metrics.summarize(rule_matcher.to_grades(matches, strict=False), n),
        "disclosure_summary": disc,
        "contradiction_catch_rate": metrics.contradiction_catch_rate(world.contradictions, grades, turns),
        "apprentice": app + baseline, "apprentice_gain": float(gain) if gain is not None else None,
        "cost": read_json(sdir / "cost.json"),
    })


def evaluate_gap(llm, cfg, world: World, wout: Path):
    if (wout / "gap_eval.json").exists():
        return
    res = {}
    for name, f in (("sakana", "gap_cards.json"), ("simple", "gap_cards_simple.json")):
        cards = [GapCard(**c) for c in read_json(wout / f)]
        m = rule_matcher.match_gap_cards(llm, cards, world.rules, cfg)
        res[name] = {"map": m, "precision": metrics.gap_card_precision(m) or 0.0,
                     "coverage": metrics.gap_card_coverage(m, len(world.rules)), "n_cards": len(cards)}
    write_json(wout / "gap_eval.json", res)


def run(llm, cfg, strategies: list[str], n: int, run_id: str, world_filter: str | None, pilot: bool,
        do_eval: bool = True, log=print) -> Path:
    prereg_commit = None if pilot else require_committed_prereg()
    run_dir = RUNS / run_id
    write_json(run_dir / "run_meta.json", {"config_hash": config_hash(cfg), "prereg_commit": prereg_commit,
                                           "n_questions": n, "strategies": strategies, "pilot": pilot})
    ws = worlds(world_filter)
    if pilot:
        ws = ws[:1]
    if not ws:
        raise SystemExit("no worlds found; run `kcap exp1 generate` or `kcap exp1 import-pilot` first")
    for wi, world in enumerate(ws):
        seed = cfg.get("seed", 0) * 1000 + wi
        wout = run_dir / world.id
        corpus = PublicCorpus.load(world.public_corpus_path, cfg, world.public_description)
        log(f"[{world.id}] gap mining")
        cards, _ = _cards(llm, cfg, world, corpus, wout, seed)
        for s in strategies:
            log(f"[{world.id}] interview {s} (N={n})")
            interview_one(llm, cfg, world, corpus, cards, s, n, strat_dir(wout, s), seed)
        if do_eval:
            evaluate_world(llm, cfg, world, corpus, wout, strategies, n, log)
    if do_eval:
        report(run_id, llm)
    return run_dir


def evaluate_world(llm, cfg, world, corpus, wout, strategies, n, log=print):
    log(f"[{world.id}] evaluating")
    evaluate_gap(llm, cfg, world, wout)
    base = _baseline(llm, cfg, world, corpus, wout)
    for s in strategies:
        sdir = strat_dir(wout, s)
        if (sdir / "kus.json").exists():
            evaluate_one(llm, cfg, world, corpus, sdir, n, base)


def evaluate(llm, cfg, run_id: str, log=print):
    """Re-run evaluation only, from cached transcripts (e.g. after fixing the judge)."""
    run_dir = RUNS / run_id
    meta = read_json(run_dir / "run_meta.json")
    for world in worlds():
        wout = run_dir / world.id
        if wout.exists():
            for f in [wout / "gap_eval.json", wout / "apprentice_baseline.json"]:
                f.unlink(missing_ok=True)
            corpus = PublicCorpus.load(world.public_corpus_path, cfg, world.public_description)
            evaluate_world(llm, cfg, world, corpus, wout, meta["strategies"], meta["n_questions"], log)
    return report(run_id, llm)


def report(run_id: str, llm=None) -> Path:
    run_dir = RUNS / run_id
    meta = read_json(run_dir / "run_meta.json") | {"usd": llm.spent_usd if llm else 0}
    return write_report(run_dir, load_prereg(), meta)


def budget_curve(llm, cfg, world_filter: str, ns: list[int], strategies: list[str], log=print) -> Path:
    """Run the budget sweep N in ns on one world; writes runs/budget_curve/summary.json + png."""
    world = worlds(world_filter)[0]
    corpus = PublicCorpus.load(world.public_corpus_path, cfg, world.public_description)
    root = RUNS / "budget_curve" / world.id
    cards, _ = _cards(llm, cfg, world, corpus, root, 0)
    base = _baseline(llm, cfg, world, corpus, root)
    summary = {}
    for n in ns:
        for s in strategies:
            sdir = root / f"N{n}" / safe_name(s)
            log(f"[budget] N={n} {s}")
            interview_one(llm, cfg, world, corpus, cards, s, n, sdir, 0)
            if not (sdir / "eval.json").exists():
                evaluate_one(llm, cfg, world, corpus, sdir, n, base)
            summary.setdefault(s, {})[n] = read_json(sdir / "eval.json")["summary"]["rule_recall"]
    write_json(root / "summary.json", summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for s, d in summary.items():
            ax.plot(list(d), [v * 100 for v in d.values()], marker="o", label=s)
        ax.set_xlabel("Question budget N")
        ax.set_ylabel("Rule recall (%)")
        ax.legend(frameon=False)
        fig.savefig(root / "budget_curve.png", dpi=150)
    except ImportError:
        pass
    return root


def chat_persona(llm, cfg, world_filter: str):
    """Owner step 3: interview the persona yourself; disclosure log printed after each answer."""
    world = worlds(world_filter)[0]
    sops = [d.text for d in PublicCorpus.load(world.public_corpus_path, cfg).sops()]
    log = RUNS / "persona_chat" / f"{world.id}_{int(time.time())}.jsonl"
    p = SyntheticPersona(llm, cfg, world.domain, world.rules, sops, 0, log)
    print(f"Interviewing the persona for {world.id}. Empty line to quit.\n")
    while True:
        q = input("You> ").strip()
        if not q:
            break
        print(f"Expert> {p.answer(q)}")
        disclosed = read_jsonl(log)[-1]["disclosed"]
        print(f"   [log: disclosed {disclosed or 'nothing'}]\n")
    print(f"Saved to {log}")


__all__ = ["generate", "import_pilot", "run", "evaluate", "report", "budget_curve", "chat_persona", "worlds"]
