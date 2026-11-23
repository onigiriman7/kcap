"""Experiment 2: does the gap miner find real undocumented rules in an open-source project?"""
from pathlib import Path

from kcap.components import gap_miner, gap_miner_simple
from kcap.components.corpus import PublicCorpus
from kcap.components.gap_common import cluster_corpus
from kcap.components.interviewer.base import run_interview
from kcap.components.synthesizer import synthesize
from kcap.config import load_prereg
from kcap.eval import metrics, rule_matcher
from kcap.io import read_json, read_jsonl, write_json
from kcap.oss.fetch import repo_dir
from kcap.persona.oss_persona import OSSPersona
from kcap.runners.common import build_interviewer, strat_dir
from kcap.schemas import HiddenRule


def run(llm, cfg: dict, repo: str, with_persona: bool = False, n: int = 40, log=print) -> Path:
    rdir = repo_dir(repo)
    rules = [HiddenRule(**r) for r in read_json(rdir / "private" / "rulebook.json")]
    if len(rules) < cfg["exp2"]["min_verified_rules"]:
        log(f"WARNING: only {len(rules)} verified rules (< {cfg['exp2']['min_verified_rules']}); results are weak.")
    desc = read_json(rdir / "public" / "domain.json")["description"]
    corpus = PublicCorpus.load(rdir / "public" / "corpus.jsonl", cfg, f"Open-source project {repo}: {desc}")
    out = rdir / "results"
    clusters = cluster_corpus(llm, corpus, cfg["gap_miner"]["min_cluster_size"])
    log("gap mining (AI Scientist loop)")
    cards = gap_miner.mine(llm, corpus, cfg, out / "gap_miner", 0, clusters)
    log("gap mining (simple baseline)")
    simple = gap_miner_simple.mine_simple(llm, corpus, cfg, 0, clusters)
    write_json(out / "gap_cards.json", cards)
    write_json(out / "gap_cards_simple.json", simple)
    res = {"n_verified_rules": len(rules)}
    for name, cs in (("sakana", cards), ("simple", simple)):
        m = rule_matcher.match_gap_cards(llm, cs, rules, cfg)
        res[name] = {"precision": metrics.gap_card_precision(m), "recall": metrics.gap_card_coverage(m, len(rules)),
                     "n_cards": len(cs), "map": m}
    if with_persona:
        snippets = [d.text for d in corpus.docs[:5]]
        for s in ("C", "D"):
            sdir = strat_dir(out, s)
            log(f"interview {s} vs OSS persona")
            persona = OSSPersona(llm, cfg, repo, rules, snippets, 0, sdir / "disclosure.jsonl")
            iv = build_interviewer(s, llm, corpus, cards, cfg, 0)
            turns = run_interview(iv, persona.answer, n, sdir / "transcript.jsonl")
            kus = synthesize(llm, turns, corpus)
            write_json(sdir / "kus.json", kus)
            grades = rule_matcher.to_grades(rule_matcher.match_rules(llm, rules, kus, cfg))
            res[f"persona_{s}"] = {k: v for k, v in metrics.summarize(grades, n).items() if k != "recall_curve"}
            res[f"persona_{s}"]["disclosure_recall"] = metrics.summarize(
                metrics.from_disclosure_log(read_jsonl(sdir / "disclosure.jsonl"), [r.id for r in rules]), n)["rule_recall"]
    write_json(out / "exp2_results.json", res)
    thr = ((load_prereg() or {}).get("exp2", {}).get("pass", {})).get("gap_card_precision_min")
    p = res["sakana"]["precision"] or 0
    lines = [f"# Experiment 2 report: {repo}", "", f"Verified hidden rules: {len(rules)}", "",
             "| Miner | Precision | Recall (rules hit) | Cards |", "|---|---|---|---|"]
    for name in ("sakana", "simple"):
        r = res[name]
        lines.append(f"| {name} | {r['precision'] or 0:.2f} | {r['recall']:.2f} | {r['n_cards']} |")
    if thr is not None:
        lines += ["", f"Pre-registered gap-card precision >= {thr}: **{'PASS' if p >= thr else 'FAIL'}** ({p:.2f})"]
    for s in ("C", "D"):
        if f"persona_{s}" in res:
            lines.append(f"\nPersona run {s}: rule recall {res[f'persona_{s}']['rule_recall']:.2f}")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {out / 'report.md'}")
    return out
