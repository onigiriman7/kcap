"""Exp 1 report: per-strategy table, paired bootstrap diffs over (domain, seed), PASS/FAIL vs prereg, plots."""
from pathlib import Path

import numpy as np

from kcap.eval.stats import bootstrap_ci, paired_bootstrap
from kcap.io import read_json

PAIRS = [("D", "A"), ("D", "C"), ("C", "B"), ("C", "A"), ("B", "A"), ("D", "C+K"), ("D", "C+S"), ("C+K", "C"),
         ("C+S", "C")]
METRICS = ["rule_recall", "full_rule_recall", "contradiction_catch_rate", "apprentice_gain", "disclosure_recall"]


def collect(run_dir: Path) -> dict:
    """{strategy: {world: flat metrics}}"""
    out: dict = {}
    for ev in sorted(Path(run_dir).glob("*/*/eval.json")):
        e = read_json(ev)
        s, w = ev.parent.name.replace("plus", "+"), ev.parent.parent.name
        out.setdefault(s, {})[w] = {
            "rule_recall": e["summary"]["rule_recall"],
            "full_rule_recall": e["summary"]["full_rule_recall"],
            "facet_recall": e["summary"]["facet_recall"],
            "questions_per_rule": e["summary"]["questions_per_rule"],
            "recall_curve": e["summary"]["recall_curve"],
            "contradiction_catch_rate": e.get("contradiction_catch_rate"),
            "apprentice_gain": e.get("apprentice_gain"),
            "disclosure_recall": (e.get("disclosure_summary") or {}).get("rule_recall"),
            "interview_tokens": e["cost"]["interview_tokens"],
            "search_tokens": e["cost"]["search_tokens"],
        }
    return out


def _vals(d: dict, metric: str):
    return {w: m[metric] for w, m in d.items() if m.get(metric) is not None}


def pairwise(data: dict) -> list[dict]:
    rows = []
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        for m in METRICS:
            va, vb = _vals(data[a], m), _vals(data[b], m)
            ws = sorted(set(va) & set(vb))
            if not ws:
                continue
            mean, lo, hi = paired_bootstrap([vb[w] for w in ws], [va[w] for w in ws])
            rows.append({"pair": f"{a}-{b}", "metric": m, "n": len(ws), "diff": mean, "lo": lo, "hi": hi})
    return rows


def check_prereg(data: dict, rows: list[dict], gap: dict, prereg: dict | None) -> list[tuple[str, str, str]]:
    if not prereg:
        return [("prereg.yaml", "missing", "N/A")]
    p = prereg.get("exp1", {}).get("pass", {})
    res = []

    def row(pair, metric):
        return next((r for r in rows if r["pair"] == pair and r["metric"] == metric), None)

    if "rule_recall_D_minus_A_min" in p:
        r = row("D-A", "rule_recall")
        res.append(("rule recall D - A >= %.2f" % p["rule_recall_D_minus_A_min"],
                    f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" if r else "n/a",
                    "PASS" if r and r["diff"] >= p["rule_recall_D_minus_A_min"] else "FAIL"))
    if p.get("rule_recall_D_minus_C_ci_excludes_zero"):
        r = row("D-C", "rule_recall")
        res.append(("rule recall D - C CI excludes 0", f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" if r else "n/a",
                    "PASS" if r and (r["lo"] > 0 or r["hi"] < 0) else "FAIL"))
    if "contradiction_catch_rate_min" in p and "D" in data:
        v = list(_vals(data["D"], "contradiction_catch_rate").values())
        m = float(np.mean(v)) if v else float("nan")
        res.append(("D contradiction catch rate >= %.2f" % p["contradiction_catch_rate_min"], f"{m:.3f}",
                    "PASS" if v and m >= p["contradiction_catch_rate_min"] else "FAIL"))
    return res


def plots(data: dict, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, d in sorted(data.items()):
        curves = np.array([m["recall_curve"] for m in d.values()])
        if curves.size == 0:
            continue
        mean = curves.mean(axis=0) * 100
        lo = np.quantile(curves, 0.025, axis=0) * 100 if len(curves) > 1 else mean
        hi = np.quantile(curves, 0.975, axis=0) * 100 if len(curves) > 1 else mean
        xs = np.arange(1, len(mean) + 1)
        ax.plot(xs, mean, label=s)
        ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xlabel("Questions asked")
    ax.set_ylabel("Hidden rules found (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "recall_curve.png", dpi=150)
    files.append("recall_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    strategies = sorted(data)
    for i, f in enumerate(("action", "reason", "scope")):
        ax.bar(np.arange(len(strategies)) + i * 0.27,
               [np.mean([m["facet_recall"][f] for m in data[s].values()]) * 100 for s in strategies], 0.27, label=f)
    ax.set_xticks(np.arange(len(strategies)) + 0.27, strategies)
    ax.set_ylabel("Facet recall (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "facet_recall.png", dpi=150)
    files.append("facet_recall.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in strategies:
        xs = [m["interview_tokens"] + m["search_tokens"] for m in data[s].values()]
        ys = [m["rule_recall"] * 100 for m in data[s].values()]
        ax.scatter(xs, ys, label=s)
    ax.set_xlabel("Tokens per run")
    ax.set_ylabel("Rule recall (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "cost_vs_recall.png", dpi=150)
    files.append("cost_vs_recall.png")
    plt.close(fig)
    return files


def write_report(run_dir: Path, prereg: dict | None, meta: dict) -> Path:
    run_dir = Path(run_dir)
    data = collect(run_dir)
    rows = pairwise(data)
    gap = {}
    for g in run_dir.glob("*/gap_eval.json"):
        gap[g.parent.name] = read_json(g)
    checks = check_prereg(data, rows, gap, prereg)
    files = plots(data, run_dir)

    L = [f"# Experiment 1 report: {run_dir.name}", "",
         f"Config hash `{meta.get('config_hash')}` · prereg commit `{meta.get('prereg_commit')}` · "
         f"worlds: {len({w for d in data.values() for w in d})} · N = {meta.get('n_questions')}", "",
         "## Pre-registered thresholds", "", "| Threshold | Observed | Result |", "|---|---|---|"]
    L += [f"| {a} | {b} | **{c}** |" for a, b, c in checks]
    L += ["", "## Per strategy (mean over worlds, 95% bootstrap CI)", "",
          "| Strategy | Rule recall | Full-rule recall | Action | Reason | Scope | Q per rule | Contradictions caught | Apprentice gain | Disclosure-log recall | Interview tok | Search tok |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in sorted(data.items()):
        def ci(metric):
            v = list(_vals(d, metric).values())
            if not v:
                return "—"
            m, lo, hi = bootstrap_ci(v)
            return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
        fr = {f: np.mean([m["facet_recall"][f] for m in d.values()]) for f in ("action", "reason", "scope")}
        qpr = [m["questions_per_rule"] for m in d.values() if m["questions_per_rule"]]
        qpr_s = f"{np.mean(qpr):.1f}" if qpr else "—"
        L.append(f"| {s} | {ci('rule_recall')} | {ci('full_rule_recall')} | {fr['action']:.2f} | {fr['reason']:.2f} | "
                 f"{fr['scope']:.2f} | {qpr_s} | {ci('contradiction_catch_rate')} | {ci('apprentice_gain')} | "
                 f"{ci('disclosure_recall')} | {np.mean([m['interview_tokens'] for m in d.values()]):,.0f} | "
                 f"{np.mean([m['search_tokens'] for m in d.values()]):,.0f} |")
    L += ["", "## Paired differences (unit = domain x seed, 10,000 resamples)", "",
          "| Pair | Metric | n | Diff | 95% CI |", "|---|---|---|---|---|"]
    L += [f"| {r['pair']} | {r['metric']} | {r['n']} | {r['diff']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] |" for r in rows]
    if gap:
        L += ["", "## Gap miner: AI Scientist loop vs. simple baseline", "",
              "| World | Sakana precision | Simple precision | Sakana coverage | Simple coverage |", "|---|---|---|---|---|"]
        for w, g in sorted(gap.items()):
            L.append(f"| {w} | {g['sakana']['precision']:.2f} | {g['simple']['precision']:.2f} | "
                     f"{g['sakana']['coverage']:.2f} | {g['simple']['coverage']:.2f} |")
    L += ["", "## Example transcripts (best / worst rule recall per strategy)", ""]
    for s, d in sorted(data.items()):
        best = max(d, key=lambda w: d[w]["rule_recall"])
        worst = min(d, key=lambda w: d[w]["rule_recall"])
        L.append(f"- **{s}**: best [{best}]({best}/{s}/transcript.jsonl) ({d[best]['rule_recall']:.2f}), "
                 f"worst [{worst}]({worst}/{s}/transcript.jsonl) ({d[worst]['rule_recall']:.2f})")
    L += ["", "## Cost", "", f"Total spend logged this process: ${meta.get('usd', 0):.2f}", ""]
    L += [f"![{f}]({f})" for f in files]
    p = run_dir / "report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p
