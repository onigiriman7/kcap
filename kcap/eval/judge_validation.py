"""Export judge decisions for hand-labelling, then compute agreement and Cohen's kappa (plan section 10)."""
import csv
import random
from collections import Counter
from pathlib import Path

from kcap.io import read_json, write_json


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def export(run_dir: Path, n_rule: int = 60, n_app: int = 40, seed: int = 0) -> Path:
    rng = random.Random(seed)
    rule_rows, app_rows = [], []
    for ev in sorted(Path(run_dir).glob("*/*/eval.json")):
        e = read_json(ev)
        tag = f"{ev.parent.parent.name}/{ev.parent.name}"
        for rid, fs in e.get("matches", {}).items():
            for f, m in fs.items():
                rule_rows.append({"kind": "rule_match", "src": f"{tag}/{rid}/{f}", "facet": f,
                                  "item": f"FACET TO CHECK: {f}\n{m['item']}", "judge": m["verdict"]})
        for g in e.get("apprentice", []):
            app_rows.append({"kind": "apprentice", "src": f"{tag}/{g['scenario_id']}/{g['condition']}",
                             "facet": "", "item": g["item"], "judge": str(int(round(g["score"])))})
    # stratify rule rows by (facet, captured vs not)
    strata: dict = {}
    for r in rule_rows:
        strata.setdefault((r["facet"], r["judge"] == "not"), []).append(r)
    picked = []
    per = max(1, n_rule // max(1, len(strata)))
    for rows in strata.values():
        picked += rng.sample(rows, min(per, len(rows)))
    picked += rng.sample(app_rows, min(n_app, len(app_rows)))
    rng.shuffle(picked)
    out = Path(run_dir) / "judge_validation"
    out.mkdir(exist_ok=True)
    key = {}
    with (out / "labels.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "kind", "allowed_labels", "item", "your_label"])
        for i, r in enumerate(picked):
            rid = f"J{i + 1:03d}"
            key[rid] = {"kind": r["kind"], "judge": r["judge"], "src": r["src"]}
            allowed = "captured|partially|not" if r["kind"] == "rule_match" else "0|1|2"
            w.writerow([rid, r["kind"], allowed, r["item"], ""])
    write_json(out / "judge_key.json", key)
    return out / "labels.csv"


def score(run_dir: Path, labels_csv: Path, prereg: dict | None = None) -> dict:
    key = read_json(Path(run_dir) / "judge_validation" / "judge_key.json")
    pairs: dict[str, list] = {"rule_match": [], "apprentice": []}
    with Path(labels_csv).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lab = (row.get("your_label") or "").strip().lower()
            if lab and row["row_id"] in key:
                k = key[row["row_id"]]
                pairs[k["kind"]].append((lab, k["judge"]))
    res = {}
    allp = []
    for kind, ps in pairs.items():
        allp += ps
        res[kind] = {"n": len(ps), "agreement": sum(h == j for h, j in ps) / len(ps) if ps else None,
                     "kappa": cohen_kappa([h for h, _ in ps], [j for _, j in ps]) if ps else None}
    res["overall"] = {"n": len(allp), "agreement": sum(h == j for h, j in allp) / len(allp) if allp else None}
    pj = (prereg or {}).get("judge", {"agreement_min": 0.85, "kappa_min": 0.7})
    res["gate_pass"] = all(v["agreement"] is not None and v["agreement"] >= pj["agreement_min"]
                           and v["kappa"] is not None and v["kappa"] >= pj["kappa_min"]
                           for k, v in res.items() if k in pairs and v["n"])
    write_json(Path(run_dir) / "judge_validation" / "result.json", res)
    return res
