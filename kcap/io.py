"""Paths, JSON(L) helpers and the leakage path guard."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("KCAP_ROOT", Path(__file__).resolve().parents[1]))
CONFIG_DIR = ROOT / "config"
RUNS = ROOT / "runs"
SEALED = ROOT / "sealed"

# Path components that hold ground truth. Components may never read below these.
GUARDED_PARTS = ("sealed", "private")


class LeakageError(RuntimeError):
    pass


def public_path(p) -> Path:
    p = Path(p).resolve()
    if any(part in GUARDED_PARTS for part in p.parts):
        raise LeakageError(f"components may not read ground-truth path: {p}")
    return p


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def read_jsonl(p):
    out = []
    # split on "\n" only: splitlines() also breaks on U+2028/U+0085 inside JSON strings
    for line in Path(p).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            out.append(json.loads(line))
    return out


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, default=_default) + "\n" for r in rows), encoding="utf-8")


def append_jsonl(p, row):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_default) + "\n")


def _default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    raise TypeError(type(o))
