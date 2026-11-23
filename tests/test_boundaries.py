"""Boundary tests: no ground-truth leakage, Sakana code only via adapters."""
import ast
import subprocess
from pathlib import Path

import pytest

from kcap.components.corpus import PublicCorpus, strip
from kcap.io import LeakageError, ROOT, write_jsonl

PKG = ROOT / "kcap"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    return mods


def _closure(start: Path) -> set[str]:
    """All kcap modules transitively imported from `start`."""
    seen, todo = set(), [start]
    while todo:
        p = todo.pop()
        for m in _imports(p):
            if m.startswith("kcap") and m not in seen:
                seen.add(m)
                f = ROOT / (m.replace(".", "/") + ".py")
                if not f.exists():
                    f = ROOT / m.replace(".", "/") / "__init__.py"
                if f.exists():
                    todo.append(f)
    return seen


@pytest.mark.parametrize("f", sorted((PKG / "components").rglob("*.py")), ids=lambda p: p.name)
def test_no_leakage_imports(f):
    reach = _closure(f)
    bad = [m for m in reach if m.startswith(("kcap.synth", "kcap.persona", "kcap.eval", "kcap.runners",
                                             "kcap.human", "kcap.oss"))]
    assert not bad, f"{f.name} can reach ground-truth modules: {bad}"
    src = f.read_text(encoding="utf-8")
    assert "sealed" not in src and "private" not in src and "SEALED" not in src


def test_runtime_path_guard(tmp_path, cfg):
    for part in ("private", "sealed"):
        p = tmp_path / part / "corpus.jsonl"
        write_jsonl(p, [])
        with pytest.raises(LeakageError):
            PublicCorpus.load(p, cfg)


def test_corpus_stripping(docs, cfg):
    corpus = PublicCorpus(docs, cfg)
    assert all(d.related_rule_ids == [] for d in corpus.docs)
    assert strip(docs[2]).related_rule_ids == [] and docs[2].related_rule_ids == ["R01"]


def test_simulator_is_corpus_only():
    reach = _closure(PKG / "components" / "interviewer" / "corpus_simulator.py")
    assert not any(m.startswith(("kcap.synth", "kcap.persona")) for m in reach)


def test_sakana_boundary():
    for f in PKG.rglob("*.py"):
        if "sakana_adapters" in f.parts:
            continue
        mods = _imports(f)
        assert not any(m.split(".")[0] in ("treequest", "ai_scientist", "third_party") for m in mods), f
    sub = ROOT / "third_party" / "AI-Scientist-v2"
    if (sub / ".git").exists():
        r = subprocess.run(["git", "diff", "--exit-code", "--quiet"], cwd=sub)
        assert r.returncode == 0, "AI-Scientist-v2 submodule has been modified"
