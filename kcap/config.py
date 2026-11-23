import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from kcap.io import CONFIG_DIR, ROOT


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def load_prereg() -> dict | None:
    p = CONFIG_DIR / "prereg.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None


def require_committed_prereg():
    """Rule 6: full runs need a committed, unmodified prereg.yaml."""
    p = CONFIG_DIR / "prereg.yaml"
    if not p.exists():
        raise SystemExit("config/prereg.yaml missing. Copy prereg.example.yaml, fill it in, commit it.")
    rel = p.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT, capture_output=True)
    if tracked.returncode != 0 or dirty.returncode != 0:
        raise SystemExit("config/prereg.yaml must be committed and unmodified before a full run.")
    head = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True, text=True)
    return head.stdout.strip()
