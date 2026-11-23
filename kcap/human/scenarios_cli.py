"""Owner step 8: write sealed scenarios + reference answers, and record their SHA-256 in prereg.yaml."""
import hashlib
import json

import yaml

from kcap.human.term import read_multiline
from kcap.io import CONFIG_DIR, SEALED, write_json
from kcap.schemas import Scenario

SEALED_FILE = SEALED / "scenarios.json"


def sha256_file(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_and_seal(n: int) -> str:
    if SEALED_FILE.exists():
        raise SystemExit(f"{SEALED_FILE} already exists. Sealed scenarios must not be rewritten.")
    print(f"Write {n} scenarios from your past project. Each needs a realistic situation and the answer YOU "
          "would give (what to do + why). Type /done to stop early.\n")
    scen = []
    for i in range(n):
        print(f"--- Scenario {i + 1}/{n} ---")
        topic = input("Short topic label (e.g. 'supplier B returns'): ").strip()
        prompt = read_multiline("Situation:")
        if prompt == "/done":
            break
        ref = read_multiline("Your reference answer (what to do and why):")
        scen.append(Scenario(id=f"H{i + 1:02d}", prompt=prompt, reference_answer=ref, topic=topic))
    return seal(scen)


def seal(scen: list[Scenario]) -> str:
    if SEALED_FILE.exists():
        raise SystemExit(f"{SEALED_FILE} already exists. Sealed scenarios must not be rewritten.")
    write_json(SEALED_FILE, scen)
    h = sha256_file(SEALED_FILE)
    p = CONFIG_DIR / "prereg.yaml"
    if p.exists():
        y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        y.setdefault("exp3", {})["sealed_scenarios_sha256"] = h
        p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
        print("Hash written to config/prereg.yaml. Commit it now: git add config/prereg.yaml && git commit")
    print(f"Sealed {len(scen)} scenarios. SHA-256: {h}")
    return h


def load_sealed(expected_sha: str | None) -> list[Scenario]:
    """Eval-only access. Refuses if the file changed since sealing."""
    h = sha256_file(SEALED_FILE)
    if expected_sha and h != expected_sha:
        raise SystemExit(f"sealed scenarios hash mismatch: {h} != prereg {expected_sha}")
    return [Scenario(**s) for s in json.loads(SEALED_FILE.read_text(encoding="utf-8"))]
