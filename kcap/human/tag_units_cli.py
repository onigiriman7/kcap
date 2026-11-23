"""Owner step 10a: tag every captured knowledge unit."""
from pathlib import Path

from kcap.human.term import ask_choice
from kcap.io import read_json, write_json
from kcap.schemas import KnowledgeUnit


def tag(exp_dir: Path, arms=("A", "D")):
    out_p = exp_dir / "tags.json"
    tags = read_json(out_p) if out_p.exists() else {}
    for half in arms:
        p = exp_dir / f"kus_{half}.json"
        if not p.exists():
            continue
        for k in [KnowledgeUnit(**x) for x in read_json(p)]:
            key = f"{half}:{k.id}"
            if key in tags:
                continue
            print("=" * 80 + f"\n[{key}] {k.text()}\n")
            t = ask_choice("Is this true?", ["y", "n"])
            d = ask_choice("Is it already documented somewhere?", ["y", "n"])
            u = ask_choice("Would you have written this down unprompted?", ["y", "n"])
            tags[key] = {"true": t == "y", "documented": d == "y", "unprompted": u == "y",
                         "surprise": t == "y" and d == "n" and u == "n"}
            write_json(out_p, tags)
    print(f"Tagged {len(tags)} units. Surprises: {sum(v['surprise'] for v in tags.values())}")
    return tags
