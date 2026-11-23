"""Owner step 10b: blind-grade apprentice answers (condition hidden, order shuffled)."""
import random
from pathlib import Path

from kcap.eval.judge_validation import cohen_kappa
from kcap.human.term import ask_choice
from kcap.io import read_json, write_json


def blind_grade(exp_dir: Path, scenarios: dict, seed: int = 0):
    answers = read_json(exp_dir / "apprentice.json")
    out_p = exp_dir / "owner_grades.json"
    grades = read_json(out_p) if out_p.exists() else {}
    order = list(range(len(answers)))
    random.Random(seed).shuffle(order)
    print("Grade each answer by your own judgment: 0 = wrong/harmful, 1 = partly right (misses a key condition or "
          "exception), 2 = what you would actually do.\n")
    for n, i in enumerate(order, 1):
        a = answers[i]
        key = f"{a['scenario_id']}|{a['condition']}"
        if key in grades:
            continue
        sc = scenarios[a["scenario_id"]]
        ref = f"YOUR REFERENCE: {sc.reference_answer}\n\n" if sc.reference_answer.strip() else ""
        print("=" * 80 + f"\n({n}/{len(order)}) SITUATION: {sc.prompt}\n\n{ref}ANSWER: {a['answer']}\n")
        grades[key] = int(ask_choice("Score", ["0", "1", "2"]))
        write_json(out_p, grades)
    pairs = [(grades[f"{a['scenario_id']}|{a['condition']}"], int(round(a["score"]))) for a in answers
             if f"{a['scenario_id']}|{a['condition']}" in grades and a.get("score") is not None]
    agree = sum(h == j for h, j in pairs) / len(pairs) if pairs else None
    res = {"n": len(pairs), "agreement": agree, "kappa": cohen_kappa([h for h, _ in pairs], [j for _, j in pairs])}
    write_json(exp_dir / "owner_vs_judge.json", res)
    print(f"Owner vs LLM judge: agreement {agree}, kappa {res['kappa']}")
    return grades
