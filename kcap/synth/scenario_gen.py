from kcap.schemas import Domain, HiddenRule, Scenario


def generate_scenarios(llm, domain: Domain, rules: list[HiddenRule], n: int, min_scope: int, seed: int,
                       batch: int = 5) -> list[Scenario]:
    rt = "\n".join(f"{r.id}: trigger={r.trigger} | action={r.action} | reason={r.reason} | scope={r.scope}"
                   for r in rules)
    out: list[dict] = []
    while len(out) < n:
        k = min(batch, n - len(out))
        n_scope_have = sum(1 for s in out if s.get("depends_on_scope"))
        need = max(0, min_scope - n_scope_have)
        used = {rid for s in out for rid in s.get("rule_ids", [])}
        prompt = f"""Domain: {domain.name}. {domain.description}
HIDDEN RULES:
{rt}

Write {k} realistic scenarios a new employee might face. Each must be answerable correctly ONLY by someone who
knows at least one hidden rule. Prefer rules not yet used: {sorted(set(r.id for r in rules) - used)}.
At least {min(k, need)} must hinge on a rule's SCOPE exception (the correct answer differs because the
exception applies or does not); mark those "depends_on_scope": true.
Do not hint at the rule in the scenario. The reference answer states what to do and why in 1-3 sentences.
Return JSON {{"scenarios": [{{"prompt": "...", "reference_answer": "...", "rule_ids": ["R.."],
"depends_on_scope": false}}]}}."""
        res = llm.json("generator", "synth.scenarios", prompt, seed=seed * 100 + len(out))
        out.extend(res.get("scenarios", [])[:k])
    return [Scenario(id=f"S{i + 1:02d}", prompt=s["prompt"], reference_answer=s["reference_answer"],
                     rule_ids=s.get("rule_ids", []), topic="scope" if s.get("depends_on_scope") else "")
            for i, s in enumerate(out)]
