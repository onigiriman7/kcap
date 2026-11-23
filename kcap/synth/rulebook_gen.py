from kcap.schemas import Domain, HiddenRule

RULE_SPEC = """Each rule is tacit expert knowledge: judgment the expert uses that is NOT written in any SOP.
Fields: "trigger" (the specific situation), "action" (what the expert does, concrete), "reason" (why, a
non-obvious causal explanation), "scope" (an exception or narrowing condition, e.g. "only for Supplier B" or
"except when ..."), "importance" (1-3), "visible_effects" (2-4 ways the behavior shows up in records WITHOUT
the reason being written: line items, timing patterns, notes like 'per JM', differing outcomes)."""


def generate_rulebook(llm, domain: Domain, n_rules: int, min_scope_exceptions: int, seed: int,
                      batch: int = 9) -> list[HiddenRule]:
    rules: list[dict] = []
    while len(rules) < n_rules:
        k = min(batch, n_rules - len(rules))
        need_scope = max(0, min_scope_exceptions - sum(1 for r in rules if r.get("_nonobvious")))
        prev = "\n".join(f"- {r['trigger']} -> {r['action']}" for r in rules) or "(none)"
        prompt = f"""Domain: {domain.name}
{domain.description}
Entities you may use: {", ".join(domain.entities)}

Write {k} hidden expert rules. {RULE_SPEC}
At least {min(k, need_scope)} of these must have a NON-OBVIOUS scope exception (one a newcomer would never guess);
mark them with "nonobvious_scope": true.
Rules must be diverse (different entities, processes, risks) and must not duplicate these existing rules:
{prev}
Return JSON {{"rules": [ ... ]}}."""
        out = llm.json("generator", "synth.rulebook", prompt, seed=seed * 100 + len(rules))
        for r in out.get("rules", [])[:k]:
            r["_nonobvious"] = bool(r.pop("nonobvious_scope", False))
            rules.append(r)
    return [HiddenRule(id=f"R{i + 1:02d}", **{k: v for k, v in r.items() if not k.startswith("_") and k != "id"})
            for i, r in enumerate(rules)]
