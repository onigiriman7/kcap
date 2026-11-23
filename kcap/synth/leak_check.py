"""Judge every document for statements of hidden-rule content; regenerate or drop leaking docs."""
from kcap.schemas import Document, Domain, HiddenRule


def _rules_text(rules: list[HiddenRule]) -> str:
    return "\n".join(f"{r.id}: action={r.action} | reason={r.reason} | scope={r.scope}" for r in rules)


def find_leaks(llm, docs: list[Document], rules: list[HiddenRule], seed: int = 0, batch: int = 8) -> dict[str, list]:
    """Return {doc_id: [{"rule_id", "facet"}]} for documents that STATE a rule's action, reason or scope."""
    leaks: dict[str, list] = {}
    rt = _rules_text(rules)
    for b in range(0, len(docs), batch):
        chunk = docs[b:b + batch]
        prompt = f"""HIDDEN RULES:
{rt}

DOCUMENTS:
{chr(10).join(f"[{d.id}] {d.text}" for d in chunk)}

A document LEAKS a rule facet if it explicitly states the rule as a rule/policy/habit, explains the reason, or
states the exception condition, so that a reader could learn the facet from this one document. Merely showing
the behavior (e.g. a $0 line item, a scheduling choice) is NOT a leak.
Return JSON {{"leaks": [{{"doc_id": "...", "rule_id": "...", "facet": "action|reason|scope"}}]}} (empty list if none)."""
        out = llm.json("judge", "synth.leak_check", prompt, seed=seed + b)
        ids = {d.id for d in chunk}
        for l in out.get("leaks", []):
            if l.get("doc_id") in ids:
                leaks.setdefault(l["doc_id"], []).append({"rule_id": l.get("rule_id"), "facet": l.get("facet")})
    return leaks


def _regenerate(llm, domain: Domain, doc: Document, issues: list, rules: dict, seed: int) -> Document:
    detail = "; ".join(f"{i['facet']} of rule {i['rule_id']} ({getattr(rules.get(i['rule_id']), i['facet'] or 'action', '')})"
                       for i in issues)
    prompt = f"""Domain: {domain.name}.
Rewrite this {doc.doc_type} so it keeps the same events and observable behavior but NO LONGER states: {detail}.
Remove explanations, policies and exception conditions; keep only what happened.
ORIGINAL:
{doc.text}
Return JSON {{"text": "..."}}."""
    out = llm.json("generator", "synth.leak_regen", prompt, seed=seed)
    return doc.model_copy(update={"text": out.get("text", doc.text)})


def leak_check(llm, domain: Domain, docs: list[Document], rules: list[HiddenRule], max_attempts: int = 3,
               seed: int = 0):
    by_rule = {r.id: r for r in rules}
    leaks = find_leaks(llm, docs, rules, seed)
    initial = len(leaks)
    docs = list(docs)
    dropped = []
    for attempt in range(max_attempts):
        if not leaks:
            break
        idx = {d.id: i for i, d in enumerate(docs)}
        fixed = [_regenerate(llm, domain, docs[idx[did]], issues, by_rule, seed + attempt * 7919 + idx[did])
                 for did, issues in leaks.items()]
        for d in fixed:
            docs[idx[d.id]] = d
        leaks = find_leaks(llm, fixed, rules, seed + 100 + attempt)
    if leaks:
        dropped = sorted(leaks)
        docs = [d for d in docs if d.id not in leaks]
    report = {"n_docs": len(docs) + len(dropped), "initial_leaking": initial,
              "initial_leak_rate": initial / max(1, len(docs) + len(dropped)),
              "dropped": dropped, "final_leak_rate": 0.0}
    return docs, report
