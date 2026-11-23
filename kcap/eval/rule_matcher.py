"""Judge whether hidden rules were captured by KUs (per facet) and whether gap cards hit real rules."""
from kcap.retrieval.index import HybridIndex
from kcap.sakana_adapters.review_adapter import perform_review
from kcap.schemas import GapCard, HiddenRule, KnowledgeUnit

FACETS = ("action", "reason", "scope")


def match_rules(llm, rules: list[HiddenRule], kus: list[KnowledgeUnit], cfg: dict, k: int = 8) -> dict:
    """Return {rule_id: {facet: {"verdict", "ku_id", "turn", "item"}}}."""
    j = cfg.get("judge", {})
    by_id = {u.id: u for u in kus}
    idx = HybridIndex.from_cfg([(u.id, u.text()) for u in kus], cfg) if kus else None
    out = {}
    for r in rules:
        rule_txt = f"HIDDEN RULE {r.id}\n  situation: {r.trigger}\n  action: {r.action}\n  reason: {r.reason}\n  scope: {r.scope}"
        cands = [by_id[i] for i, _, _ in idx.search(f"{r.trigger} {r.action} {r.scope}", k)] if idx else []
        item = rule_txt + "\n\nKNOWLEDGE UNITS:\n" + ("\n".join(f"{u.id}: {u.text()}" for u in cands) or "(none)")
        if not cands:
            out[r.id] = {f: {"verdict": "not", "ku_id": None, "turn": None, "item": item} for f in FACETS}
            continue
        rev = perform_review(llm, "rule_match", item, "eval.rule_match", j.get("ensemble", 3),
                             j.get("num_reflections", 1))
        res = {}
        for f in FACETS:
            v = rev.get(f) or {}
            verdict = v.get("verdict", "not") if isinstance(v, dict) else "not"
            ku = by_id.get(v.get("ku_id")) if isinstance(v, dict) else None
            if ku is None and verdict != "not":
                ku = cands[0]  # judge gave no/invalid id: attribute to the best retrieved KU
            turn = min(ku.source_turns) if ku and ku.source_turns and verdict != "not" else None
            res[f] = {"verdict": verdict, "ku_id": ku.id if ku else None, "turn": turn, "item": item}
        out[r.id] = res
    return out


def to_grades(matches: dict, strict: bool = True) -> dict:
    """{rule: {facet: first turn captured or None}}, the pilot's grades format."""
    ok = ("captured",) if strict else ("captured", "partially")
    return {rid: {f: (m["turn"] if m["verdict"] in ok else None) for f, m in fs.items()} for rid, fs in matches.items()}


def match_gap_cards(llm, cards: list[GapCard], rules: list[HiddenRule], cfg: dict) -> dict:
    j = cfg.get("judge", {})
    rules_txt = "\n".join(f"{r.id}: when {r.trigger} -> {r.action} (scope: {r.scope})" for r in rules)
    valid = {r.id for r in rules}
    out = {}
    for c in cards:
        rev = perform_review(llm, "gap_card_match", f"HYPOTHESIS: {c.hypothesis}\n\nRULES:\n{rules_txt}",
                             "eval.gap_card_match", j.get("ensemble", 3), j.get("num_reflections", 1))
        rid = rev.get("rule_id")
        out[c.id] = rid if rid in valid else None
    return out
