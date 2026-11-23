"""Transcript -> KnowledgeUnits (extract per turn, merge, check against records) -> KB index."""
from kcap.components.corpus import PublicCorpus
from kcap.retrieval.index import HybridIndex
from kcap.schemas import KnowledgeUnit, Turn


def _extract(llm, turn: Turn, prev: Turn | None, seed: int) -> list[dict]:
    ctx = f"PREVIOUS Q: {prev.question}\nPREVIOUS A: {prev.answer}\n\n" if prev else ""
    prompt = f"""{ctx}Q: {turn.question}
A: {turn.answer}

Extract the concrete, reusable work knowledge the expert states in THIS answer (A), using the previous turn
only to resolve references. Ignore platitudes and generic advice. A knowledge unit is:
condition (when it applies), action (what to do), reason (why, if stated), scope (exceptions/limits, if stated).
Use null for missing fields. Do not invent anything not said.
Return JSON {{"units": [{{"condition": "...", "action": "...", "reason": null, "scope": null}}]}}."""
    return llm.json("interviewer", "synth.extract", prompt, seed=seed).get("units", [])


def _merge(llm, units: list[dict], seed: int) -> list[dict]:
    if len(units) <= 1:
        return units
    listing = "\n".join(f"{i}: WHEN {u['condition']} THEN {u['action']} | WHY {u.get('reason')} | SCOPE {u.get('scope')}"
                        for i, u in enumerate(units))
    prompt = f"""These knowledge units were extracted turn by turn from one interview. Merge units that describe
the SAME practice (combine their reason/scope details). Keep distinct practices separate.
{listing}
Return JSON {{"merged": [{{"condition": "...", "action": "...", "reason": null, "scope": null,
"from": [<indexes>]}}]}} covering every index exactly once."""
    out = llm.json("interviewer", "synth.merge", prompt, seed=seed).get("merged", [])
    res, covered = [], set()
    for m in out:
        idx = [i for i in m.get("from", []) if isinstance(i, int) and 0 <= i < len(units) and i not in covered]
        if not idx or not m.get("action"):
            continue
        covered |= set(idx)
        res.append({"condition": m.get("condition") or "", "action": m["action"], "reason": m.get("reason"),
                    "scope": m.get("scope"), "source_turns": sorted({t for i in idx for t in units[i]["source_turns"]})})
    res += [units[i] for i in range(len(units)) if i not in covered]
    return res


def _confidence(llm, kus: list[KnowledgeUnit], corpus: PublicCorpus, seed: int, batch: int = 8):
    for b in range(0, len(kus), batch):
        chunk = kus[b:b + batch]
        blocks = []
        for i, k in enumerate(chunk):
            docs = corpus.search(k.text(), 4)
            blocks.append(f"UNIT {i}: {k.text()}\nRECORDS:\n{corpus.render(docs, 300)}")
        prompt = """For each unit, compare it with the records shown:
"confirmed_by_records" = records show this behavior; "conflicts_with_records" = records show different
behavior; "stated" = records neither confirm nor contradict.
""" + "\n\n".join(blocks) + """
Return JSON {"checks": [{"unit": 0, "confidence": "...", "doc_ids": ["..."]}]}."""
        out = llm.json("interviewer", "synth.confidence", prompt, seed=seed + b)
        for c in out.get("checks", []):
            i = c.get("unit")
            if isinstance(i, int) and 0 <= i < len(chunk) and c.get("confidence") in (
                    "stated", "confirmed_by_records", "conflicts_with_records"):
                chunk[i].confidence = c["confidence"]
                chunk[i].source_docs = [d for d in c.get("doc_ids", []) if d in corpus.by_id]


def synthesize(llm, turns: list[Turn], corpus: PublicCorpus | None, seed: int = 0) -> list[KnowledgeUnit]:
    units = []
    for j, t in enumerate(turns):
        if not t.answer.strip():
            continue
        for u in _extract(llm, t, turns[j - 1] if j else None, seed + t.idx):
            if isinstance(u, dict) and u.get("action"):
                units.append({"condition": u.get("condition") or "", "action": u["action"],
                              "reason": u.get("reason"), "scope": u.get("scope"), "source_turns": [t.idx]})
    merged = _merge(llm, units, seed)
    kus = [KnowledgeUnit(id=f"KU{i + 1:03d}", **m) for i, m in enumerate(merged)]
    if corpus is not None and kus:
        _confidence(llm, kus, corpus, seed)
    return kus


def build_kb(kus: list[KnowledgeUnit], cfg: dict) -> HybridIndex:
    return HybridIndex.from_cfg([(k.id, k.text()) for k in kus], cfg)


def distill(llm, turns: list[Turn], description: str, seed: int = 0) -> list[KnowledgeUnit]:
    """Abstract an interview's past decisions into reusable principles for future, different cases."""
    tx = "\n\n".join(f"[turn {t.idx}] Q: {t.question}\nA: {t.answer}" for t in turns if t.answer.strip())
    if not tx:
        return []
    out = llm.json("interviewer", "synth.distill", f"""You turn an expert's interview into guidance for a successor.
{description}

INTERVIEW:
{tx}

The answers mostly describe PAST decisions. Do not restate them as instructions to repeat what was already done.
Instead, abstract the reusable judgment behind them: principles a successor can apply to a FUTURE, DIFFERENT
situation of the same kind. For each principle give:
- situation: the general class of situations it applies to (not the specific past case)
- principle: the default rule of thumb / how to decide (trade-off, priority, criterion)
- why: the reasoning or goal behind it
- exceptions: when it does not apply, and warning signs to watch for
- examples: 1-2 past decisions from the interview that illustrate it (short, with project names, cite [turn N])
- source_turns: the turn numbers it is based on
Only use what the expert said or clearly implied; skip "I don't know" answers; merge overlapping principles.
Prefer fewer, stronger principles over many weak ones. Write in English.
Return JSON {{"principles": [{{"situation": "...", "principle": "...", "why": "...", "exceptions": "...",
"examples": ["..."], "source_turns": [1]}}]}}.""", seed=seed)
    valid = {t.idx for t in turns}
    kus = []
    for p in out.get("principles", []):
        if not isinstance(p, dict) or not p.get("principle"):
            continue
        kus.append(KnowledgeUnit(id=f"P{len(kus) + 1:02d}", condition=p.get("situation") or "", action=p["principle"],
                                 reason=p.get("why"), scope=p.get("exceptions"),
                                 examples=[str(e) for e in p.get("examples") or []][:3],
                                 source_turns=[n for n in p.get("source_turns") or [] if n in valid]))
    return kus
