"""Shared gap-miner steps: entity clustering, workshop files, dedup + ranking."""
from collections import defaultdict

from kcap.components.corpus import PublicCorpus
from kcap.schemas import GapCard


def cluster_corpus(llm, corpus: PublicCorpus, min_size: int = 3, batch: int = 30) -> dict[str, list[str]]:
    assign: dict[str, str] = {}
    known: list[str] = []
    for b in range(0, len(corpus.docs), batch):
        chunk = corpus.docs[b:b + batch]
        prompt = f"""Assign each record to its MAIN entity (a specific part/equipment model, supplier, customer,
site, system or process). Reuse these existing entity names where they fit: {known[:60] or "(none yet)"}.
Records:
{chr(10).join(f"[{d.id}] {d.text[:250]}" for d in chunk)}
Return JSON {{"assignments": {{"<doc_id>": "<entity name>"}}}}."""
        out = llm.json("interviewer", "gap_miner.cluster", prompt, seed=b)
        for did, ent in out.get("assignments", {}).items():
            if did in corpus.by_id and isinstance(ent, str):
                assign[did] = ent.strip()
                if ent.strip() not in known:
                    known.append(ent.strip())
    groups = defaultdict(list)
    for d in corpus.docs:
        groups[assign.get(d.id, "misc")].append(d.id)
    out, misc = {}, []
    for ent, ids in groups.items():
        (out.__setitem__(ent, ids) if len(ids) >= min_size and ent != "misc" else misc.extend(ids))
    if misc:
        out["misc"] = misc
    return out


def workshop_file(corpus: PublicCorpus, entity: str, doc_ids: list[str], max_docs: int = 25) -> str:
    docs = [corpus.by_id[i] for i in doc_ids[:max_docs]]
    sops = corpus.sops()
    return f"""# Cluster: {entity}

## Organization
{corpus.description}

## Records about this entity ({len(doc_ids)} total, showing {len(docs)})
{corpus.render(docs, 500)}

## Official procedures that may apply (SOPs)
{chr(10).join(f"- [{s.id}] {s.text[:200]}" for s in sops) or "(none)"}

## Task
Find decisions, priorities or judgments in these documents that deviate from procedures or from what a newcomer
would do, and whose reasons no document explains. Cite document IDs.
"""


def finalize(raw: list[dict], corpus: PublicCorpus, cfg: dict, top_k: int) -> list[GapCard]:
    cards = []
    for c in raw:
        try:
            cards.append(GapCard(id="", hypothesis=c["hypothesis"], evidence_doc_ids=c["evidence_doc_ids"],
                                 importance=float(c.get("importance", 0.5)),
                                 undocumentedness=float(c.get("undocumentedness", 0.5)),
                                 opening_question=c.get("opening_question") or c["hypothesis"] + " Is that right?",
                                 cluster=c.get("cluster", "")))
        except (KeyError, ValueError, TypeError):
            continue
    cards.sort(key=lambda c: -c.priority)
    thr = cfg["gap_miner"]["dedup_threshold"] if cfg.get("retrieval", {}).get("embeddings") else \
        cfg["gap_miner"].get("dedup_threshold_lexical", 0.4)
    kept: list[GapCard] = []
    for c in cards:
        dup = any(corpus.index.similarity(c.hypothesis, k.hypothesis) >= thr or
                  (set(c.evidence_doc_ids) and set(c.evidence_doc_ids) == set(k.evidence_doc_ids)) for k in kept)
        if not dup:
            kept.append(c)
    return [c.model_copy(update={"id": f"G{i + 1:02d}"}) for i, c in enumerate(kept[:top_k])]
