"""Extract candidate maintainer rules from review comments; keep those the docs do not state."""
from pathlib import Path

from kcap.components.corpus import PublicCorpus
from kcap.io import read_jsonl, write_json
from kcap.retrieval.index import HybridIndex


def extract(llm, cfg: dict, rdir: Path, batch: int = 15, log=print) -> list[dict]:
    reviews = read_jsonl(rdir / "private" / "reviews.jsonl")
    corpus = PublicCorpus.load(rdir / "public" / "corpus.jsonl", cfg)
    cands = []
    for b in range(0, len(reviews), batch):
        chunk = reviews[b:b + batch]
        out = llm.json("judge", "oss.extract", f"""These are code-review comments by maintainers. Extract project
RULES they enforce: recurring conventions or judgments like "we don't do X because Y", "always Z in module M".
Skip one-off bug comments, style nits handled by linters, and questions.
{chr(10).join(f"[{c['id']}] ({c.get('path')}) {c['body']}" for c in chunk)}
Return JSON {{"rules": [{{"trigger": "<situation>", "action": "...", "reason": "<or empty>", "scope": "<or empty>",
"importance": 1-3, "comment_ids": ["C..."]}}]}}.""", seed=b)
        cands += out.get("rules", [])
        log(f"extracted from {b + len(chunk)}/{len(reviews)} comments: {len(cands)} candidates")
    kept = []
    for i, r in enumerate(cands):
        docs = corpus.search(f"{r.get('trigger', '')} {r.get('action', '')}", 5)
        v = llm.json("judge", "oss.doccheck", f"""RULE: when {r.get('trigger')}, {r.get('action')} ({r.get('reason')})
DOCUMENTATION EXCERPTS:
{corpus.render(docs, 700)}
Does the documentation STATE this rule (so a newcomer reading the docs would know it)?
Return JSON {{"documented": true|false, "doc_ids": ["..."]}}.""", seed=i)
        r["documented"] = bool(v.get("documented"))
        r["doc_hits"] = [d.id for d in docs]
        if not r["documented"]:
            kept.append(r)
    by_id = {c["id"]: c for c in reviews}
    ridx = HybridIndex.from_cfg([(c["id"], c["body"]) for c in reviews], cfg)
    for i, r in enumerate(kept):
        ids = list(dict.fromkeys(r.get("comment_ids", []) +
                                 [x for x, _, _ in ridx.search(f"{r.get('trigger')} {r.get('action')}", 6)]))
        shown = [by_id[x] for x in ids if x in by_id][:8]
        v = llm.json("judge", "oss.support", f"""RULE: when {r.get('trigger')}, {r.get('action')}
REVIEW COMMENTS:
{chr(10).join(f"[{c['id']}] ({c.get('author')}) {c['body'][:600]}" for c in shown)}
Which comments state or enforce this rule as a GENERAL project practice (not a one-off fix)?
Return JSON {{"supporting_ids": ["C..."], "is_general_rule": true|false, "why": "<one sentence>"}}.""", seed=i)
        r["support_ids"] = [x for x in v.get("supporting_ids", []) if x in by_id]
        r["support_authors"] = sorted({by_id[x].get("author") for x in r["support_ids"]})
        r["llm_general"] = bool(v.get("is_general_rule"))
        r["llm_why"] = v.get("why", "")
    write_json(rdir / "private" / "candidate_rules.json", kept)
    log(f"{len(kept)} undocumented candidates (of {len(cands)})")
    return kept
