"""Non-Sakana baseline gap miner: one prompt per cluster, no tools, no reflection."""
from kcap.components.corpus import PublicCorpus
from kcap.components.gap_common import cluster_corpus, finalize, workshop_file
from kcap.schemas import GapCard


def mine_simple(llm, corpus: PublicCorpus, cfg: dict, seed: int = 0, clusters: dict | None = None) -> list[GapCard]:
    g = cfg["gap_miner"]
    clusters = clusters or cluster_corpus(llm, corpus, g["min_cluster_size"])
    raw = []
    for ci, (ent, ids) in enumerate(clusters.items()):
        prompt = workshop_file(corpus, ent, ids) + f"""
Propose up to {g['generations_per_cluster']} hypotheses about undocumented expert rules.
Return JSON {{"gap_cards": [{{"hypothesis": "...", "evidence_doc_ids": ["..."], "importance": 0.0,
"undocumentedness": 0.0, "opening_question": "<hypothesis-correction question citing evidence>"}}]}}."""
        out = llm.json("interviewer", "gap_miner.simple", prompt, seed=seed * 100 + ci)
        for c in out.get("gap_cards", []):
            c["evidence_doc_ids"] = [d for d in c.get("evidence_doc_ids", []) if d in corpus.by_id]
            if c.get("hypothesis") and c["evidence_doc_ids"]:
                raw.append(c | {"cluster": ent})
    return finalize(raw, corpus, cfg, g["top_k"])
