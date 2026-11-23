"""Gap miner built on the AI Scientist ideation loop (via sakana_adapters/ideation_adapter)."""
from pathlib import Path

from kcap.components.corpus import PublicCorpus
from kcap.components.gap_common import cluster_corpus, finalize, workshop_file
from kcap.io import write_json
from kcap.sakana_adapters.ideation_adapter import generate_temp_free_gap_cards
from kcap.schemas import GapCard


def mine(llm, corpus: PublicCorpus, cfg: dict, out_dir: Path | None = None, seed: int = 0,
         clusters: dict | None = None) -> list[GapCard]:
    g = cfg["gap_miner"]
    clusters = clusters or cluster_corpus(llm, corpus, g["min_cluster_size"])
    raw, log = [], []
    for ci, (ent, ids) in enumerate(clusters.items()):
        ws = workshop_file(corpus, ent, ids)
        if out_dir:
            (Path(out_dir) / "workshops").mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "workshops" / f"{ci:02d}.md").write_text(ws, encoding="utf-8")
        for c in generate_temp_free_gap_cards(llm, corpus, ws, g["generations_per_cluster"], g["num_reflections"],
                                              seed=seed * 100 + ci, log=log):
            raw.append(c | {"cluster": ent})
    cards = finalize(raw, corpus, cfg, g["top_k"])
    if out_dir:
        write_json(Path(out_dir) / "ideation_log.json", log)
        write_json(Path(out_dir) / "clusters.json", clusters)
    return cards
