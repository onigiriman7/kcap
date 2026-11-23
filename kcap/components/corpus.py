"""The only way components see documents: stripped of ground truth, loaded via the path guard."""
from pathlib import Path

from kcap.io import public_path, read_jsonl
from kcap.retrieval.index import HybridIndex
from kcap.schemas import Document


def strip(doc: Document) -> Document:
    return doc.model_copy(update={"related_rule_ids": []})


class PublicCorpus:
    def __init__(self, docs: list[Document], cfg: dict, description: str = ""):
        self.docs = [strip(d) for d in docs]
        self.by_id = {d.id: d for d in self.docs}
        self.description = description
        self.cfg = cfg
        self.index = HybridIndex.from_cfg([(d.id, f"[{d.doc_type} {d.date}] {d.text}") for d in self.docs], cfg)

    @classmethod
    def load(cls, path: Path, cfg: dict, description: str = "") -> "PublicCorpus":
        rows = read_jsonl(public_path(path))
        return cls([Document(**r) for r in rows], cfg, description)

    def search(self, query: str, k: int = 5, exclude=None) -> list[Document]:
        return [self.by_id[i] for i, _, _ in self.index.search(query, k, exclude)]

    def render(self, docs: list[Document], max_chars: int = 900) -> str:
        return "\n\n".join(f"[{d.id} | {d.doc_type} | {d.date}]\n{d.text[:max_chars]}" for d in docs)

    def sops(self) -> list[Document]:
        return [d for d in self.docs if d.doc_type == "sop"]
