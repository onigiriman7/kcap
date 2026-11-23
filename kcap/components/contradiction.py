"""Contradiction checker: claims in answers (and SOPs) vs. operational records."""
from kcap.components.corpus import PublicCorpus
from kcap.schemas import ContradictionItem


class ContradictionChecker:
    def __init__(self, llm, corpus: PublicCorpus, seed: int = 0):
        self.llm, self.corpus, self.seed = llm, corpus, seed
        self.queue: list[ContradictionItem] = []
        self._n = 0
        self._seen: list[set[str]] = []

    def _enqueue(self, items: list[dict], source: str):
        added = []
        for it in items:
            ids = [d for d in it.get("doc_ids", []) if d in self.corpus.by_id]
            if not ids or not it.get("question"):
                continue
            if any(len(set(ids) & s) >= max(1, len(ids) // 2) for s in self._seen):
                continue
            self._n += 1
            item = ContradictionItem(id=f"X{self._n:02d}", claim=it.get("claim", ""), doc_ids=ids,
                                     question=it["question"], source=source)
            self._seen.append(set(ids))
            self.queue.append(item)
            added.append(item)
        return added

    def _judge(self, claims_with_docs: list[tuple[str, list]], task: str, seed: int) -> list[dict]:
        blocks = "\n\n".join(f"CLAIM {i + 1}: {c}\nRECORDS:\n{self.corpus.render(ds, 400)}"
                             for i, (c, ds) in enumerate(claims_with_docs))
        prompt = f"""For each claim, decide if the records are consistent with it, conflict with it, or are unrelated.
A conflict means records show behavior different from what the claim says (e.g. an SOP default that is not
followed, or an expert statement the records contradict).
{blocks}

For each CONFLICT, write a neutral follow-up question to the expert citing the record IDs, e.g.
"You said X, but WO-1182 shows Y. What explains the difference?"
Return JSON {{"conflicts": [{{"claim": "...", "doc_ids": ["..."], "question": "..."}}]}}."""
        return self.llm.json("interviewer", task, prompt, seed=seed).get("conflicts", [])

    def pre_interview(self) -> list[ContradictionItem]:
        added = []
        for i, sop in enumerate(self.corpus.sops()):
            ops = [d for d in self.corpus.search(sop.text[:600], 10, exclude={sop.id}) if d.doc_type != "sop"][:8]
            if ops:
                added += self._enqueue(self._judge([(f"[{sop.id}] {sop.text[:800]}", ops)],
                                                   "contradiction.pre", self.seed + i), "pre_interview")
        return added

    def check_answer(self, question: str, answer: str, turn_idx: int) -> list[ContradictionItem]:
        out = self.llm.json("interviewer", "contradiction.claims", f"""Extract up to 5 atomic factual claims about
how work is done from this interview answer. Skip opinions and platitudes.
Q: {question}
A: {answer}
Return JSON {{"claims": ["..."]}}.""", seed=self.seed + turn_idx)
        claims = [c for c in out.get("claims", []) if isinstance(c, str)][:5]
        if not claims:
            return []
        pairs = [(c, self.corpus.search(c, 5)) for c in claims]
        return self._enqueue(self._judge(pairs, "contradiction.judge", self.seed + 7 * turn_idx), "answer")

    def pop(self, item_id: str | None = None) -> ContradictionItem | None:
        if not self.queue:
            return None
        idx = next((i for i, x in enumerate(self.queue) if x.id == item_id), 0)
        return self.queue.pop(idx)
