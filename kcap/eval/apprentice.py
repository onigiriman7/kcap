"""Apprentice: answers scenarios from the corpus (+ optionally captured KUs) only."""
from kcap.components.corpus import PublicCorpus
from kcap.retrieval.index import HybridIndex
from kcap.schemas import KnowledgeUnit, Scenario


def answer(llm, scenario: Scenario, corpus: PublicCorpus, kus: list[KnowledgeUnit] | None, cfg: dict,
           condition: str, seed: int = 0) -> str:
    docs = corpus.search(scenario.prompt, 6)
    kb = ""
    if kus:
        idx = HybridIndex.from_cfg([(k.id, k.text()) for k in kus], cfg)
        kb = "\n".join(f"- {t}" for _, t, _ in idx.search(scenario.prompt, 6))
    prompt = f"""You are a new employee. Decide what to do in this situation and say why, in 1-3 sentences.
Use the company knowledge below; if it does not cover the case, use your best judgment.

COMPANY RECORDS:
{corpus.render(docs, 400)}
{"EXPERT KNOWLEDGE BASE:" + chr(10) + kb if kb else ""}

SITUATION: {scenario.prompt}"""
    return llm.chat("apprentice", f"apprentice.{condition}", prompt, seed=seed).strip()
