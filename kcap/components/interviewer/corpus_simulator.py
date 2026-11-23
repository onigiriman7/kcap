"""Strategy D's internal 'expert': plays a plausible expert from corpus snippets only.
Must never import from kcap.synth or kcap.persona (rule 2)."""
from kcap.components.corpus import PublicCorpus
from kcap.components.interviewer.base import recent


class CorpusSimulator:
    def __init__(self, llm, corpus: PublicCorpus, seed: int = 0):
        self.llm, self.corpus, self.seed = llm, corpus, seed

    def answer(self, question: str, turns, salt: int = 0) -> str:
        docs = self.corpus.search(question, 5)
        return self.llm.chat("simulator", "D.simulate", f"""Play an experienced employee of this organization.
You only know what these records show; where records are silent, guess plausibly and say you are unsure.
{self.corpus.description}

RECORDS:
{self.corpus.render(docs, 400)}

INTERVIEW SO FAR:
{recent(turns, 4)}

QUESTION: {question}
Answer briefly (2-4 sentences).""", seed=self.seed + salt).strip()
