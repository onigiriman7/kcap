"""Strategy B: staged interview (basics -> workflow -> details), using only a corpus summary for terminology."""
import random

from kcap.components.corpus import PublicCorpus
from kcap.components.interviewer.base import Interviewer
from kcap.schemas import Turn

STAGES = [("basics", "the role, responsibilities, main customers/equipment, key terms"),
          ("workflow", "how typical jobs flow end to end, hand-offs, scheduling, ordering, approvals"),
          ("details", "specific judgment calls, edge cases, thresholds, special handling")]


class StagedInterviewer(Interviewer):
    name = "B"

    def __init__(self, llm, corpus: PublicCorpus, seed: int = 0, focus: list[str] | None = None):
        self.llm, self.corpus, self.seed = llm, corpus, seed
        self.focus = focus or []
        self.questions: list[tuple[str, str]] | None = None

    def _summary(self) -> str:
        rng = random.Random(self.seed)
        sample = rng.sample(self.corpus.docs, min(40, len(self.corpus.docs)))
        return self.llm.chat("interviewer", "B.summary", f"""Summarize the organization's work, terminology,
entities and record types in under 250 words. Do NOT list specific incidents or patterns.
{self.corpus.description}
{self.corpus.render(sample, 250)}""", seed=self.seed).strip()

    def _plan(self, n: int):
        summary = self._summary()
        if self.focus:
            summary += "\n\nRestrict the interview to these areas: " + "; ".join(self.focus)
        sizes = [n // 3, n // 3, n - 2 * (n // 3)]
        qs = []
        for (stage, focus), k in zip(STAGES, sizes):
            if k <= 0:
                continue
            out = self.llm.json("interviewer", "B.plan", f"""You are preparing a structured knowledge-transfer
interview with a departing expert. Organization summary:
{summary}

Write {k} questions for the "{stage}" stage ({focus}). Each question must include a short example answer
to show the expected level of detail, formatted as: "<question> (For example: <example answer>)".
Questions so far: {[q for q, _ in qs]}
Return JSON {{"questions": ["..."]}}.""", seed=self.seed + len(qs))
            qs += [(q, stage) for q in out.get("questions", [])[:k]]
        while len(qs) < n:
            qs.append(("Is there anything else about the details of your work a successor should know?", "details"))
        self.questions = qs

    def next_question(self, turns, n):
        if self.questions is None:
            self._plan(n)
        q, stage = self.questions[len(turns)]
        return Turn(idx=0, strategy=self.name, question=q, question_kind="staged", meta={"stage": stage})
