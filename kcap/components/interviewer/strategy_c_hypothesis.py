"""Strategy C: hypothesis + correction over gap cards with a fixed follow-up policy.
With use_contradiction=True this is the C+K ablation (contradiction checker, no search)."""
from kcap.components.contradiction import ContradictionChecker
from kcap.components.interviewer.base import Interviewer
from kcap.schemas import GapCard, Turn

FOLLOWUPS = [("why", "why they do it this way"), ("exceptions", "exceptions, edge cases, or when it does NOT apply")]


class HypothesisInterviewer(Interviewer):
    def __init__(self, llm, cards: list[GapCard], followups: int = 2, checker: ContradictionChecker | None = None,
                 seed: int = 0):
        self.llm, self.seed = llm, seed
        self.cards = sorted(cards, key=lambda c: -c.priority)
        self.followups = followups
        self.checker = checker
        self.name = "C+K" if checker else "C"
        self.ci = 0
        self.pending: list[tuple[str, str]] = []
        self.current: GapCard | None = None

    def observe(self, turns):
        if self.checker:
            t = turns[-1]
            self.checker.check_answer(t.question, t.answer, t.idx)

    def _phrase_followup(self, turns, kind, desc):
        last = turns[-1]
        out = self.llm.json("interviewer", "C.followup", f"""You are interviewing an expert about this hypothesis:
{self.current.hypothesis}
Last question: {last.question}
Expert answer: {last.answer}

1) Did the expert REJECT the hypothesis outright (it is not something they do)? 
2) Write ONE short follow-up asking about {desc}.
Return JSON {{"rejected": true|false, "question": "..."}}.""", seed=self.seed + len(turns))
        return bool(out.get("rejected")), out.get("question") or f"Can you tell me {desc}?"

    def next_question(self, turns, n):
        while self.pending and turns:
            kind, desc = self.pending.pop(0)
            rejected, q = self._phrase_followup(turns, kind, desc)
            if rejected:
                self.pending = []
                break
            return Turn(idx=0, strategy=self.name, question=q, question_kind="deepen",
                        gap_card_id=self.current.id, meta={"followup": kind})
        if self.checker and self.checker.queue:
            item = self.checker.pop()
            return Turn(idx=0, strategy=self.name, question=item.question, question_kind="contradiction",
                        meta={"contradiction_id": item.id, "doc_ids": item.doc_ids})
        if self.ci < len(self.cards):
            self.current = self.cards[self.ci]
            self.ci += 1
            self.pending = FOLLOWUPS[:self.followups]
            return Turn(idx=0, strategy=self.name, question=self.current.opening_question,
                        question_kind="hypothesis", gap_card_id=self.current.id)
        return Turn(idx=0, strategy=self.name, question="Is there anything related to these topics that we haven't covered?",
                    question_kind="generic")
