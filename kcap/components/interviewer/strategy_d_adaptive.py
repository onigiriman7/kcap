"""Strategy D: TreeQuest (AB-MCTS) picks each next question via offline search against the corpus simulator.
use_contradiction=False gives the C+S ablation (search, no contradiction checker)."""
import random

from kcap.components.contradiction import ContradictionChecker
from kcap.components.corpus import PublicCorpus
from kcap.components.interviewer.base import Interviewer, recent
from kcap.components.interviewer.corpus_simulator import CorpusSimulator
from kcap.sakana_adapters import treequest_adapter
from kcap.schemas import GapCard, Turn


class AdaptiveInterviewer(Interviewer):
    def __init__(self, llm, corpus: PublicCorpus, cards: list[GapCard], cfg: dict,
                 checker: ContradictionChecker | None = None, seed: int = 0, log: list | None = None):
        self.llm, self.corpus, self.seed = llm, corpus, seed
        self.cards = sorted(cards, key=lambda c: -c.priority)
        self.by_id = {c.id: c for c in cards}
        self.explored: set[str] = set()
        self.current: GapCard | None = None
        self.checker = checker
        self.sim = CorpusSimulator(llm, corpus, seed)
        d = cfg["strategy_d"]
        self.steps, self.w = d["search_steps"], d["weights"]
        self.name = "D" if checker else "C+S"
        self.log = log if log is not None else []
        self._salt = 0
        self._turns: list[Turn] = []

    def observe(self, turns):
        if self.checker:
            t = turns[-1]
            self.checker.check_answer(t.question, t.answer, t.idx)

    # ---- generate functions: parent_state | None -> (state, score) ----
    def _salt_next(self):
        self._salt += 1
        return self.seed * 100000 + len(self._turns) * 1000 + self._salt

    def _refine(self, parent, kind: str, base_q: str, salt: int) -> str:
        if parent is None or parent["kind"] != kind:
            return base_q
        return self.llm.chat("interviewer", "D.refine", f"""Improve this interview question so the expert's answer
reveals more specific, undocumented knowledge (conditions, numbers, exceptions). Keep it one question.
Simulated answer it got: {parent['simulated_answer']}
QUESTION: {parent['candidate_question']}
Output only the improved question.""", seed=salt).strip() or base_q

    def _deepen(self, parent):
        salt = self._salt_next()
        last = self._turns[-1]
        if parent is not None and parent["kind"] == "deepen":
            q = self._refine(parent, "deepen", parent["candidate_question"], salt)
        else:
            q = self.llm.chat("interviewer", "D.deepen", f"""You interview a departing expert.
Current hypothesis: {self.current.hypothesis if self.current else "(none)"}
INTERVIEW SO FAR:
{recent(self._turns, 6)}

Write ONE follow-up on the last answer that would reveal something not yet said: the reason, an exception,
a what-if, or how they notice the situation. Do NOT ask about anything already answered.
Output only the question.""", seed=salt).strip()
        return self._score({"candidate_question": q, "kind": "deepen",
                            "gap_card_id": self.current.id if self.current else None}, salt)

    def _widen(self, parent):
        salt = self._salt_next()
        todo = [c for c in self.cards if c.id not in self.explored]
        if parent is not None and parent["kind"] == "widen":
            card = self.by_id[parent["gap_card_id"]]
            q = self._refine(parent, "widen", card.opening_question, salt)
        else:
            card = random.Random(salt).choice(todo[:3])
            q = card.opening_question
        return self._score({"candidate_question": q, "kind": "widen", "gap_card_id": card.id}, salt)

    def _contradict(self, parent):
        salt = self._salt_next()
        if parent is not None and parent["kind"] == "contradiction":
            item_id = parent["contradiction_id"]
            q = self._refine(parent, "contradiction", parent["candidate_question"], salt)
        else:
            item = random.Random(salt).choice(self.checker.queue[:3])
            item_id, q = item.id, item.question
        return self._score({"candidate_question": q, "kind": "contradiction", "contradiction_id": item_id,
                            "gap_card_id": None}, salt)

    def _score(self, state, salt):
        sim = self.sim.answer(state["candidate_question"], self._turns, salt)
        out = self.llm.json("interviewer", "D.score", f"""Rate the expected information value of asking a departing
expert this question, given a plausible answer simulated from company records.
INTERVIEW SO FAR:
{recent(self._turns, 6)}
QUESTION: {state['candidate_question']}
SIMULATED ANSWER: {sim}

novelty (0-1): would the real answer likely contain knowledge NOT already in the records or the interview so far?
specificity (0-1): would it be concrete (conditions, numbers, exceptions)?
Return JSON {{"novelty": 0.0, "specificity": 0.0, "rationale": "..."}}.""", seed=salt)
        card = self.by_id.get(state.get("gap_card_id") or "")
        pr = card.priority if card else 0.6
        nov, spec = float(out.get("novelty", 0)), float(out.get("specificity", 0))
        score = self.w["novelty"] * nov + self.w["specificity"] * spec + self.w["priority"] * min(1.0, pr)
        state |= {"simulated_answer": sim, "rationale": out.get("rationale", ""), "novelty": nov, "specificity": spec}
        return state, max(0.0, min(1.0, score))

    def actions(self) -> dict:
        fns = {}
        if any(c.id not in self.explored for c in self.cards):
            fns["widen"] = self._widen
        if self._turns:
            fns["deepen"] = self._deepen
        if self.checker and self.checker.queue:
            fns["contradict"] = self._contradict
        return fns

    def next_question(self, turns, n):
        self._turns = turns
        fns = self.actions()
        if not fns:
            return Turn(idx=0, strategy=self.name, question="Is there anything else a successor must know?",
                        question_kind="generic")
        random.seed(self.seed + len(turns))
        try:
            import numpy as np
            np.random.seed(self.seed + len(turns))
        except ImportError:
            pass
        tok0 = self.llm.tokens("D.")
        top, nodes = treequest_adapter.search(fns, self.steps)
        state, score = top[0]
        cost = self.llm.tokens("D.") - tok0
        kind = state["kind"]
        meta = {"score": score, "tree_nodes": nodes, "actions": sorted(fns), "rationale": state.get("rationale")}
        if kind == "widen":
            self.current = self.by_id[state["gap_card_id"]]
            self.explored.add(self.current.id)
        elif kind == "contradiction":
            item = self.checker.pop(state["contradiction_id"])
            meta |= {"contradiction_id": state["contradiction_id"], "doc_ids": item.doc_ids if item else []}
        self.log.append({"turn": len(turns) + 1, "chosen": kind, "score": score, "nodes": nodes,
                         "search_tokens": cost, "actions": sorted(fns)})
        return Turn(idx=0, strategy=self.name, question=state["candidate_question"], question_kind=kind,
                    gap_card_id=state.get("gap_card_id"), search_cost_tokens=cost, meta=meta)
