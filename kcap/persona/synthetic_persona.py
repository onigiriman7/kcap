"""Evaluation persona: knows the hidden rules, discloses facets progressively, only when targeted.

The answer model only ever sees the facets it is allowed to disclose, so gating is enforced structurally.
"""
import random
from pathlib import Path

from kcap.io import append_jsonl
from kcap.retrieval.index import jaccard
from kcap.schemas import Domain, HiddenRule

FACETS = ("action", "reason", "scope")


class SyntheticPersona:
    def __init__(self, llm, cfg: dict, domain: Domain, rules: list[HiddenRule], sop_texts: list[str] | None = None,
                 seed: int = 0, log_path: Path | None = None, task_prefix: str = "persona"):
        self.llm, self.domain, self.rules = llm, domain, {r.id: r for r in rules}
        pc = cfg.get("persona", {})
        self.misremember_p = pc.get("misremember_prob", 0.1)
        self.volunteer_p = pc.get("volunteer_prob", 0.0)
        self.rng = random.Random(seed)
        self.seed = seed
        self.sop_texts = sop_texts or []
        self.disclosed: dict[str, set[str]] = {rid: set() for rid in self.rules}
        self.history: list[tuple[str, str]] = []
        self.log_path = log_path
        self.tp = task_prefix
        self.role = (domain.roles or ["senior specialist"])[0]

    # ---- gating ----
    def gate(self, question: str) -> list[dict]:
        rules_txt = "\n".join(f"{r.id}: situation={r.trigger} | shows up as: {'; '.join(r.visible_effects)}"
                              for r in self.rules.values())
        recent = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.history[-2:]) or "(start of interview)"
        prompt = f"""RULES (situations):
{rules_txt}

RECENT CONVERSATION:
{recent}

NEW QUESTION: {question}

Which rules does the new question SPECIFICALLY target? A question targets a rule only if it names that rule's
situation concretely or cites evidence of it (records, entities, patterns), or it is a direct follow-up
("why?", "any exceptions?") to an answer that discussed the rule. Generic questions ("what should your
successor know?", "what goes wrong most?") target NO rules.
For each targeted rule, classify the ask:
  "what" = asks what is done; "why" = asks for the reason; "exceptions" = asks about exceptions, edge cases,
  limits or whether it always applies; "challenge" = confronts with records that contradict the official way.
Return JSON {{"targets": [{{"rule_id": "R..", "ask": "what|why|exceptions|challenge"}}]}} (max 2 targets)."""
        out = self.llm.json("judge", f"{self.tp}.gate", prompt, seed=self.seed + len(self.history))
        return [t for t in out.get("targets", []) if t.get("rule_id") in self.rules][:2]

    def _release(self, targets: list[dict]) -> list[dict]:
        released = []
        for t in targets:
            d = self.disclosed[t["rule_id"]]
            ask = t.get("ask", "what")
            if ask == "challenge":
                new = {"action", "reason"}
            elif ask == "why":
                new = {"reason"} if "action" in d else {"action"}
            elif ask == "exceptions":
                new = {"scope"} if "action" in d else {"action"}
            else:
                new = {"action"}
            new -= d
            d |= new
            if new:
                released.append({"rule_id": t["rule_id"], "facets": sorted(new)})
        return released

    def _volunteer(self, question: str) -> list[dict]:
        if self.volunteer_p <= 0 or self.rng.random() >= self.volunteer_p:
            return []
        cands = [r for r in self.rules.values() if not self.disclosed[r.id]]
        if not cands:
            return []
        r = max(cands, key=lambda r: (jaccard(question, r.trigger), self.rng.random()))
        self.disclosed[r.id].add("action")
        return [{"rule_id": r.id, "facets": ["action"], "volunteered": True}]

    # ---- answering ----
    def answer(self, question: str) -> str:
        targets = self.gate(question)
        released = self._release(targets) if targets else self._volunteer(question)
        known = []
        for rid, facets in self.disclosed.items():
            if facets:
                r = self.rules[rid]
                known.append(f"- Situation: {r.trigger}\n" + "".join(f"  {f}: {getattr(r, f)}\n" for f in FACETS if f in facets))
        new_ids = {x["rule_id"] for x in released}
        misremember = bool(released) and self.rng.random() < self.misremember_p
        focus = ", ".join(sorted(new_ids)) or "none"
        sop = "\n".join(t[:400] for t in self.sop_texts[:5])
        system = f"""You are a {self.role} with many years of experience at: {self.domain.name}. {self.domain.description}
You are being interviewed before leaving. You are cooperative but busy, and you do not volunteer things you
are not asked about. Speak naturally and briefly (2-6 sentences), like a real person.

THE ONLY PRACTICES YOU MAY TALK ABOUT IN SPECIFIC TERMS (do not add details beyond these):
{chr(10).join(known) or "(none yet)"}

For anything else, answer vaguely with general platitudes or official-procedure-level statements, e.g. from:
{sop or "(general good practice)"}
Never invent other specific practices, numbers, names or exceptions."""
        prompt = f"Question: {question}\n\n"
        if released:
            prompt += ("In this answer, share the practice(s) relevant to the question in natural words. "
                       "Only reveal the facets listed above.")
        else:
            prompt += "Nothing specific applies; answer vaguely."
        if misremember:
            prompt += " Get ONE small detail (a number, name or date) slightly wrong, as people do when recalling."
        text = self.llm.chat("persona", f"{self.tp}.answer", prompt, system=system, seed=self.seed + len(self.history))
        self.history.append((question, text))
        if self.log_path:
            append_jsonl(self.log_path, {"turn": len(self.history), "question": question, "targets": targets,
                                         "disclosed": released, "misremember": misremember, "focus": focus})
        return text
