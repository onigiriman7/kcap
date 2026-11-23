from typing import Literal

from pydantic import BaseModel, Field


class Domain(BaseModel):
    name: str
    description: str
    roles: list[str] = []
    entities: list[str] = []


class HiddenRule(BaseModel):
    id: str
    trigger: str
    action: str
    reason: str
    scope: str
    importance: int = 2
    visible_effects: list[str] = []


class Document(BaseModel):
    id: str
    doc_type: str
    date: str
    text: str
    related_rule_ids: list[str] = []


class PlantedContradiction(BaseModel):
    id: str
    rule_id: str
    doc_ids: list[str]  # first id is the SOP stating the default
    description: str = ""


class GapCard(BaseModel):
    id: str
    hypothesis: str
    evidence_doc_ids: list[str]
    importance: float
    undocumentedness: float
    opening_question: str
    cluster: str = ""

    @property
    def priority(self) -> float:
        return self.importance * self.undocumentedness


QuestionKind = Literal["generic", "staged", "hypothesis", "deepen", "widen", "contradiction"]


class Turn(BaseModel):
    idx: int
    strategy: str
    question: str
    answer: str = ""
    question_kind: QuestionKind
    gap_card_id: str | None = None
    search_cost_tokens: int = 0
    answer_seconds: float | None = None
    meta: dict = Field(default_factory=dict)


class KnowledgeUnit(BaseModel):
    id: str
    condition: str
    action: str
    reason: str | None = None
    scope: str | None = None
    source_turns: list[int] = []
    source_docs: list[str] = []
    confidence: Literal["stated", "confirmed_by_records", "conflicts_with_records"] = "stated"
    examples: list[str] = []

    def text(self) -> str:
        ex = f" EXAMPLES: {' | '.join(self.examples)}" if self.examples else ""
        return f"WHEN {self.condition} THEN {self.action}. WHY: {self.reason or '-'}. SCOPE: {self.scope or '-'}.{ex}"


class Scenario(BaseModel):
    id: str
    prompt: str
    reference_answer: str
    rule_ids: list[str] = []
    topic: str = ""
    arm: str = ""


class ContradictionItem(BaseModel):
    """A queued follow-up produced by the contradiction checker (public, no ground truth)."""
    id: str
    claim: str
    doc_ids: list[str]
    question: str
    source: Literal["pre_interview", "answer"] = "answer"
