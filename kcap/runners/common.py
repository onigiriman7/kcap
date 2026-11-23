"""Shared runner helpers: build interviewers, run + synthesize, token accounting."""
from pathlib import Path

from kcap.components.contradiction import ContradictionChecker
from kcap.components.corpus import PublicCorpus
from kcap.components.interviewer.strategy_a_generic import GenericInterviewer
from kcap.components.interviewer.strategy_b_staged import StagedInterviewer
from kcap.components.interviewer.strategy_c_hypothesis import HypothesisInterviewer
from kcap.components.interviewer.strategy_d_adaptive import AdaptiveInterviewer
from kcap.schemas import GapCard

STRATEGIES = ("A", "B", "C", "D", "C+K", "C+S")
INTERVIEW_PREFIXES = ("A.", "B.", "C.", "contradiction.")


def build_interviewer(strategy: str, llm, corpus: PublicCorpus, cards: list[GapCard], cfg: dict, seed: int,
                      d_log: list | None = None):
    if strategy == "A":
        return GenericInterviewer(llm, seed)
    if strategy == "B":
        return StagedInterviewer(llm, corpus, seed)
    fu = cfg.get("strategy_c", {}).get("followups_per_card", 2)
    if strategy == "C":
        return HypothesisInterviewer(llm, cards, fu, None, seed)
    checker = None
    if strategy in ("C+K", "D"):
        checker = ContradictionChecker(llm, corpus, seed)
        checker.pre_interview()
    if strategy == "C+K":
        return HypothesisInterviewer(llm, cards, fu, checker, seed)
    if strategy in ("D", "C+S"):
        return AdaptiveInterviewer(llm, corpus, cards, cfg, checker, seed, d_log)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")


def token_split(before: dict, after: dict) -> dict:
    diff = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    return {"interview_tokens": sum(v for k, v in diff.items() if k.startswith(INTERVIEW_PREFIXES)),
            "search_tokens": sum(v for k, v in diff.items() if k.startswith("D.")),
            "synthesis_tokens": sum(v for k, v in diff.items() if k.startswith("synth."))}


def safe_name(s: str) -> str:
    return s.replace("+", "plus")


def strat_dir(world_dir: Path, s: str) -> Path:
    return Path(world_dir) / safe_name(s)
