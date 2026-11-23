import time
from pathlib import Path
from typing import Callable

from kcap.io import append_jsonl
from kcap.schemas import Turn

AnswerFn = Callable[[str], str]


class Interviewer:
    name = "?"

    def next_question(self, turns: list[Turn], n: int) -> Turn:
        raise NotImplementedError

    def observe(self, turns: list[Turn]) -> None:
        """Called after the latest turn has been answered."""


def recent(turns: list[Turn], k: int = 6) -> str:
    return "\n".join(f"Q{t.idx}: {t.question}\nA{t.idx}: {t.answer}" for t in turns[-k:]) or "(none)"


def run_interview(interviewer: Interviewer, answer: AnswerFn, n: int, transcript_path: Path | None = None,
                  time_limit_s: float | None = None) -> list[Turn]:
    """Ask exactly n questions (rule 3), unless a human session's time limit runs out.
    The time limit counts only the expert's answering time, not the interviewer's search time."""
    turns: list[Turn] = []
    spent = 0.0
    for i in range(n):
        if time_limit_s and spent >= time_limit_s:
            break
        t = interviewer.next_question(turns, n)
        t.idx = i + 1
        if time_limit_s:
            print(f"[{(time_limit_s - spent) / 60:.0f} min of your time left]")
        t0 = time.monotonic()
        t.answer = answer(t.question)
        t.answer_seconds = round(time.monotonic() - t0, 2)
        spent += t.answer_seconds
        turns.append(t)
        if transcript_path:
            append_jsonl(transcript_path, t)
        interviewer.observe(turns)
    return turns
