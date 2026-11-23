"""Owner steps 9: self exit interview (Half A) and tool interview (Half D), time-limited, with burden rating."""
import time
from pathlib import Path

from kcap.components.contradiction import ContradictionChecker
from kcap.components.corpus import PublicCorpus
from kcap.components.interviewer.base import run_interview
from kcap.components.interviewer.strategy_d_adaptive import AdaptiveInterviewer
from kcap.human.term import ask_choice, read_multiline, record_and_transcribe
from kcap.io import append_jsonl, write_json
from kcap.schemas import GapCard, Turn


def _burden(sess_dir: Path, mode: str):
    r = ask_choice("How burdensome was this session? 1 = effortless ... 5 = exhausting", list("12345"))
    note = input("Any note about the session (optional): ").strip()
    write_json(sess_dir / f"burden_{mode}.json", {"rating": int(r), "note": note})


def self_exit(sess_dir: Path, topics: dict[str, str], minutes: float):
    path = sess_dir / "self_exit.jsonl"
    path.unlink(missing_ok=True)
    labels = list(topics.values())
    print(f"SELF EXIT INTERVIEW - {minutes:.0f} minutes. Write what your successor should know about these topics:")
    for i, t in enumerate(labels, 1):
        print(f"  {i}. {t}")
    print("Write as many notes as you like. Type /done to finish early.\n")
    end = time.monotonic() + minutes * 60
    idx = 0
    while time.monotonic() < end:
        left = (end - time.monotonic()) / 60
        t = input(f"[{left:.0f} min left] Topic number (Enter = general): ").strip()
        topic = labels[int(t) - 1] if t.isdigit() and 0 < int(t) <= len(labels) else "general"
        t0 = time.monotonic()
        text = read_multiline("Note:")
        if text == "/done":
            break
        if text:
            idx += 1
            append_jsonl(path, Turn(idx=idx, strategy="self_exit", question=f"What should your successor know about: {topic}?",
                                    answer=text, question_kind="generic", answer_seconds=round(time.monotonic() - t0, 1)))
    _burden(sess_dir, "self_exit")


def tool(llm, cfg: dict, sess_dir: Path, corpus: PublicCorpus, cards: list[GapCard], minutes: float, voice: bool):
    path = sess_dir / "tool.jsonl"
    path.unlink(missing_ok=True)
    print(f"TOOL INTERVIEW - {minutes:.0f} minutes of YOUR answering time (the tool's thinking time between "
          "questions is not counted). Answer each question; an empty line ends an answer. "
          "\"I don't know\" is a fine answer.\nPreparing (checking the documents for contradictions)...\n")
    checker = ContradictionChecker(llm, corpus)
    checker.pre_interview()
    log: list = []
    iv = AdaptiveInterviewer(llm, corpus, cards, cfg, checker, 0, log)

    def answer(q: str) -> str:
        print(f"\nQ: {q}")
        return record_and_transcribe() if voice else read_multiline("A:")

    run_interview(iv, answer, 10_000, path, time_limit_s=minutes * 60)
    write_json(sess_dir / "tool_search_log.json", log)
    _burden(sess_dir, "tool")
