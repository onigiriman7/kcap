"""Strategy A: generic exit interview. Does not read the corpus."""
from kcap.components.interviewer.base import Interviewer
from kcap.schemas import Turn

QUESTION_BANK = [
    "What should your successor know that isn't written down anywhere?",
    "What goes wrong most often in your work, and how do you handle it?",
    "Walk me through a typical week. What takes the most judgment?",
    "Which customers or partners need special handling?",
    "What mistakes do new people usually make?",
    "Which official procedures don't quite match how things are really done?",
    "What are the early warning signs that a job is going to go badly?",
    "Which suppliers or vendors do you trust most, and which least?",
    "What do you check that others usually forget?",
    "What shortcuts are safe, and which ones are not?",
    "What decisions do you make that your manager probably doesn't know about?",
    "What is the most expensive mistake you've seen, and how do you prevent it now?",
    "Which equipment or systems have quirks a newcomer should know?",
    "Are there seasonal or timing patterns that change how you work?",
    "Who do you call when you're stuck, and for what?",
    "What would you do differently from the written process, and why?",
    "What do customers complain about, and how do you head it off?",
    "Which jobs do you always schedule or price differently?",
    "What rules of thumb do you use for estimates?",
    "What safety issues do you watch for beyond the rulebook?",
    "What tools or parts do you always keep on hand, and why?",
    "Which paperwork matters most, and which can slide?",
    "What has changed over the years that newcomers wouldn't know?",
    "What exceptions do you make for particular sites or locations?",
    "How do you decide when to escalate a problem?",
    "What do you wish someone had told you in your first year?",
    "Which recurring problems have a known fix that isn't documented?",
    "What do you do when the official procedure doesn't fit the situation?",
    "What should your successor never do?",
    "Is there anything else important we haven't covered?",
]


class GenericInterviewer(Interviewer):
    name = "A"

    def __init__(self, llm, seed: int = 0):
        self.llm, self.seed, self.q = llm, seed, 0

    def next_question(self, turns, n):
        if turns and turns[-1].question_kind == "generic" and turns[-1].meta.get("bank"):
            last = turns[-1]
            text = self.llm.chat("interviewer", "A.followup", f"""You are conducting a standard exit interview.
Previous question: {last.question}
Answer: {last.answer}
Ask ONE short generic follow-up in the spirit of "Can you give an example?". Output only the question.""",
                                 seed=self.seed + len(turns)).strip()
            return Turn(idx=0, strategy=self.name, question=text, question_kind="generic")
        q = QUESTION_BANK[self.q % len(QUESTION_BANK)]
        self.q += 1
        return Turn(idx=0, strategy=self.name, question=q, question_kind="generic", meta={"bank": True})
