# Adapted from SakanaAI/AI-Scientist-v2 ai_scientist/perform_ideation_temp_free.py@96bd516
# (generate_temp_free_idea: workshop file -> generate -> tool use -> reflect -> FinalizeIdea).
# Kept: ACTION/ARGUMENTS protocol, archive of previous ideas, reflection prompt, regex parsing.
# Changed: research topic -> corpus cluster "workshop file"; SearchSemanticScholar -> CorpusSearch
#          (local hybrid retrieval, NO network); FinalizeIdea -> FinalizeGapCard (our GapCard fields);
#          persona: AI researcher -> operations analyst. No generated code is ever executed.
import json
import re
import traceback

from kcap.components.corpus import PublicCorpus


class CorpusSearch:
    name = "CorpusSearch"
    description = ("Search the organization's documents (records, specs, plans, notes, procedures). Use it to find "
                   "supporting evidence for a hypothesis AND to check whether the practice or reasoning is already "
                   "written down. Returns the top matching documents with IDs.")

    def __init__(self, corpus: PublicCorpus, k: int = 6):
        self.corpus, self.k = corpus, k

    def use_tool(self, query: str) -> str:
        docs = self.corpus.search(query, self.k)
        return self.corpus.render(docs, 600) if docs else "No documents found."


FINALIZE = {
    "name": "FinalizeGapCard",
    "description": """Finalize your hypothesis by providing the gap card details.
The GAP CARD JSON should include:
- "hypothesis": a concrete statement "When <situation>, the expert does <action> (likely because <reason>), except <limits if seen>".
- "evidence_doc_ids": list of document IDs that show the behavior (must be real IDs you saw).
- "importance": 0-1, how costly it would be if a successor did not know this.
- "undocumentedness": 0-1, 1 = no document states this rule or its reason; lower if a document states it.
- "opening_question": a hypothesis-correction question to the expert citing the evidence, e.g.
  "In WO-1182 and WO-1240 you did X when Y, unlike SOP-3. Is that right, and why?" """,
}

SYSTEM_PROMPT = f"""You are an experienced analyst looking for decisions that no document explains.
You read an organization's documents (records, specs, plans, notes; possibly not in English) and propose
hypotheses about undocumented expert knowledge: repeated choices, priorities, exclusions or judgments that
deviate from procedures or from what a newcomer would expect, and whose reasons are not written down.
Each hypothesis should stem from a concrete pattern in specific documents. Prefer patterns that recur across
several documents and that a successor would get wrong. Do not propose things a document already states.
Write in English.

You have access to the following tools:

- **{CorpusSearch.name}**: {CorpusSearch.description}

- **{FINALIZE['name']}**: {FINALIZE['description']}

Respond in the following format:

ACTION:
<The action to take, exactly one of "{CorpusSearch.name}", "{FINALIZE['name']}">

ARGUMENTS:
<If ACTION is "{CorpusSearch.name}", provide {{"query": "your search query"}}. If ACTION is "{FINALIZE['name']}",
provide {{"gap_card": {{"hypothesis": "...", "evidence_doc_ids": ["..."], "importance": 0.0,
"undocumentedness": 0.0, "opening_question": "..."}}}}>

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one corpus search before finalizing, to check the rule is not already documented."""

GENERATION_PROMPT = """{workshop_description}

Here are the hypotheses that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating a new hypothesis about an undocumented expert rule that differs from what you have previously proposed.
"""

REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider whether the evidence really supports the hypothesis, whether it
is already documented, and whether the opening question is concrete and cites evidence.
Drop details the evidence does not support. Stick to the spirit of the original hypothesis unless there are
glaring issues. When ready, use FinalizeGapCard.

Results from your last action (if any):

{last_tool_results}
"""


def _parse(response_text: str):
    action_match = re.search(r"ACTION:\s*(.*?)\s*ARGUMENTS:", response_text, re.DOTALL | re.IGNORECASE)
    arguments_match = re.search(r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)", response_text, re.DOTALL | re.IGNORECASE)
    if not all([action_match, arguments_match]):
        raise ValueError("Failed to parse the LLM response.")
    action = action_match.group(1).strip().strip('"')
    arguments_text = arguments_match.group(1).strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", arguments_text, re.DOTALL)
    if m:
        arguments_text = m.group(1)
    s, e = arguments_text.find("{"), arguments_text.rfind("}")
    return action, json.loads(arguments_text[s:e + 1])


def generate_temp_free_gap_cards(llm, corpus: PublicCorpus, workshop_description: str, max_num_generations: int = 4,
                                 num_reflections: int = 3, seed: int = 0, log: list | None = None) -> list[dict]:
    tool = CorpusSearch(corpus)
    archive: list[str] = []
    for gen_idx in range(max_num_generations):
        try:
            prev_ideas_string = "\n\n".join(archive)
            last_tool_results, searched, msg_history = "", False, []
            for reflection_round in range(num_reflections + 1):
                if reflection_round == 0:
                    prompt_text = GENERATION_PROMPT.format(workshop_description=workshop_description,
                                                           prev_ideas_string=prev_ideas_string or "(none)")
                else:
                    prompt_text = REFLECTION_PROMPT.format(current_round=reflection_round + 1,
                                                           num_reflections=num_reflections + 1,
                                                           last_tool_results=last_tool_results or "No new results.")
                if reflection_round == num_reflections:
                    prompt_text += "\nThis is the final round: you MUST use FinalizeGapCard now."
                msg_history.append({"role": "user", "content": prompt_text})
                response_text = llm.chat("interviewer", "gap_miner.ideation", msg_history, system=SYSTEM_PROMPT,
                                         seed=seed * 1000 + gen_idx * 10 + reflection_round)
                msg_history.append({"role": "assistant", "content": response_text})
                try:
                    action, args = _parse(response_text)
                except (ValueError, json.JSONDecodeError):
                    last_tool_results = "Could not parse your response. Follow the ACTION/ARGUMENTS format exactly."
                    continue
                if action == tool.name:
                    last_tool_results = tool.use_tool(**args)
                    searched = True
                elif action == FINALIZE["name"]:
                    if not searched and reflection_round < num_reflections:
                        last_tool_results = "You must run at least one CorpusSearch before finalizing."
                        continue
                    card = args.get("gap_card") or args
                    card["evidence_doc_ids"] = [d for d in card.get("evidence_doc_ids", []) if d in corpus.by_id]
                    if card.get("hypothesis") and card["evidence_doc_ids"]:
                        archive.append(json.dumps(card))
                        if log is not None:
                            log.append({"gen": gen_idx, "rounds": reflection_round + 1, "card": card})
                    break
                else:
                    last_tool_results = f"Invalid action {action!r}."
        except Exception:
            traceback.print_exc()
            continue
    return [json.loads(s) for s in archive]
