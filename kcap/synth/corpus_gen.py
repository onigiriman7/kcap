import random
from datetime import date, timedelta

from kcap.schemas import Document, Domain, HiddenRule, PlantedContradiction

DOC_TYPES = "work_order | ticket | sop | email | incident"
STYLE = ("Write realistic, terse records as they would appear in the company's systems: IDs, dates omitted, "
         "names, part numbers, amounts, abbreviations. Records describe WHAT happened, never WHY an unwritten "
         "practice exists.")


def _rule_docs(llm, domain: Domain, rule: HiddenRule, k: int, seed: int) -> list[dict]:
    prompt = f"""Domain: {domain.name}. {domain.description}
A hidden expert rule (SECRET - must never be stated in any document):
  trigger: {rule.trigger}
  action: {rule.action}
  reason: {rule.reason}
  scope/exception: {rule.scope}
  visible effects: {"; ".join(rule.visible_effects)}

Write {k} operational records ({DOC_TYPES}, not sop) in which the behavior is VISIBLE only through its effects.
Include at least one record where the scope exception applies and the behavior is therefore absent.
Make traces subtle: spread them across documents, mix with routine content, and include at least one record
where the effect is easy to miss. NEVER write the action as a rule, the reason, or the exception condition.
{STYLE}
Return JSON {{"docs": [{{"doc_type": "...", "text": "..."}}]}}."""
    return llm.json("generator", "synth.corpus.rule", prompt, seed=seed).get("docs", [])[:k]


def _noise_docs(llm, domain: Domain, k: int, seed: int) -> list[dict]:
    prompt = f"""Domain: {domain.name}. {domain.description}
Entities: {", ".join(domain.entities)}
Write {k} routine records ({DOC_TYPES}) unrelated to any special practice: scheduling, HR notes, routine jobs,
customer chatter, inventory, misleading-looking but explainable one-off events. Vary types and length.
{STYLE}
Return JSON {{"docs": [{{"doc_type": "...", "text": "..."}}]}}."""
    return llm.json("generator", "synth.corpus.noise", prompt, seed=seed).get("docs", [])[:k]


def _sop_docs(llm, domain: Domain, k: int, seed: int) -> list[dict]:
    prompt = f"""Domain: {domain.name}. {domain.description}
Write {k} standard operating procedures (doc_type "sop") covering routine processes. Formal SOP style with
SOP numbers and revision. Return JSON {{"docs": [{{"doc_type": "sop", "text": "..."}}]}}."""
    return llm.json("generator", "synth.corpus.sop", prompt, seed=seed).get("docs", [])[:k]


def _contradiction_sop(llm, domain: Domain, rule: HiddenRule, seed: int) -> dict:
    prompt = f"""Domain: {domain.name}.
The expert secretly follows this practice: when {rule.trigger}, {rule.action}
Write ONE formal SOP (doc_type "sop") that states the official DEFAULT practice for this situation, which the
expert's practice deviates from. The SOP must not mention the deviation, the expert, or any exception.
Return JSON {{"doc_type": "sop", "text": "..."}}."""
    return llm.json("generator", "synth.corpus.contradiction", prompt, seed=seed)


def generate_corpus(llm, domain: Domain, rules: list[HiddenRule], n_docs: int, noise_frac: float,
                    n_contradictions: int, seed: int, n_sops: int = 5):
    rng = random.Random(seed)
    raw: list[tuple[dict, list[str]]] = []
    n_noise = int(n_docs * noise_frac)
    per_rule = max(3, (n_docs - n_noise - n_sops - n_contradictions) // max(1, len(rules)))
    rule_doc_idx: dict[str, list[int]] = {}
    for i, r in enumerate(rules):
        for d in _rule_docs(llm, domain, r, per_rule, seed * 1000 + i):
            rule_doc_idx.setdefault(r.id, []).append(len(raw))
            raw.append((d, [r.id]))
    for b in range(0, n_noise, 10):
        for d in _noise_docs(llm, domain, min(10, n_noise - b), seed * 1000 + 500 + b):
            raw.append((d, []))
    for d in _sop_docs(llm, domain, n_sops, seed * 1000 + 900):
        raw.append((d | {"doc_type": "sop"}, []))
    contra_rules = sorted(rules, key=lambda r: (-r.importance, rng.random()))[:n_contradictions]
    contra_idx = {}
    for j, r in enumerate(contra_rules):
        d = _contradiction_sop(llm, domain, r, seed * 1000 + 950 + j)
        contra_idx[r.id] = len(raw)
        raw.append((d | {"doc_type": "sop"}, [r.id]))

    start = date(2022, 1, 1)
    dates = sorted(start + timedelta(days=rng.randrange(0, 3 * 365)) for _ in raw)
    order = list(range(len(raw)))
    rng.shuffle(order)
    # SOPs get the earliest dates so operational docs "deviate" afterwards
    order.sort(key=lambda i: 0 if raw[i][0].get("doc_type") == "sop" else 1)
    new_id = {}
    docs = []
    for pos, i in enumerate(order):
        d, rids = raw[i]
        did = f"D{pos + 1:03d}"
        new_id[i] = did
        docs.append(Document(id=did, doc_type=str(d.get("doc_type", "ticket")), date=dates[pos].isoformat(),
                             text=str(d.get("text", "")), related_rule_ids=rids))
    docs.sort(key=lambda d: d.id)
    contradictions = [
        PlantedContradiction(id=f"K{j + 1}", rule_id=r.id,
                             doc_ids=[new_id[contra_idx[r.id]]] + [new_id[i] for i in rule_doc_idx.get(r.id, [])],
                             description=f"SOP default vs. hidden rule {r.id}")
        for j, r in enumerate(contra_rules)]
    return docs, contradictions
