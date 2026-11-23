"""Owner step 7: verify extracted rules from EVIDENCE ONLY (no knowledge of the project needed)."""
import random
from pathlib import Path

from kcap.components.corpus import PublicCorpus
from kcap.human.term import ask_choice
from kcap.io import read_json, read_jsonl, write_json
from kcap.schemas import HiddenRule


def _save_rules(rdir: Path, cands: list, idxs: list[int]):
    rules = [HiddenRule(id=f"O{k + 1:02d}", trigger=cands[i].get("trigger", ""), action=cands[i].get("action", ""),
                        reason=cands[i].get("reason") or "", scope=cands[i].get("scope") or "",
                        importance=int(cands[i].get("importance", 2) or 2)) for k, i in enumerate(idxs)]
    write_json(rdir / "private" / "rulebook.json", rules)
    return rules


def auto_verify(rdir: Path, min_support: int = 1):
    """No-human fallback: keep rules the judge calls general with >= min_support quoted comments. Weaker evidence."""
    cands = read_json(rdir / "private" / "candidate_rules.json")
    idxs = [i for i, c in enumerate(cands) if c.get("llm_general") and len(c.get("support_ids", [])) >= min_support]
    write_json(rdir / "private" / "verified_rules.json", {"verified": idxs, "rejected": [], "seen": idxs, "auto": True})
    rules = _save_rules(rdir, cands, idxs)
    print(f"{len(rules)} rules auto-verified (LLM only). Report them as 'auto-verified', not hand-verified.")
    return rules


def verify(cfg: dict, rdir: Path, sample: int, seed: int = 0):
    cands = read_json(rdir / "private" / "candidate_rules.json")
    reviews = {r["id"]: r for r in read_jsonl(rdir / "private" / "reviews.jsonl")}
    corpus = PublicCorpus.load(rdir / "public" / "corpus.jsonl", cfg)
    out_p = rdir / "private" / "verified_rules.json"
    prev = read_json(out_p) if out_p.exists() else {}
    done = prev if prev and not prev.get("auto") else {"verified": [], "rejected": [], "seen": []}
    order = [i for i in range(len(cands)) if i not in done["seen"]]
    random.Random(seed).shuffle(order)
    # strongest evidence first: judged general, more distinct reviewers, more supporting comments
    order.sort(key=lambda i: (not cands[i].get("llm_general"), -len(cands[i].get("support_authors", [])),
                              -len(cands[i].get("support_ids", []))))
    print("You do NOT need to know the project: read the quotes and answer two reading questions.\n"
          "Type q at any prompt to stop (progress is saved).\n")
    for i in order[:max(0, sample - len(done["seen"]))]:
        c = cands[i]
        print("=" * 80)
        print(f"CLAIMED RULE: when {c.get('trigger')}\n  -> {c.get('action')}\n  why: {c.get('reason') or '-'}")
        print(f"  AI pre-check: {'general rule' if c.get('llm_general') else 'maybe one-off'} - {c.get('llm_why', '')}")
        for cid in (c.get("support_ids") or c.get("comment_ids", []))[:3]:
            if cid in reviews:
                rv = reviews[cid]
                url = (rv.get("pr_url") or "").replace("api.github.com/repos", "github.com").replace("/pulls/", "/pull/")
                print(f"\n  MAINTAINER COMMENT by {rv.get('author')} ({url}):\n  {rv['body'][:700]}")
        print("\n  CLOSEST DOCUMENTATION EXCERPTS:")
        for d in [corpus.by_id[x] for x in c.get("doc_hits", []) if x in corpus.by_id][:3]:
            print(f"   [{d.id}] {d.text[:350]!r}")
        a1 = ask_choice("\nQ1. Do the comments state this as a general practice (not just a fix for this one PR)?",
                        ["y", "n", "q"])
        if a1 == "q":
            break
        a2 = "n"
        if a1 == "y":
            a2 = ask_choice("Q2. Is the rule MISSING from the documentation excerpts above?", ["y", "n", "q"])
            if a2 == "q":
                break
        done["seen"].append(i)
        (done["verified"] if a1 == a2 == "y" else done["rejected"]).append(i)
        write_json(out_p, done)
    rules = _save_rules(rdir, cands, done["verified"])
    print(f"\n{len(rules)} verified rules saved ({len(done['seen'])} reviewed).")
    return rules
