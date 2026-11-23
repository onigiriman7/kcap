"""World = one (domain, seed): public corpus + private ground truth. Runner/eval only; never components."""
import re
from dataclasses import dataclass
from pathlib import Path

from kcap.io import read_json, read_jsonl, write_json, write_jsonl
from kcap.schemas import Document, Domain, HiddenRule, PlantedContradiction, Scenario
from kcap.synth.corpus_gen import generate_corpus
from kcap.synth.domain_gen import generate_domain
from kcap.synth.leak_check import leak_check
from kcap.synth.rulebook_gen import generate_rulebook
from kcap.synth.scenario_gen import generate_scenarios


@dataclass
class World:
    dir: Path
    domain: Domain
    rules: list[HiddenRule]
    contradictions: list[PlantedContradiction]
    scenarios: list[Scenario]

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def public_corpus_path(self) -> Path:
        return self.dir / "public" / "corpus.jsonl"

    @property
    def public_description(self) -> str:
        return read_json(self.dir / "public" / "domain.json")["description"]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def save_world(out: Path, domain: Domain, rules, docs, contradictions, scenarios, leak_report=None):
    write_json(out / "public" / "domain.json", {"name": domain.name, "description": domain.description})
    write_jsonl(out / "public" / "corpus.jsonl", [d.model_dump(exclude={"related_rule_ids"}) for d in docs])
    write_json(out / "private" / "domain.json", domain)
    write_json(out / "private" / "rulebook.json", rules)
    write_jsonl(out / "private" / "full_corpus.jsonl", docs)
    write_json(out / "private" / "contradictions.json", contradictions)
    write_json(out / "private" / "scenarios.json", scenarios)
    if leak_report is not None:
        write_json(out / "private" / "leak_report.json", leak_report)


def load_world(d: Path) -> World:
    d = Path(d)
    return World(
        dir=d,
        domain=Domain(**read_json(d / "private" / "domain.json")),
        rules=[HiddenRule(**r) for r in read_json(d / "private" / "rulebook.json")],
        contradictions=[PlantedContradiction(**c) for c in read_json(d / "private" / "contradictions.json")],
        scenarios=[Scenario(**s) for s in read_json(d / "private" / "scenarios.json")],
    )


def generate_world(llm, cfg: dict, domain_name: str, seed: int, out: Path, log=print) -> World:
    c = cfg["exp1"]
    log(f"[{out.name}] domain")
    domain = generate_domain(llm, domain_name, seed)
    log(f"[{out.name}] rulebook ({c['n_rules']} rules)")
    rules = generate_rulebook(llm, domain, c["n_rules"], c["min_scope_exception_rules"], seed)
    log(f"[{out.name}] corpus (~{c['n_docs']} docs)")
    docs, contras = generate_corpus(llm, domain, rules, c["n_docs"], c["noise_frac"], c["n_contradictions"], seed)
    log(f"[{out.name}] leak check")
    docs, report = leak_check(llm, domain, docs, rules, c["leak_max_attempts"], seed)
    kept = {d.id for d in docs}
    contras = [k.model_copy(update={"doc_ids": [i for i in k.doc_ids if i in kept]}) for k in contras]
    log(f"[{out.name}] leak rate {report['initial_leak_rate']:.1%} initial, dropped {len(report['dropped'])}")
    log(f"[{out.name}] scenarios")
    scen = generate_scenarios(llm, domain, rules, c["n_scenarios"], c["min_scope_scenarios"], seed)
    save_world(out, domain, rules, docs, contras, scen, report)
    return load_world(out)


def import_pilot(pilot_dir: Path, out: Path, llm=None, cfg=None) -> World:
    """Import the 2026-11 HVAC pilot (kcap_pilot/) as a world. Scenarios are generated if llm is given."""
    p = Path(pilot_dir)
    rules = [HiddenRule(**r) for r in read_json(p / "private" / "rulebook.json")]
    doc_map = read_json(p / "private" / "doc_rule_map.json") if (p / "private" / "doc_rule_map.json").exists() else {}
    docs = []
    for r in read_jsonl(p / "public" / "corpus.jsonl"):
        rids = doc_map.get(r["id"], []) if isinstance(doc_map, dict) else []
        docs.append(Document(id=r["id"], doc_type=r["doc_type"], date=r["date"], text=r["text"],
                             related_rule_ids=rids if isinstance(rids, list) else [rids]))
    contras = [PlantedContradiction(id=k["id"], rule_id=k["rule_id"], doc_ids=[k["sop_doc"]] + k["deviating_docs"])
               for k in read_json(p / "private" / "contradictions.json")]
    domain = Domain(name="regional HVAC field service", description="Regional HVAC field-service company (pilot world).")
    scen = []
    if llm is not None:
        scen = generate_scenarios(llm, domain, rules, cfg["exp1"]["n_scenarios"], cfg["exp1"]["min_scope_scenarios"], 0)
    save_world(out, domain, rules, docs, contras, scen)
    return load_world(out)
