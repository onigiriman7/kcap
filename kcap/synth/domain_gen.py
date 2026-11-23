from kcap.schemas import Domain


def generate_domain(llm, name: str, seed: int) -> Domain:
    prompt = f"""Invent a realistic but fictional organization for this domain: "{name}".
Return JSON {{"name": "<domain name>", "description": "<150-word description of the organization, its work,
systems and typical records>", "roles": ["<expert job title>", "..."], "entities": ["<15-25 named entities:
suppliers, customers, equipment models, sites, systems, people>"]}}.
Use invented names. Variation seed: {seed}."""
    d = llm.json("generator", "synth.domain", prompt, seed=seed)
    d["name"] = d.get("name") or name
    return Domain(**d)
