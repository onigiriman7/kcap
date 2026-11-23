"""Exp 2 secondary: a persona playing a repo maintainer, built from verified review-comment rules."""
from kcap.persona.synthetic_persona import SyntheticPersona
from kcap.schemas import Domain, HiddenRule


class OSSPersona(SyntheticPersona):
    def __init__(self, llm, cfg: dict, repo: str, rules: list[HiddenRule], doc_snippets: list[str], seed=0,
                 log_path=None):
        domain = Domain(name=f"the open-source project {repo}",
                        description=f"You are a long-time maintainer of {repo} who reviews most pull requests.",
                        roles=["core maintainer"])
        super().__init__(llm, cfg, domain, rules, doc_snippets, seed, log_path, task_prefix="oss_persona")
