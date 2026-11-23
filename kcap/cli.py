"""`kcap` command line. Run `kcap --help` or `kcap exp1 --help`."""
from pathlib import Path

import typer

from kcap.config import load_config, load_prereg
from kcap.io import RUNS
from kcap.llm.client import LLM

app = typer.Typer(no_args_is_help=True, help="Expert knowledge-capture PoC")
exp1_app = typer.Typer(no_args_is_help=True, help="Experiment 1: synthetic expert")
exp2_app = typer.Typer(no_args_is_help=True, help="Experiment 2: open-source ground truth")
exp3_app = typer.Typer(no_args_is_help=True, help="Experiment 3: you as the expert")
judge_app = typer.Typer(no_args_is_help=True, help="Judge validation (hand labels)")
for name, sub in (("exp1", exp1_app), ("exp2", exp2_app), ("exp3", exp3_app), ("judge", judge_app)):
    app.add_typer(sub, name=name)

ConfigOpt = typer.Option(None, "--config", help="Alternative config YAML")


def _setup(config: Path | None):
    cfg = load_config(config)
    return cfg, LLM(cfg)


@app.command()
def check(config: Path | None = ConfigOpt):
    """Make one tiny call per role to verify API keys and caching."""
    cfg, llm = _setup(config)
    for role in cfg["roles"]:
        out = llm.chat(role, "check", "Reply with the single word OK.")
        typer.echo(f"{role:12s} {cfg['roles'][role]['model']:28s} -> {out.strip()[:40]}")
    typer.echo(f"spent ${llm.spent_usd:.4f}; calls are cached in {RUNS / '.cache'}")


@app.command()
def models(config: Path | None = ConfigOpt):
    """List models available through the GitHub Copilot SDK."""
    from kcap.llm.copilot_backend import get_backend
    cfg = load_config(config)
    for m in get_backend(cfg["providers"]["copilot"]).list_models():
        typer.echo(m)


# ---------------- Experiment 1 ----------------
@exp1_app.command("generate")
def exp1_generate(domains: int = 5, seeds: int = 5, config: Path | None = ConfigOpt):
    """Generate synthetic worlds (rulebook, corpus, leak check, scenarios)."""
    from kcap.runners import exp1
    cfg, llm = _setup(config)
    exp1.generate(llm, cfg, domains, seeds)
    typer.echo(f"spent ${llm.spent_usd:.2f}")


@exp1_app.command("import-pilot")
def exp1_import_pilot(pilot_dir: Path, with_scenarios: bool = True, config: Path | None = ConfigOpt):
    """Import the earlier HVAC pilot folder (kcap_pilot/) as a world."""
    from kcap.runners import exp1
    cfg, llm = _setup(config)
    w = exp1.import_pilot(pilot_dir, exp1.WORLDS / "pilot-hvac_s0", llm if with_scenarios else None, cfg)
    typer.echo(f"imported {w.id}: {len(w.rules)} rules, {len(w.scenarios)} scenarios")


@exp1_app.command("chat-persona")
def exp1_chat_persona(world: str = typer.Option(..., help="world id prefix"), config: Path | None = ConfigOpt):
    """Owner step 3: interview the persona yourself."""
    from kcap.runners import exp1
    cfg, llm = _setup(config)
    exp1.chat_persona(llm, cfg, world)


@exp1_app.command("run")
def exp1_run(strategies: str = "A,B,C,D", n: int = 40, run_id: str = "exp1", worlds: str | None = None,
             pilot: bool = typer.Option(False, help="1 world only; skips the prereg check"),
             no_eval: bool = False, config: Path | None = ConfigOpt):
    """Interview every world with each strategy, then evaluate and write the report."""
    from kcap.runners import exp1
    cfg, llm = _setup(config)
    d = exp1.run(llm, cfg, strategies.split(","), n, run_id, worlds, pilot, not no_eval)
    typer.echo(f"done: {d}  (spent ${llm.spent_usd:.2f})")


@exp1_app.command("evaluate")
def exp1_evaluate(run_id: str = "exp1", config: Path | None = ConfigOpt):
    """Re-run evaluation only (cached transcripts), e.g. after fixing the judge."""
    from kcap.runners import exp1
    cfg, llm = _setup(config)
    typer.echo(exp1.evaluate(llm, cfg, run_id))


@exp1_app.command("report")
def exp1_report(run_id: str = "exp1"):
    from kcap.runners import exp1
    typer.echo(exp1.report(run_id))


@exp1_app.command("budget-curve")
def exp1_budget_curve(world: str, ns: str = "10,20,40,60", strategies: str = "A,B,C,D",
                      config: Path | None = ConfigOpt):
    from kcap.runners import exp1
    cfg, llm = _setup(config)
    typer.echo(exp1.budget_curve(llm, cfg, world, [int(x) for x in ns.split(",")], strategies.split(",")))


# ---------------- Judge validation ----------------
@judge_app.command("export")
def judge_export(run_id: str = "exp1"):
    """Write labels.csv for you to hand-label (~100 rows)."""
    from kcap.eval import judge_validation
    typer.echo(judge_validation.export(RUNS / run_id))


@judge_app.command("score")
def judge_score(labels_csv: Path, run_id: str = "exp1"):
    from kcap.eval import judge_validation
    res = judge_validation.score(RUNS / run_id, labels_csv, load_prereg())
    typer.echo(res)
    typer.echo("GATE PASS" if res["gate_pass"] else "GATE FAIL: revise judge prompts / examples, then `kcap exp1 evaluate`")


# ---------------- Experiment 2 ----------------
@exp2_app.command("fetch")
def exp2_fetch(repo: str, config: Path | None = ConfigOpt):
    from kcap.oss.fetch import fetch
    cfg = load_config(config)
    typer.echo(fetch(repo, cfg["exp2"]["max_review_pages"]))


@exp2_app.command("extract")
def exp2_extract(repo: str, config: Path | None = ConfigOpt):
    from kcap.oss.fetch import repo_dir
    from kcap.oss.rule_extract import extract
    cfg, llm = _setup(config)
    extract(llm, cfg, repo_dir(repo))


@exp2_app.command("verify")
def exp2_verify(repo: str, auto: bool = typer.Option(False, help="LLM-only verification (weaker; no human)"),
                config: Path | None = ConfigOpt):
    """Owner step 7: verify ~30 extracted rules by reading the quoted evidence."""
    from kcap.oss.fetch import repo_dir
    from kcap.oss.verify_cli import auto_verify, verify
    cfg = load_config(config)
    if auto:
        auto_verify(repo_dir(repo))
    else:
        verify(cfg, repo_dir(repo), cfg["exp2"]["sample_to_verify"])


@exp2_app.command("run")
def exp2_run(repo: str, with_persona: bool = False, n: int = 40, config: Path | None = ConfigOpt):
    from kcap.runners import exp2
    cfg, llm = _setup(config)
    exp2.run(llm, cfg, repo, with_persona, n)


# ---------------- Experiment 3 ----------------
@exp3_app.command("ingest")
def exp3_ingest(folder: Path, description: str = typer.Option(..., help="2-3 sentences about your project")):
    """Load your own project records (text files) as the Exp 3 corpus."""
    from kcap.runners import exp3
    exp3.ingest(folder, description)


@exp3_app.command("seal")
def exp3_seal(n: int = 20):
    """Owner step 8: write and seal scenarios (do this BEFORE any session)."""
    from kcap.human.scenarios_cli import write_and_seal
    write_and_seal(n)


@exp3_app.command("gen-scenarios")
def exp3_gen_scenarios(n: int = 20, seed: int = 0, config: Path | None = ConfigOpt):
    """Generate hidden test scenarios from random docs and seal them without showing you."""
    from kcap.runners import exp3
    cfg, llm = _setup(config)
    exp3.gen_scenarios(llm, cfg, n, seed)


@exp3_app.command("gen-scenarios-balanced")
def exp3_gen_scenarios_balanced(per_arm: int = 7, seed: int = 1, config: Path | None = ConfigOpt):
    """Equal hidden scenarios per interview, from topic labels only. Seal unseen."""
    from kcap.runners import exp3
    cfg, llm = _setup(config)
    exp3.gen_scenarios_balanced(llm, cfg, per_arm, seed)


@exp3_app.command("gui")
def exp3_gui(port: int = 8765, config: Path | None = ConfigOpt):
    """Browser GUI: blinded interview sessions (and scenario sealing). localhost only."""
    from kcap.human.gui import serve
    cfg, llm = _setup(config)
    serve(port, llm=llm, cfg=cfg)


@exp3_app.command("split")
def exp3_split(config: Path | None = ConfigOpt):
    from kcap.runners import exp3
    cfg, llm = _setup(config)
    exp3.split(llm, cfg)


@exp3_app.command("resplit")
def exp3_resplit(seed: int = 0):
    """Re-assign existing gap cards to arms B/C/D (no new mining)."""
    from kcap.runners import exp3
    exp3.resplit(seed)


@exp3_app.command("interview")
def exp3_interview(arm: str = typer.Option(..., help="B | C | D (the GUI keeps this blinded)"),
                   minutes: float | None = None, config: Path | None = ConfigOpt):
    from kcap.runners import exp3
    cfg, llm = _setup(config)
    exp3.interview(llm, cfg, arm, minutes)


@exp3_app.command("synthesize")
def exp3_synth(config: Path | None = ConfigOpt):
    from kcap.runners import exp3
    cfg, llm = _setup(config)
    exp3.synth(llm, cfg)


@exp3_app.command("tag")
def exp3_tag():
    from kcap.runners import exp3
    exp3.tag()


@exp3_app.command("apprentice")
def exp3_apprentice(config: Path | None = ConfigOpt):
    from kcap.runners import exp3
    cfg, llm = _setup(config)
    exp3.run_apprentice(llm, cfg)


@exp3_app.command("grade")
def exp3_grade():
    from kcap.runners import exp3
    exp3.grade()


@exp3_app.command("report")
def exp3_report():
    from kcap.runners import exp3
    exp3.report()


if __name__ == "__main__":
    app()
