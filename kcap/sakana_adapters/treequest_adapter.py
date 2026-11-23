"""Thin wrapper around SakanaAI/treequest (AB-MCTS). The only module that imports treequest."""
from typing import Any, Callable

import treequest as tq

GenerateFn = Callable[[Any | None], tuple[Any, float]]


def search(generate_fns: dict[str, GenerateFn], steps: int, algo: str = "ABMCTSA", k: int = 1):
    """Run `steps` AB-MCTS steps; return (top-k [(state, score)], number of nodes expanded)."""
    algorithm = getattr(tq, algo)()
    tree = algorithm.init_tree()
    for _ in range(steps):
        tree = algorithm.step(tree, generate_fns)
    return tq.top_k(tree, algorithm, k=k), steps
