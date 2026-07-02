"""URI-based edge inference.

Edges are derived by matching each consumer input URI against the output URIs of other tasks in the
same run scope. Two producers matching the *identical* URI are a copy-then-overwrite (last writer
wins, ordered by ``after``); producers matching *different* URIs via a glob are a many-to-one fan-in
(depend on all). Tasks operate on canonical URIs (voxel spacing always attached), so vs is enforced
by the match.
"""

from __future__ import annotations

from typing import List, Tuple

from copick_dag.plan.concrete import ConcreteTask
from copick_dag.uris import is_concrete_uri, patterns_overlap, same_uri

# (producer_task, its concrete output URI)
ProducerIndex = List[Tuple[ConcreteTask, str]]


def build_producer_index(tasks: List[ConcreteTask]) -> ProducerIndex:
    index: ProducerIndex = []
    for t in tasks:
        for out_uri in t.output_uris():
            index.append((t, out_uri))
    return index


def find_producers(
    in_uri: str,
    index: ProducerIndex,
    exclude: ConcreteTask,
) -> Tuple[List[ConcreteTask], List[ConcreteTask]]:
    """Return ``(exact_writers, glob_writers)`` matching ``in_uri``.

    ``exact_writers`` write the identical concrete URI (overwrite chain); ``glob_writers`` write
    different URIs that the (possibly wildcard) ``in_uri`` selects (fan-in).
    """
    exact: List[ConcreteTask] = []
    glob: List[ConcreteTask] = []
    seen_exact = set()
    seen_glob = set()
    for task, out_uri in index:
        if task is exclude:
            continue
        if not patterns_overlap(out_uri, in_uri):
            continue
        # Overwrite (exact) only when BOTH sides are concrete and identical; any wildcard on either
        # side is a fan-in edge (a wildcard producer/consumer must never be treated as an overwrite).
        if is_concrete_uri(out_uri) and is_concrete_uri(in_uri) and same_uri(out_uri, in_uri):
            if task.task_id not in seen_exact:
                exact.append(task)
                seen_exact.add(task.task_id)
        elif task.task_id not in seen_glob:
            glob.append(task)
            seen_glob.add(task.task_id)
    return exact, glob
