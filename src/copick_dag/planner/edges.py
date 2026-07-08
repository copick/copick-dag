"""URI-based edge inference.

Edges are derived by matching each consumer input URI against the output URIs of other tasks in the
same run scope. Matching is **object-type aware**: an edge only forms between a producer output and a
consumer input of the *same* copick object type (a mesh ``membrane:u/s`` must never match a
segmentation ``membrane:u/s@10``). Two producers matching the *identical* URI are a
copy-then-overwrite (last writer wins, ordered by ``after``); producers matching *different* URIs via
a glob are a many-to-one fan-in (depend on all).
"""

from __future__ import annotations

from typing import List, Tuple

from copick_dag.plan.concrete import ConcreteTask
from copick_dag.registry import get_op
from copick_dag.uris import is_concrete_uri, patterns_overlap, same_uri

# (producer_task, its concrete output URI, that output slot's object_type)
ProducerIndex = List[Tuple[ConcreteTask, str, str]]


def build_producer_index(tasks: List[ConcreteTask]) -> ProducerIndex:
    index: ProducerIndex = []
    for t in tasks:
        op = get_op(t.op_id)
        for slot, out_uri in t.outputs.items():
            if out_uri is None:
                continue
            sspec = op.output_slot(slot)
            otype = sspec.object_type if sspec else "segmentation"
            index.append((t, out_uri, otype))
    return index


def find_producers(
    in_uri: str,
    index: ProducerIndex,
    exclude: ConcreteTask,
    in_object_type: str = "segmentation",
) -> Tuple[List[ConcreteTask], List[ConcreteTask]]:
    """Return ``(exact_writers, glob_writers)`` matching ``in_uri`` (of type ``in_object_type``).

    Cross-type producers are skipped. ``exact_writers`` write the identical concrete URI (overwrite
    chain); ``glob_writers`` write different URIs that the (possibly wildcard) ``in_uri`` selects
    (fan-in).
    """
    exact: List[ConcreteTask] = []
    glob: List[ConcreteTask] = []
    seen_exact = set()
    seen_glob = set()
    for task, out_uri, out_otype in index:
        if task is exclude:
            continue
        if out_otype != in_object_type:
            continue
        if not patterns_overlap(out_uri, in_uri, in_object_type):
            continue
        # Overwrite (exact) only when BOTH sides are concrete and identical; any wildcard on either
        # side is a fan-in edge (a wildcard producer/consumer must never be treated as an overwrite).
        if (
            is_concrete_uri(out_uri, in_object_type)
            and is_concrete_uri(in_uri, in_object_type)
            and same_uri(out_uri, in_uri, in_object_type)
        ):
            if task.task_id not in seen_exact:
                exact.append(task)
                seen_exact.add(task.task_id)
        elif task.task_id not in seen_glob:
            glob.append(task)
            seen_glob.add(task.task_id)
    return exact, glob
