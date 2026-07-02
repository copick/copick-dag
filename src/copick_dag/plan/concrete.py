"""The concrete DAG: the backend-agnostic lowering target produced by the planner.

A :class:`ResolvedPlan` is a flat, topologically-sortable list of :class:`ConcreteTask`. Each task
is a single unit of work bound to a concrete run and (for segmentation ops) a concrete voxel
spacing. Every backend -- bash, in-memory, nextflow -- reads *only* this structure; it never sees
the abstract IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# scope of a task on the run axis
SCOPE_PER_RUN = "per_run"
SCOPE_BARRIER = "barrier"

# dynamic-frontier markers (resolution deferred to run time); unused until M3
DYN_FANIN = "fanin"
DYN_FANOUT = "fanout"
DYN_ROUTE = "route"

URIValue = Union[str, List[str]]


@dataclass
class ConcreteTask:
    """One concrete unit of work in the resolved DAG.

    Attributes:
        task_id: Globally-unique id, ``f"{node_id}::{run}::{vs}"`` for per-run tasks.
        op_id: Registry key, e.g. ``"logical.segop"`` / ``"core.cp"``.
        node_id: The abstract node this task came from.
        run: Concrete run name, or ``None`` for a graph-level barrier that owns all runs.
        voxel_spacing: Resolved float spacing for segmentation ops (``None`` if not applicable).
        params: Resolved scalar parameters (``operation``, ``max_distance``, ``margin`` ...).
        inputs: slot -> concrete input URI(s). A value may be a single URI or a list.
        outputs: slot -> concrete output URI.
        scope: ``per_run`` or ``barrier``.
        deps: task_ids this task depends on.
        optional: try-if-exists; failure/absence is tolerated.
        dynamic: ``None`` for statically-resolved tasks, else a ``DYN_*`` marker.
    """

    task_id: str
    op_id: str
    node_id: str
    run: Optional[str]
    voxel_spacing: Optional[float] = None
    params: Dict[str, object] = field(default_factory=dict)
    inputs: Dict[str, URIValue] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    scope: str = SCOPE_PER_RUN
    deps: List[str] = field(default_factory=list)
    optional: bool = False
    dynamic: Optional[str] = None
    run_scope: Optional[List[str]] = None  # for barrier tasks: the runs to aggregate over

    def input_uris(self) -> List[str]:
        """Flatten all input URIs across slots into a single list."""
        out: List[str] = []
        for v in self.inputs.values():
            if isinstance(v, list):
                out.extend(v)
            elif v is not None:
                out.append(v)
        return out

    def output_uris(self) -> List[str]:
        return [v for v in self.outputs.values() if v is not None]


@dataclass
class ResolvedPlan:
    """A topologically-sortable collection of concrete tasks plus the project config path."""

    config: str
    tasks: List[ConcreteTask] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    external_inputs: List[str] = field(default_factory=list)

    def by_id(self) -> Dict[str, ConcreteTask]:
        return {t.task_id: t for t in self.tasks}

    def topo_order(self) -> List[ConcreteTask]:
        """Deterministic topological order (Kahn's algorithm, ties broken by insertion order).

        Raises:
            ValueError: if the dependency graph contains a cycle or references an unknown task.
        """
        by_id = self.by_id()
        indeg: Dict[str, int] = {t.task_id: 0 for t in self.tasks}
        adj: Dict[str, List[str]] = {t.task_id: [] for t in self.tasks}
        for t in self.tasks:
            for d in t.deps:
                if d not in by_id:
                    raise ValueError(f"task {t.task_id!r} depends on unknown task {d!r}")
                adj[d].append(t.task_id)
                indeg[t.task_id] += 1

        # ready queue in original insertion order for stable output
        order_index = {t.task_id: i for i, t in enumerate(self.tasks)}
        ready = sorted((tid for tid, d in indeg.items() if d == 0), key=order_index.get)
        result: List[ConcreteTask] = []
        while ready:
            tid = ready.pop(0)
            result.append(by_id[tid])
            newly: List[str] = []
            for nxt in adj[tid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    newly.append(nxt)
            # keep determinism: merge newly-ready by original index
            for tid2 in sorted(newly, key=order_index.get):
                ready.append(tid2)
            ready.sort(key=order_index.get)

        if len(result) != len(self.tasks):
            remaining = [t.task_id for t in self.tasks if t.task_id not in {r.task_id for r in result}]
            raise ValueError(f"dependency cycle detected among tasks: {remaining}")
        return result

    def consumers_of(self) -> Dict[str, List[str]]:
        """Map each task_id -> list of task_ids that depend on it (used for refcount eviction)."""
        cons: Dict[str, List[str]] = {t.task_id: [] for t in self.tasks}
        for t in self.tasks:
            for d in t.deps:
                cons.setdefault(d, []).append(t.task_id)
        return cons
