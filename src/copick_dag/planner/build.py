"""Compile a ``GraphSpec`` into a ``ResolvedPlan`` (the concrete DAG).

Front-end (``planner/expand.py``) statically expands sweep/CSV/switch/expr/coalesce into a flat list
of :class:`~copick_dag.planner.expand.RNode`; this module turns each resolved node into one task
per ``(run, voxel_spacing)`` and infers data-dependency edges by URI overlap (grouped by
``(row_tag, run)`` so sweep rows never cross-match).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from copick_dag.ir.model import GraphSpec
from copick_dag.plan.concrete import SCOPE_PER_RUN, ConcreteTask, ResolvedPlan
from copick_dag.planner.axes import resolve_runs, resolve_vs_axis
from copick_dag.planner.context import ProjectView
from copick_dag.planner.edges import build_producer_index, find_producers
from copick_dag.planner.expand import RNode, expand
from copick_dag.plan.concrete import SCOPE_BARRIER
from copick_dag.registry import get_op
from copick_dag.uris import get_vs, substitute, with_vs


class PlanError(ValueError):
    pass


def _is_barrier(r: RNode) -> bool:
    return (r.scope or get_op(r.op).scope) == SCOPE_BARRIER


def plan(
    spec: GraphSpec,
    project: Optional[ProjectView] = None,
    validate_preconditions: bool = False,
) -> ResolvedPlan:
    """Build the concrete DAG (see module docstring)."""
    try:
        rnodes = expand(spec)
    except ValueError as e:
        raise PlanError(str(e)) from None

    for r in rnodes:
        try:
            get_op(r.op)
        except KeyError as e:
            raise PlanError(str(e)) from None

    rnode_by_id: Dict[str, RNode] = {r.id: r for r in rnodes}
    rnode_index: Dict[str, int] = {r.id: i for i, r in enumerate(rnodes)}

    global_runs = resolve_runs(spec, project)
    per_run_rnodes = [r for r in rnodes if not _is_barrier(r)]
    barrier_rnodes = [r for r in rnodes if _is_barrier(r)]

    tasks: List[ConcreteTask] = []
    groups: Dict[Tuple[str, Optional[str]], List[ConcreteTask]] = {}

    for r in per_run_rnodes:
        runs = r.runs if r.runs is not None else global_runs
        for run in runs:
            vs_axis = [r.vs] if r.vs is not None else resolve_vs_axis(spec, run, project)
            for vs in vs_axis:
                task = _build_task(r, run, vs)
                tasks.append(task)
                groups.setdefault((r.row_tag, run), []).append(task)

    warnings: List[str] = []
    external: List[str] = []
    for group in groups.values():
        _infer_group_deps(group, rnode_by_id, rnode_index, warnings, external)

    # barrier tasks (e.g. seg-stats): one per (node, vs), aggregating over the row's run-set
    per_run_tasks = list(tasks)
    for r in barrier_rnodes:
        run_scope = [x for x in (r.runs if r.runs is not None else global_runs) if x is not None]
        anchor = run_scope[0] if run_scope else None
        vs_list = [r.vs] if r.vs is not None else resolve_vs_axis(spec, anchor, project)
        for vs in vs_list:
            bt = _build_barrier_task(r, run_scope, vs)
            _infer_barrier_deps(bt, per_run_tasks, external)
            tasks.append(bt)

    if validate_preconditions and project is not None:
        _check_preconditions(tasks, external, project, warnings)

    rp = ResolvedPlan(config=spec.project.config, tasks=tasks, warnings=warnings, external_inputs=sorted(set(external)))
    rp.topo_order()  # surface cycles / dangling deps early
    return rp


# ---------------------------------------------------------------------------
# task construction
# ---------------------------------------------------------------------------


def _vs_tag(vs: Optional[float]) -> str:
    return "novs" if vs is None else repr(vs)


def _canon_uri(uri: str, vs: Optional[float], run: Optional[str], object_type: str) -> str:
    mapping: Dict[str, object] = {}
    if vs is not None:
        mapping["vs"] = vs
    if run is not None:
        mapping["run"] = run
    u = substitute(uri, mapping)
    if object_type == "segmentation" and vs is not None:
        u = with_vs(u, vs)
    return u


def _task_vs(r: RNode, axis_vs: Optional[float]) -> Optional[float]:
    if axis_vs is not None:
        return float(axis_vs)
    if "voxel_spacing" in r.params and r.params["voxel_spacing"] is not None:
        return float(r.params["voxel_spacing"])
    for val in list(r.inputs.values()) + list(r.out.values()):
        uris = val if isinstance(val, list) else [val]
        for u in uris:
            vs = get_vs(u)
            if vs not in (None, "*"):
                try:
                    return float(vs)
                except (TypeError, ValueError):
                    pass
    return None


def _build_task(r: RNode, run: Optional[str], axis_vs: Optional[float]) -> ConcreteTask:
    op = get_op(r.op)
    vs = _task_vs(r, axis_vs)

    inputs: Dict[str, object] = {}
    for slot, val in r.inputs.items():
        sspec = op.input_slot(slot)
        otype = sspec.object_type if sspec else "segmentation"
        if isinstance(val, list):
            inputs[slot] = [_canon_uri(u, vs, run, otype) for u in val]
        else:
            inputs[slot] = _canon_uri(val, vs, run, otype)

    outputs: Dict[str, str] = {}
    for slot, val in r.out.items():
        sspec = op.output_slot(slot)
        otype = sspec.object_type if sspec else "segmentation"
        outputs[slot] = _canon_uri(val, vs, run, otype)

    subst: Dict[str, object] = {}
    if vs is not None:
        subst["vs"] = vs
    if run is not None:
        subst["run"] = run
    params = {
        k: (substitute(v, subst) if isinstance(v, str) else v)
        for k, v in r.params.items()
        if k != "voxel_spacing"
    }

    scope = r.scope or op.scope or SCOPE_PER_RUN
    task_id = f"{r.id}::{run if run is not None else 'ALL'}::{_vs_tag(vs)}"
    return ConcreteTask(
        task_id=task_id,
        op_id=op.op_id,
        node_id=r.id,
        run=run,
        voxel_spacing=vs,
        params=params,
        inputs=inputs,
        outputs=outputs,
        scope=scope,
        deps=[],
        optional=bool(r.optional),
    )


def _build_barrier_task(r: RNode, run_scope: List[str], vs: Optional[float]) -> ConcreteTask:
    op = get_op(r.op)
    inputs: Dict[str, object] = {}
    for slot, val in r.inputs.items():
        sspec = op.input_slot(slot)
        otype = sspec.object_type if sspec else "segmentation"
        if isinstance(val, list):
            inputs[slot] = [_canon_uri(u, vs, None, otype) for u in val]
        else:
            inputs[slot] = _canon_uri(val, vs, None, otype)
    subst: Dict[str, object] = {"vs": vs} if vs is not None else {}
    params = {
        k: (substitute(v, subst) if isinstance(v, str) else v)
        for k, v in r.params.items()
        if k != "voxel_spacing"
    }
    return ConcreteTask(
        task_id=f"{r.id}::BARRIER::{_vs_tag(vs)}",
        op_id=op.op_id,
        node_id=r.id,
        run=None,
        voxel_spacing=vs,
        params=params,
        inputs=inputs,
        outputs={},
        scope=SCOPE_BARRIER,
        deps=[],
        optional=bool(r.optional),
        run_scope=list(run_scope),
    )


def _infer_barrier_deps(bt: ConcreteTask, per_run_tasks: List[ConcreteTask], external: List[str]) -> None:
    scope_tasks = [t for t in per_run_tasks if t.run in (bt.run_scope or [])]
    index = build_producer_index(scope_tasks)
    deps: set = set()
    for in_uri in bt.input_uris():
        exact, glob = find_producers(in_uri, index, exclude=bt)
        found = exact + glob
        if found:
            for p in found:
                deps.add(p.task_id)
        else:
            external.append(in_uri)
    bt.deps = sorted(deps)


# ---------------------------------------------------------------------------
# edge inference within a (row, run) group
# ---------------------------------------------------------------------------


def _infer_group_deps(
    group: List[ConcreteTask],
    rnode_by_id: Dict[str, RNode],
    rnode_index: Dict[str, int],
    warnings: List[str],
    external: List[str],
) -> None:
    index = build_producer_index(group)
    task_of: Dict[Tuple[str, Optional[float]], ConcreteTask] = {(t.node_id, t.voxel_spacing): t for t in group}

    for t in group:
        r = rnode_by_id[t.node_id]
        deps: set = set()

        for aid in r.after:
            dep = task_of.get((aid, t.voxel_spacing)) or _any_vs_task(group, aid)
            if dep is not None:
                deps.add(dep.task_id)
            else:
                warnings.append(f"node {t.node_id!r}: `after` references unknown node {aid!r}")

        for slot, val in t.inputs.items():
            if slot in r.needs:
                target = r.needs[slot]
                dep = task_of.get((target, t.voxel_spacing)) or _any_vs_task(group, target)
                if dep is not None:
                    deps.add(dep.task_id)
                    continue
                warnings.append(f"node {t.node_id!r}: `needs.{slot}` references unknown node {target!r}")

            for in_uri in val if isinstance(val, list) else [val]:
                exact, glob = find_producers(in_uri, index, exclude=t)
                if exact:
                    last = _last_writer(exact, rnode_index)
                    _check_overwrite_order(exact, last, rnode_by_id, in_uri)
                    deps.add(last.task_id)
                elif glob:
                    for g in glob:
                        deps.add(g.task_id)
                else:
                    external.append(in_uri)

        if r.is_gc:
            for other in group:
                if other is t or rnode_by_id[other.node_id].is_gc:
                    continue
                if other.voxel_spacing == t.voxel_spacing:
                    deps.add(other.task_id)

        t.deps = sorted(deps)


def _any_vs_task(group: List[ConcreteTask], node_id: str) -> Optional[ConcreteTask]:
    for t in group:
        if t.node_id == node_id:
            return t
    return None


def _last_writer(writers: List[ConcreteTask], rnode_index: Dict[str, int]) -> ConcreteTask:
    return max(writers, key=lambda t: rnode_index.get(t.node_id, -1))


def _check_overwrite_order(writers: List[ConcreteTask], last: ConcreteTask, rnode_by_id: Dict[str, RNode], uri: str) -> None:
    if len(writers) <= 1:
        return
    last_node = rnode_by_id[last.node_id]
    others = {w.node_id for w in writers if w.task_id != last.task_id}
    if not others.issubset(set(last_node.after)):
        raise PlanError(
            f"URI {uri!r} is written by multiple nodes {sorted(w.node_id for w in writers)} without an "
            f"explicit `after` ordering on the last writer {last.node_id!r}; add `after: "
            f"[{', '.join(sorted(others))}]` to disambiguate the copy-then-overwrite.",
        )


def _check_preconditions(tasks: List[ConcreteTask], external: List[str], project: ProjectView, warnings: List[str]) -> None:
    for t in tasks:
        if t.run is None:
            continue
        for u in [x for x in t.input_uris() if x in external]:
            if not project.segmentation_exists(u, t.run) and not t.optional:
                warnings.append(f"run {t.run!r}: required input {u!r} for {t.node_id!r} not found at plan time")
