"""Render a ``ResolvedPlan`` as a Nextflow DSL2 workflow (+ ``nextflow.config``).

Strategy: **one process per node** (per voxel spacing), fanned out over a channel of runs -- so the
module size is independent of the run count. (A per-run-*unrolled* emission -- one process per
run x node -- generates a monolithic workflow that hits Groovy's hard 65,535-char string/method
limit at a few hundred runs.)

Each per-run process runs exactly the bash backend's command (``copick `` + ``task_argv``) with the
run substituted as a Nextflow ``${run}`` input **inside bash single-quotes** (so run values stay
glob/``$``-safe); every other argument is a run-independent literal, giving command-equivalence with
the bash/memory backends.

Dependencies are run-keyed value channels (copick objects live in a shared store addressed by the
config, so nothing is staged): a node consumes its dependency node's per-run channel; a multi-input
node ``join``s its dependency channels by run (pairing a run with its *own* upstreams -- a within-run
match, never cross-run); fan-out reuses a channel across several consumer processes (DSL2 broadcast).
Barrier (cross-run aggregation) nodes stay a single process that bakes the whole ``-r r1 -r r2 ...``
run-set, gated by ``.collect()``.

Optional tasks mirror bash exactly: ``... || true`` in the script and the done token (the run) is
still emitted, so a failed-optional node never starves its mandatory downstreams.

Two entry points:

* :func:`render_nextflow` -- a standalone top-level ``workflow {}`` (``copick-dag run/render``).
* :func:`render_subworkflow` -- a named, composable ``workflow NAME { take/main/emit }`` a host
  pipeline can ``include`` and call in the *same* ``nextflow run`` (one resume history, no nested
  driver). Root nodes fan the runs out of the sub-workflow's ``take:`` gate; ``emit: done`` fires
  once every node has finished every run.
"""

from __future__ import annotations

import dataclasses
import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from copick_dag.backends.bash.emitter import task_argv
from copick_dag.plan.concrete import SCOPE_BARRIER, ConcreteTask, ResolvedPlan

_NON_IDENT = re.compile(r"[^A-Za-z0-9_]")
# A token unlikely to occur in any argv; replaced by the Nextflow ${run} placeholder *after* Groovy
# escaping, so only the run interpolates.
_RUN_SENTINEL = "\x01COPICK_DAG_RUN\x01"

UnitKey = Tuple[str, Optional[float]]


def _sanitize(task_id: str) -> str:
    s = _NON_IDENT.sub("_", task_id)
    if not s or not re.match(r"[A-Za-z_]", s):
        s = "_" + s
    return s


def _bash_sq(s: str) -> str:
    """Bash single-quote a single argv element (safe against globs, $, spaces)."""
    return "'" + s.replace("'", "'\\''") + "'"


def _groovy_sq(s: str) -> str:
    """Groovy single-quoted string literal."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


@dataclass
class _Unit:
    """One process's worth of work: an abstract node at a fixed voxel spacing, over a set of runs."""

    key: UnitKey
    node_id: str
    vs: Optional[float]
    rep: ConcreteTask          # representative task (supplies the command + op/scope/optional)
    runs: List[str]            # per_run: ordered runs; barrier: []
    scope: str
    optional: bool
    dep_keys: List[UnitKey] = field(default_factory=list)


def _build_units(plan: ResolvedPlan) -> List[_Unit]:
    """Group the plan's tasks into per-(node, voxel-spacing) units in a valid topological order.

    A unit's first appearance in ``topo_order`` is a valid topological position for it (any dependency
    task -- hence its unit -- appears earlier). Dynamic-frontier tasks (deferred to run time) are
    rejected rather than silently mis-grouped.
    """
    if any(t.dynamic for t in plan.tasks):
        raise NotImplementedError(
            "the channel-based nextflow renderer does not support dynamic-frontier tasks",
        )
    by_id = plan.by_id()
    units: Dict[UnitKey, _Unit] = {}
    run_seen: Dict[UnitKey, set] = {}
    for t in plan.topo_order():
        key = (t.node_id, t.voxel_spacing)
        u = units.get(key)
        if u is None:
            u = _Unit(key=key, node_id=t.node_id, vs=t.voxel_spacing, rep=t,
                      runs=[], scope=t.scope, optional=t.optional)
            units[key] = u
            run_seen[key] = set()
        if t.run is not None and t.run not in run_seen[key]:
            run_seen[key].add(t.run)
            u.runs.append(t.run)
        for d in t.deps:
            dep = by_id[d]
            dk = (dep.node_id, dep.voxel_spacing)
            if dk != key and dk not in u.dep_keys:
                u.dep_keys.append(dk)
    return list(units.values())


def _unit_names(units: List[_Unit]) -> Dict[UnitKey, str]:
    """unit key -> unique valid Groovy identifier base (shared by process ``p_`` and channel ``c_``)."""
    used: set = set()
    mapping: Dict[UnitKey, str] = {}
    for u in units:
        base = _sanitize(u.node_id)
        name, k = base, 1
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        mapping[u.key] = name
    return mapping


def _run_axis(plan: ResolvedPlan) -> List[str]:
    """Distinct run names in first-appearance (topological) order."""
    seen: set = set()
    axis: List[str] = []
    for t in plan.topo_order():
        if t.run is not None and t.run not in seen:
            seen.add(t.run)
            axis.append(t.run)
    return axis


def _command_template(rep: ConcreteTask, config: str) -> str:
    """Run-templated command for a per-run process: the run is a ``${run}`` placeholder inside bash
    single-quotes; every other char is Groovy-escaped for a triple-double-quoted block so ONLY
    ``${run}`` interpolates."""
    tpl = dataclasses.replace(rep, run=_RUN_SENTINEL)
    cmd = "copick " + " ".join(_bash_sq(a) for a in task_argv(tpl, config))
    if rep.optional:
        cmd += " || true"
    esc = cmd.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return esc.replace(_RUN_SENTINEL, "${run}")


def _command_literal(rep: ConcreteTask, config: str) -> str:
    """Fully-literal command (barriers bake the whole run-set); emitted in a ``'''...'''`` block."""
    cmd = "copick " + " ".join(_bash_sq(a) for a in task_argv(rep, config))
    if rep.optional:
        cmd += " || true"
    return cmd


def _emit_process(u: _Unit, names: Dict[UnitKey, str], config: str, *,
                  proc_prefix: str, label: Optional[str]) -> List[str]:
    """Emit one process: a barrier bakes the run-set (single run); a per-run node takes ``val run``."""
    pname = f"{proc_prefix}{names[u.key]}"
    out = [f"process {pname} {{"]
    if label:
        out.append(f"  label {_groovy_sq(label)}")
    if u.scope == SCOPE_BARRIER:
        out += [
            f"  tag {_groovy_sq(u.node_id)}",
            "  input:",
            "    val _gate",
            "  output:",
            "    val 'done'",
            "  script:",
            "  '''",
            "  " + _command_literal(u.rep, config),
            "  '''",
            "}",
            "",
        ]
    else:
        out += [
            f'  tag "{u.node_id}::${{run}}"',
            "  input:",
            "    val run",
            "  output:",
            "    val run",
            "  script:",
            '  """',
            "  " + _command_template(u.rep, config),
            '  """',
            "}",
            "",
        ]
    return out


def _root_runs_expr(gate_kind: str) -> str:
    """The channel of all runs feeding a root unit (``runsList`` is a def'd Groovy list in scope)."""
    return "Channel.fromList(runsList)" if gate_kind == "standalone" else "gate.flatMap { runsList }"


def _gate_expr(u: _Unit, names: Dict[UnitKey, str], unit_by_key: Dict[UnitKey, _Unit],
               gate_kind: str) -> str:
    """Channel expression gating ``u``: the run channel (roots), the dep channel (single dep), a
    per-run ``join`` of dep channels (multi-input), or ``.collect()`` of deps (barriers)."""
    per_run_deps = [dk for dk in u.dep_keys if unit_by_key[dk].scope != SCOPE_BARRIER]
    barrier_deps = [dk for dk in u.dep_keys if unit_by_key[dk].scope == SCOPE_BARRIER]

    if u.scope == SCOPE_BARRIER:
        if not u.dep_keys:
            return f"{_root_runs_expr(gate_kind)}.collect()"
        cvars = [f"c_{names[dk]}" for dk in u.dep_keys]
        base = cvars[0] if len(cvars) == 1 else f"{cvars[0]}.mix({', '.join(cvars[1:])})"
        return f"{base}.collect()"

    # per-run unit
    if barrier_deps:
        raise NotImplementedError(
            "a per-run node depending on a barrier is not supported by the channel renderer",
        )
    if not per_run_deps:
        return _root_runs_expr(gate_kind)
    expr = f"c_{names[per_run_deps[0]]}"
    for dk in per_run_deps[1:]:
        expr = f"{expr}.join(c_{names[dk]})"
    return expr


def _emit_wiring(units: List[_Unit], names: Dict[UnitKey, str], unit_by_key: Dict[UnitKey, _Unit],
                 *, proc_prefix: str, gate_kind: str) -> List[str]:
    """Emit ``c_x = <proc>(<gate>)`` lines wiring each node's process to its run/dep channel."""
    out: List[str] = []
    for u in units:
        cvar = f"c_{names[u.key]}"
        pname = f"{proc_prefix}{names[u.key]}"
        out.append(f"  {cvar} = {pname}({_gate_expr(u, names, unit_by_key, gate_kind)})")
    return out


def _done_expr(units: List[_Unit], names: Dict[UnitKey, str]) -> str:
    """A single value channel that fires once every node's every run has finished."""
    cvars = [f"c_{names[u.key]}" for u in units]
    if not cvars:
        return "Channel.value('done')"
    if len(cvars) == 1:
        return f"{cvars[0]}.collect()"
    return f"{cvars[0]}.mix({', '.join(cvars[1:])}).collect()"


def _runs_list_literal(runs: List[str]) -> str:
    return "[" + ", ".join(_groovy_sq(r) for r in runs) + "]"


def render_nextflow(plan: ResolvedPlan) -> str:
    """Render the whole plan as a standalone Nextflow DSL2 workflow string."""
    units = _build_units(plan)
    names = _unit_names(units)
    unit_by_key = {u.key: u for u in units}
    runs = _run_axis(plan)
    config = plan.config

    lines: List[str] = [
        "// Generated by copick-dag (nextflow backend). Do not edit by hand.",
        "// One process per node (per voxel spacing); runs fan out over a channel; deps are run-keyed",
        "// done-signal channels; the shared copick store is addressed by the baked-in config path.",
        "nextflow.enable.dsl = 2",
        "",
    ]
    for u in units:
        lines += _emit_process(u, names, config, proc_prefix="p_", label=None)

    lines += ["workflow {", f"  def runsList = {_runs_list_literal(runs)}"]
    lines += _emit_wiring(units, names, unit_by_key, proc_prefix="p_", gate_kind="standalone")
    lines += ["}", ""]
    return "\n".join(lines)


def render_subworkflow(
    plan: ResolvedPlan,
    workflow_name: str,
    *,
    label: Optional[str] = None,
    process_prefix: Optional[str] = None,
    include_header: bool = False,
) -> str:
    """Render the plan as a **named, composable** DSL2 sub-workflow (``take/main/emit``).

    Root nodes fan the runs out of the ``take:`` ``gate`` (a value channel from the host pipeline's
    prior phase); ``emit: done`` is a single value channel that fires once every node has finished
    every run -- the handle the host pipeline uses to gate whatever runs next.

    Args:
        workflow_name: emitted ``workflow <name>`` identifier (sanitized to a valid Groovy id).
        label: optional process ``label``; the host config maps it to a container + SLURM resources.
        process_prefix: process-name prefix for global uniqueness when multiple DAGs share one module.
            Defaults to ``f"{workflow_name}_p_"``.
        include_header: prepend ``nextflow.enable.dsl = 2`` (for a one-workflow standalone file).
    """
    wf = _sanitize(workflow_name)
    proc_prefix = process_prefix if process_prefix is not None else f"{wf}_p_"
    units = _build_units(plan)
    names = _unit_names(units)
    unit_by_key = {u.key: u for u in units}
    runs = _run_axis(plan)
    config = plan.config

    lines: List[str] = []
    if include_header:
        lines += ["nextflow.enable.dsl = 2", ""]
    lines += [f"// copick-dag sub-workflow {wf} (generated). Do not edit by hand.", ""]

    for u in units:
        lines += _emit_process(u, names, config, proc_prefix=proc_prefix, label=label)

    lines += [
        f"workflow {wf} {{",
        "  take:",
        "    gate",
        "  main:",
        f"  def runsList = {_runs_list_literal(runs)}",
    ]
    lines += _emit_wiring(units, names, unit_by_key, proc_prefix=proc_prefix, gate_kind="composable")
    lines += [
        "  emit:",
        f"    done = {_done_expr(units, names)}",
        "}",
        "",
    ]
    return "\n".join(lines)


def render_nextflow_config() -> str:
    """Render a ``nextflow.config`` with a default ``local`` profile and a ``slurm`` profile."""
    return textwrap.dedent(
        """\
        // Generated by copick-dag (nextflow backend).
        params.queue          = null      // SLURM partition/queue
        params.clusterOptions = null      // extra sbatch options, e.g. '--account=group.czii'
        params.maxForks       = 4

        process.maxForks = params.maxForks

        profiles {
            standard { process.executor = 'local' }
            slurm {
                process.executor       = 'slurm'
                process.queue          = params.queue
                process.clusterOptions = params.clusterOptions
                process.cpus           = 1
            }
        }
        """,
    )
