"""Render a ``ResolvedPlan`` as a Nextflow DSL2 workflow (+ ``nextflow.config``).

Strategy: **one process per ``ConcreteTask``** (isomorphic to the already-verified concrete DAG).
Each process runs exactly the bash backend's command (reusing ``task_argv``) as a fully-literal,
bash-single-quoted Groovy ``'''…'''`` script block — so there is zero Groovy ``$`` interpolation and
zero bash glob expansion (URIs contain ``*``). Dependencies are done-signal value channels, not
staged files (copick objects live in a shared store addressed by the config).

Optional tasks mirror bash exactly: ``… || true`` in the script **and** the done token is still
emitted, so a failed-optional task (e.g. clipseg with an absent microtubules reference) never
starves its mandatory downstreams. See the M4 plan for why ``errorStrategy 'ignore'`` is wrong here.

Two entry points:

* :func:`render_nextflow` — a standalone top-level ``workflow {}`` you run directly (``copick-dag
  run/render --backend nextflow``).
* :func:`render_subworkflow` — a **named, composable** ``workflow NAME { take/main/emit }`` (plus its
  processes) that another Nextflow pipeline can ``include`` and call. Each process can carry a
  ``label`` (so the host pipeline's config assigns the container/resources) and a per-DAG name
  prefix (so several DAGs coexist in one module without process-name collisions). This is the
  idiomatic "compile a DAG into the host pipeline" path — one ``nextflow run``, one resume history,
  no nested driver.
"""

from __future__ import annotations

import re
import textwrap
from typing import Dict, List, Optional

from copick_dag.backends.bash.emitter import task_argv
from copick_dag.plan.concrete import ConcreteTask, ResolvedPlan

_NON_IDENT = re.compile(r"[^A-Za-z0-9_]")


def _sanitize(task_id: str) -> str:
    s = _NON_IDENT.sub("_", task_id)
    if not s or not re.match(r"[A-Za-z_]", s):
        s = "_" + s
    return s


def _name_map(tasks: List[ConcreteTask]) -> Dict[str, str]:
    """task_id -> unique, valid Groovy identifier base (shared by process `p_` and channel `c_`)."""
    used: set = set()
    mapping: Dict[str, str] = {}
    for t in tasks:
        base = _sanitize(t.task_id)
        name, k = base, 1
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        mapping[t.task_id] = name
    return mapping


def _bash_sq(s: str) -> str:
    """Bash single-quote a single argv element (safe against globs, $, spaces)."""
    return "'" + s.replace("'", "'\\''") + "'"


def _groovy_sq(s: str) -> str:
    """Groovy single-quoted string literal."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _command(task: ConcreteTask, config: str) -> str:
    cmd = "copick " + " ".join(_bash_sq(a) for a in task_argv(task, config))
    if task.optional:
        cmd += " || true"
    return cmd


def _gate_expr(task: ConcreteTask, names: Dict[str, str], start_expr: str = "start") -> str:
    """Channel expression that gates ``task``: its deps' done-channels, or ``start_expr`` for roots.

    ``start_expr`` is the top-level ``start`` value channel for a standalone workflow, or the named
    sub-workflow's ``take:`` input (``gate``) when composed into a host pipeline.
    """
    deps = task.deps
    if not deps:
        return start_expr
    chans = [f"c_{names[d]}" for d in deps]
    if len(chans) == 1:
        return chans[0]
    return f"{chans[0]}.mix({', '.join(chans[1:])}).collect()"


def _emit_process(
    task: ConcreteTask,
    names: Dict[str, str],
    config: str,
    *,
    proc_prefix: str,
    label: Optional[str],
) -> List[str]:
    """Emit one process definition. ``label`` (when given) lets a host pipeline's config pick the
    container/resources for the ``copick`` command; ``proc_prefix`` namespaces the process name."""
    pname = f"{proc_prefix}{names[task.task_id]}"
    out = [f"process {pname} {{"]
    if label:
        out.append(f"  label {_groovy_sq(label)}")
    out += [
        f"  tag {_groovy_sq(task.task_id)}",
        "  input:",
        "    val _gate",
        "  output:",
        f"    val {_groovy_sq(task.task_id)}",
        "  script:",
        "  '''",
        "  " + _command(task, config),
        "  '''",
        "}",
        "",
    ]
    return out


def _emit_wiring(
    order: List[ConcreteTask],
    names: Dict[str, str],
    *,
    proc_prefix: str,
    start_expr: str,
) -> List[str]:
    """Emit the ``c_x = <proc>(<gate>).first()`` lines wiring processes by their dep channels."""
    out: List[str] = []
    for t in order:
        pname = f"{proc_prefix}{names[t.task_id]}"
        cvar = f"c_{names[t.task_id]}"
        out.append(f"  {cvar} = {pname}({_gate_expr(t, names, start_expr)}).first()")
    return out


def _done_expr(order: List[ConcreteTask], names: Dict[str, str]) -> str:
    """A single value channel that emits once every task's done-token has arrived."""
    cvars = [f"c_{names[t.task_id]}" for t in order]
    if not cvars:
        return "Channel.value('done')"
    if len(cvars) == 1:
        return f"{cvars[0]}.collect()"
    return f"{cvars[0]}.mix({', '.join(cvars[1:])}).collect()"


def render_nextflow(plan: ResolvedPlan) -> str:
    """Render the whole plan as a standalone Nextflow DSL2 workflow string."""
    order = plan.topo_order()
    names = _name_map(order)
    config = plan.config

    lines: List[str] = [
        "// Generated by copick-dag (nextflow backend). Do not edit by hand.",
        "// One process per task; deps are done-signal channels; the shared copick store is",
        "// addressed by the baked-in config path (no file staging).",
        "nextflow.enable.dsl = 2",
        "",
    ]

    for t in order:
        lines += _emit_process(t, names, config, proc_prefix="p_", label=None)

    lines += ["workflow {", "  start = Channel.value('start')"]
    lines += _emit_wiring(order, names, proc_prefix="p_", start_expr="start")
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

    Unlike :func:`render_nextflow` (a standalone top-level ``workflow {}``), this emits the task
    processes followed by ``workflow <workflow_name> { take: gate; main: …; emit: done }`` so a host
    pipeline can ``include { <workflow_name> } from …`` and call it in the *same* ``nextflow run``.
    Root tasks gate on the sub-workflow's ``take:`` input (``gate``) instead of a local ``start``
    channel, and ``emit: done`` is a single value channel that fires once every task has finished —
    the handle the host pipeline uses to gate whatever runs next.

    Args:
        workflow_name: emitted ``workflow <name>`` identifier (sanitized to a valid Groovy id).
        label: optional process ``label`` directive; the host pipeline's config maps the label to a
            container + SLURM resources (so every ``copick`` task runs in the right image).
        process_prefix: process-name prefix for global uniqueness when multiple DAGs live in one
            module file. Defaults to ``f"{workflow_name}_p_"``.
        include_header: prepend ``nextflow.enable.dsl = 2`` (for a one-workflow standalone file).
    """
    wf = _sanitize(workflow_name)
    proc_prefix = process_prefix if process_prefix is not None else f"{wf}_p_"
    order = plan.topo_order()
    names = _name_map(order)
    config = plan.config

    lines: List[str] = []
    if include_header:
        lines += ["nextflow.enable.dsl = 2", ""]
    lines += [
        f"// copick-dag sub-workflow {wf} (generated). Do not edit by hand.",
        "",
    ]

    for t in order:
        lines += _emit_process(t, names, config, proc_prefix=proc_prefix, label=label)

    lines += [
        f"workflow {wf} {{",
        "  take:",
        "    gate",
        "  main:",
    ]
    lines += _emit_wiring(order, names, proc_prefix=proc_prefix, start_expr="gate")
    lines += [
        "  emit:",
        f"    done = {_done_expr(order, names)}",
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
