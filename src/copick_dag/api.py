"""JSON-safe, high-level API for copick-dag — the boundary that MCP tools (and other callers) use.

copick-dag's internal structures (``ResolvedPlan``/``OpSpec`` dataclasses, pydantic ``GraphSpec``)
are not directly JSON-friendly. This module adds serializers + orchestration helpers that return
plain dicts/strings. All heavy logic stays in the planner / registry / backends; this only wires
them together and serializes.

A ``graph`` argument throughout is either a YAML/JSON **string** (how an LLM authors it) or a dict.
``config`` is an optional copick config path; when omitted, validate/plan/render/visualize run
**offline** (works when the graph's ``sweep``/``runs`` supply the run-set). ``run_graph`` requires it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import yaml

from copick_dag.backends.bash import render_bash
from copick_dag.backends.nextflow import render_nextflow, render_nextflow_config
from copick_dag.backends.memory import run_memory
from copick_dag.ir.model import GraphSpec
from copick_dag.plan.concrete import ConcreteTask, ResolvedPlan
from copick_dag.planner import PlanError, ProjectView, plan
from copick_dag.registry import OpSpec, ParamSpec, SlotSpec, all_ops, get_op
from copick_dag.registry.introspect import operation_available, resolve_command, selected_operation_issues

__all__ = [
    "load_graph",
    "graph_schema",
    "describe_ops",
    "describe_op",
    "list_examples",
    "validate_graph",
    "plan_graph",
    "render_graph",
    "visualize_graph",
    "run_graph",
    "op_to_dict",
    "plan_to_dict",
]


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def load_graph(graph: Union[str, Dict[str, Any]]) -> GraphSpec:
    """Parse a graph from a YAML/JSON string or a dict into a validated ``GraphSpec``."""
    if isinstance(graph, GraphSpec):
        return graph
    if isinstance(graph, dict):
        return GraphSpec.from_dict(graph)
    if isinstance(graph, str):
        data = yaml.safe_load(graph)
        if not isinstance(data, dict):
            raise ValueError("graph string must parse to a mapping (YAML/JSON object)")
        return GraphSpec.from_dict(data)
    raise TypeError(f"graph must be a str or dict, got {type(graph).__name__}")


def _spec_and_project(graph, config: Optional[str]):
    spec = load_graph(graph)
    project = None
    if config:
        spec.project.config = config
        project = ProjectView.open(config)
    return spec, project


# ---------------------------------------------------------------------------
# serializers
# ---------------------------------------------------------------------------


def _slot_to_dict(s: SlotSpec) -> Dict[str, Any]:
    return {
        "slot": s.slot,
        "object_type": s.object_type,
        "role": s.role,
        "flag": s.flag,
        "multiple": s.multiple,
        "positional": s.positional,
        "required": s.required,
        "declared": s.declared,
        "is_file": s.is_file,
    }


def _param_to_dict(p: ParamSpec) -> Dict[str, Any]:
    return {"name": p.name, "flag": p.flag, "kind": p.kind, "required": p.required, "default": p.default}


def op_to_dict(op: OpSpec, include_help: bool = True) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "op_id": op.op_id,
        "cli": " ".join(op.cli_path),
        "scope": op.scope,
        "vs_style": op.vs_style,
        "kind": op.kind,
        "streamable": op.streamable,
        "available": operation_available(op),
        "inputs": [_slot_to_dict(s) for s in op.inputs],
        "outputs": [_slot_to_dict(s) for s in op.outputs],
        "params": [_param_to_dict(p) for p in op.params],
    }
    if op.object_type_positional:
        d["object_type_positional"] = op.object_type_positional
    if op.plugin:
        d["plugin"] = op.plugin.replace("_", "-")
        d["install_extra"] = "torch" if op.plugin.replace("_", "-") == "copick-torch" else op.plugin
    if include_help:
        cmd = resolve_command(op.cli_path)
        d["help"] = (cmd.help or getattr(cmd, "short_help", "") or "") if cmd is not None else ""
    return d


def _task_to_dict(t: ConcreteTask) -> Dict[str, Any]:
    return {
        "task_id": t.task_id,
        "op_id": t.op_id,
        "node_id": t.node_id,
        "run": t.run,
        "voxel_spacing": t.voxel_spacing,
        "params": t.params,
        "inputs": t.inputs,
        "outputs": t.outputs,
        "scope": t.scope,
        "deps": t.deps,
        "optional": t.optional,
        "run_scope": t.run_scope,
    }


def plan_to_dict(rp: ResolvedPlan) -> Dict[str, Any]:
    return {
        "config": rp.config,
        "tasks": [_task_to_dict(t) for t in rp.topo_order()],
        "warnings": rp.warnings,
        "external_inputs": rp.external_inputs,
    }


def _summary(rp: ResolvedPlan) -> Dict[str, Any]:
    return {
        "task_count": len(rp.tasks),
        "runs": sorted({t.run for t in rp.tasks if t.run is not None}),
        "barriers": sum(1 for t in rp.tasks if t.scope == "barrier"),
        "op_counts": dict(Counter(t.op_id for t in rp.tasks)),
    }


# ---------------------------------------------------------------------------
# authoring aids
# ---------------------------------------------------------------------------


def graph_schema() -> Dict[str, Any]:
    """The JSON schema for a ``GraphSpec`` (so an LLM can author a valid graph)."""
    return GraphSpec.model_json_schema()


def describe_ops(include_help: bool = True) -> Dict[str, Any]:
    """List every copick-dag operation with its scope, I/O slots, params, and CLI help."""
    return {"ops": [op_to_dict(op, include_help) for _, op in sorted(all_ops().items())]}


def describe_op(op_id: str, include_help: bool = True) -> Dict[str, Any]:
    return op_to_dict(get_op(op_id), include_help)


_EXAMPLE_SIMPLE = """\
# A minimal 2-node graph: union all membrane annotations, then clip to the cell surface.
version: 1
project:
  config: ${CONFIG}          # copick config path (override via the tool's config_path arg)
defaults: {voxel_spacing: 20.0}
nodes:
  - op: logical.segop
    params: {operation: union}
    in:  {i: ["membrane:data-portal/*"]}
    out: {o: "membrane:temp/combined"}
  - op: logical.clipseg
    params: {max_distance: 200.0}
    in:  {i: "membrane:temp/combined", rs: "cell:napari/manual-scaled"}
    out: {o: "membrane:clipped/near-cell"}
"""


def list_examples() -> Dict[str, str]:
    """A couple of authoring templates. Combine with ``describe_ops`` + ``graph_schema``."""
    return {"simple_segop_clipseg": _EXAMPLE_SIMPLE}


# ---------------------------------------------------------------------------
# compile / render / visualize / run
# ---------------------------------------------------------------------------


def validate_graph(graph, config: Optional[str] = None) -> Dict[str, Any]:
    """Validate a graph (schema + registry + planning). Never raises; returns structured errors."""
    try:
        spec, project = _spec_and_project(graph, config)
    except Exception as e:  # noqa: BLE001 - parse/schema errors
        return {"valid": False, "errors": [f"parse/schema error: {e}"], "warnings": []}
    try:
        rp = plan(spec, project, validate_preconditions=project is not None)
    except PlanError as e:
        return {"valid": False, "errors": [str(e)], "warnings": []}
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "errors": [f"{type(e).__name__}: {e}"], "warnings": []}
    missing = selected_operation_issues(rp)
    if missing:
        return {"valid": False, "errors": missing, "warnings": rp.warnings, **_summary(rp)}
    return {"valid": True, "errors": [], "warnings": rp.warnings, **_summary(rp)}


def plan_graph(graph, config: Optional[str] = None) -> Dict[str, Any]:
    """Compile a graph to the concrete DAG (tasks + deps) with a summary."""
    spec, project = _spec_and_project(graph, config)
    rp = plan(spec, project, validate_preconditions=project is not None)
    d = plan_to_dict(rp)
    d["summary"] = _summary(rp)
    return d


def render_graph(graph, config: Optional[str] = None, backend: str = "bash") -> Dict[str, Any]:
    """Render a graph to a runnable artifact (``bash`` script or ``nextflow`` workflow)."""
    spec, project = _spec_and_project(graph, config)
    rp = plan(spec, project)
    if backend == "bash":
        return {"backend": "bash", "script": render_bash(rp)}
    if backend == "nextflow":
        return {"backend": "nextflow", "script": render_nextflow(rp), "nextflow_config": render_nextflow_config()}
    raise ValueError(f"unknown backend {backend!r}; expected 'bash' or 'nextflow'")


def visualize_graph(graph, config: Optional[str] = None) -> Dict[str, Any]:
    """Return a Mermaid flowchart + a text listing of the concrete DAG."""
    spec, project = _spec_and_project(graph, config)
    rp = plan(spec, project)
    return {"mermaid": plan_to_mermaid(rp), "text": plan_to_text(rp)}


def run_graph(graph, config: str, backend: str = "memory", workers: int = 1) -> Dict[str, Any]:
    """EXECUTE a graph, writing results to the copick store. ``memory`` or ``bash`` backends.

    This mutates the project. ``nextflow`` execution needs the external binary — use
    ``render_graph(backend='nextflow')`` to emit the workflow and run it yourself.
    """
    if not config:
        raise ValueError("run_graph requires a config path")
    spec, project = _spec_and_project(graph, config)
    rp = plan(spec, project, validate_preconditions=True)

    if backend == "memory":
        stats = run_memory(rp, workers=workers, root=project.root)
        hard = [e for e in stats.errors if e.startswith("[ERROR]")]
        return {
            "backend": "memory",
            "runs": stats.runs,
            "project_parses": stats.project_parses,
            "cache_hits": stats.total_hits,
            "writes": stats.total_writes,
            "errors": stats.errors,
            "ok": not hard,
            "warnings": rp.warnings,
        }
    if backend == "bash":
        import subprocess

        proc = subprocess.run(["bash", "-c", render_bash(rp)], capture_output=True, text=True)
        return {
            "backend": "bash",
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stderr_tail": proc.stderr[-2000:],
            "warnings": rp.warnings,
        }
    if backend == "nextflow":
        raise ValueError(
            "nextflow execution requires the nextflow binary; use render_graph(backend='nextflow') "
            "to emit the workflow and run it yourself.",
        )
    raise ValueError(f"unknown backend {backend!r}; expected 'memory' or 'bash'")


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------


def _mid(task_id: str) -> str:
    return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", task_id)


def plan_to_mermaid(rp: ResolvedPlan) -> str:
    """Render the concrete DAG as a Mermaid flowchart."""
    order = rp.topo_order()
    lines: List[str] = ["graph TD"]
    for t in order:
        label = f"{t.node_id}<br/>{t.op_id}"
        if t.run is not None:
            label += f"<br/>@{t.run}"
        shape = ("([", "])") if t.scope == "barrier" else ('["', '"]')
        lines.append(f"  {_mid(t.task_id)}{shape[0]}{label}{shape[1]}")
    for t in order:
        for d in t.deps:
            lines.append(f"  {_mid(d)} --> {_mid(t.task_id)}")
    return "\n".join(lines)


def plan_to_text(rp: ResolvedPlan) -> str:
    """Render the concrete DAG as a text listing (mirrors the ``copick-dag show`` CLI)."""
    lines = [f"# {len(rp.tasks)} tasks over runs {sorted({t.run for t in rp.tasks})}"]
    for t in rp.topo_order():
        deps = ", ".join(t.deps) if t.deps else "-"
        lines.append(f"{t.task_id}  [{t.op_id}]  deps=({deps})")
    return "\n".join(lines)
