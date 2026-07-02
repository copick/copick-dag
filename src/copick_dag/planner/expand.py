"""Static expansion: turn a ``GraphSpec`` into a flat list of fully-bound resolved nodes.

Handles the M3 constructs that are knowable at build time: the ``sweep`` matrix, per-node CSV
``expand``, output ``switch`` routing, ``=`` param expressions, and lowering of ``coalesce`` inputs
to ordered optional-overwrite writers. What remains data-dependent (split fan-out, ``*`` fan-in) is
left to URI edge inference + runtime resolution.
"""

from __future__ import annotations

import ast
import csv
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from copick_dag.ir.model import CoalesceSpec, FinalizeSpec, GraphNode, GraphSpec, SwitchSpec
from copick_dag.uris import substitute


@dataclass
class RNode:
    """A resolved node: sweep/CSV/switch/expr all applied; ids namespaced; coalesce lowered.

    ``inputs`` values are ``str`` or ``List[str]`` (coalesce already lowered away). ``runs``/``vs``
    are per-row overrides (``None`` => use the global axes).
    """

    id: str
    op: str
    params: Dict[str, Any]
    inputs: Dict[str, Any]
    out: Dict[str, str]
    optional: bool
    after: List[str]
    needs: Dict[str, str]
    scope: Optional[str]
    row_tag: str = ""
    runs: Optional[List[str]] = None
    vs: Optional[float] = None
    is_gc: bool = False


# ---------------------------------------------------------------------------
# safe arithmetic for `=` param expressions
# ---------------------------------------------------------------------------

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_ast(n: ast.AST):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        return _BIN[type(n.op)](_eval_ast(n.left), _eval_ast(n.right))
    if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY:
        return _UNARY[type(n.op)](_eval_ast(n.operand))
    raise ExpandError(f"unsupported expression element: {ast.dump(n)}")


class ExpandError(ValueError):
    pass


def _eval_expr(expr: str, bindings: Dict[str, Any]):
    s = substitute(expr, bindings)
    try:
        return _eval_ast(ast.parse(s, mode="eval").body)
    except ExpandError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ExpandError(f"bad expression {expr!r} -> {s!r}: {e}") from None


# ---------------------------------------------------------------------------
# value substitution
# ---------------------------------------------------------------------------


def _sub_val(v: Any, bindings: Dict[str, Any]) -> Any:
    if isinstance(v, str):
        if v.startswith("="):
            return _eval_expr(v[1:], bindings)
        return substitute(v, bindings)
    return v


def _sub_uri(u: str, bindings: Dict[str, Any]) -> str:
    return substitute(u, bindings)


def _resolve_switch(sw: SwitchSpec, bindings: Dict[str, Any]) -> str:
    key = str(bindings.get(sw.by, ""))
    if key in sw.switch:
        return sw.switch[key]
    if "default" in sw.switch:
        return sw.switch["default"]
    raise ExpandError(f"switch on {sw.by!r}={key!r} has no matching case and no default")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _primary_var(bind: Dict[str, int]) -> str:
    """The CSV var used for id-namespacing + normalization. Prefers ``name`` (the switch convention)
    so it does not depend on dict key order (an LLM / YAML round-trip may reorder ``bind``)."""
    return "name" if "name" in bind else next(iter(bind))


def _read_csv_rows(path: str, bind: Dict[str, int], normalize: Dict[str, str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    primary = _primary_var(bind)  # normalization target
    with open(path, newline="") as fh:
        for raw in csv.reader(fh):
            cols = [c.strip() for c in raw]
            if not cols or all(c == "" for c in cols):
                continue
            if any(idx > len(cols) or cols[idx - 1] == "" for idx in bind.values()):
                continue
            binding = {var: cols[idx - 1] for var, idx in bind.items()}
            binding[primary] = normalize.get(binding[primary], binding[primary])
            rows.append(binding)
    return rows


# ---------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------


def base_nodes(spec: GraphSpec) -> Tuple[List[GraphNode], set]:
    """spec.nodes + synthetic finalize nodes; return (nodes, gc_base_ids)."""
    nodes = list(spec.nodes)
    gc_ids: set = set()
    if spec.finalize:
        fin, gc_ids = _finalize_nodes(spec.finalize)
        nodes.extend(fin)
    return nodes, gc_ids


def _finalize_nodes(fin: FinalizeSpec) -> Tuple[List[GraphNode], set]:
    out: List[GraphNode] = []
    gc_ids: set = set()
    for i, c in enumerate(fin.copies):
        out.append(GraphNode(id=f"finalize_cp_{i}", op="core.cp", inputs={"src": c.src}, out={"dst": c.dst},
                             optional=c.optional, params={"overwrite": c.overwrite}))
    for i, m in enumerate(fin.moves):
        out.append(GraphNode(id=f"finalize_mv_{i}", op="core.mv", inputs={"src": m.src}, out={"dst": m.dst},
                             optional=m.optional, params={"overwrite": m.overwrite}))
    if fin.gc:
        for i, pat in enumerate(fin.gc.remove):
            nid = f"gc_rm_{i}"
            gc_ids.add(nid)
            out.append(GraphNode(id=nid, op="core.rm", inputs={"i": pat}, optional=True, params={"force": True}))
    return out, gc_ids


def _base_id(node: GraphNode, i: int) -> str:
    return node.id or f"{node.op.split('.')[-1]}_{i}"


def _row_context(row: Optional[Dict[str, Any]], ri: int) -> Tuple[Dict[str, Any], str, Optional[float], Optional[List[str]]]:
    if row is None:
        return {}, "", None, None
    row_tag = str(row.get("tag", f"row{ri}"))
    row_vs = float(row["vs"]) if row.get("vs") is not None else None
    row_runs = list(row["runs"]) if row.get("runs") is not None else None
    bindings = {k: v for k, v in row.items() if k != "runs"}
    return bindings, row_tag, row_vs, row_runs


def _expand_one(node: GraphNode, base_id: str, bindings: Dict[str, Any], row_tag: str,
                row_vs: Optional[float], row_runs: Optional[List[str]], is_gc: bool,
                id_suffix: str = "", default_optional: bool = False) -> RNode:
    nid = base_id + id_suffix + (f"__{row_tag}" if row_tag else "")

    inputs: Dict[str, Any] = {}
    for slot, val in node.inputs.items():
        if isinstance(val, CoalesceSpec):
            inputs[slot] = CoalesceSpec(coalesce=[_sub_uri(u, bindings) for u in val.coalesce])
        elif isinstance(val, list):
            inputs[slot] = [_sub_uri(u, bindings) for u in val]
        else:
            inputs[slot] = _sub_uri(val, bindings)

    out: Dict[str, str] = {}
    for slot, val in node.out.items():
        if isinstance(val, SwitchSpec):
            out[slot] = _sub_uri(_resolve_switch(val, bindings), bindings)
        else:
            out[slot] = _sub_uri(val, bindings)

    params = {k: _sub_val(v, bindings) for k, v in node.params.items()}
    after = [a + (f"__{row_tag}" if row_tag else "") for a in node.after]
    needs = {k: v + (f"__{row_tag}" if row_tag else "") for k, v in node.needs.items()}
    optional = default_optional if node.optional is None else bool(node.optional)

    return RNode(id=nid, op=node.op, params=params, inputs=inputs, out=out, optional=optional,
                 after=after, needs=needs, scope=node.scope, row_tag=row_tag, runs=row_runs,
                 vs=row_vs, is_gc=is_gc)


def expand(spec: GraphSpec) -> List[RNode]:
    nodes, gc_ids = base_nodes(spec)
    base_ids = [_base_id(n, i) for i, n in enumerate(nodes)]
    if len(set(base_ids)) != len(base_ids):
        dupes = sorted({x for x in base_ids if base_ids.count(x) > 1})
        raise ExpandError(f"duplicate node ids: {dupes}")

    rows = spec.sweep.over if spec.sweep else [None]
    rnodes: List[RNode] = []
    for ri, row in enumerate(rows):
        bindings, row_tag, row_vs, row_runs = _row_context(row, ri)
        for node, bid in zip(nodes, base_ids):
            is_gc = bid in gc_ids
            over_rows = node.over if node.over else [None]
            for oi, orow in enumerate(over_rows):
                ob = dict(bindings)
                osuffix = ""
                if orow is not None:
                    ob.update(orow)
                    osuffix = "__" + str(orow.get("tag", f"o{oi}"))
                if node.expand is not None:
                    csv_path = substitute(node.expand.from_csv, ob)
                    for crow in _read_csv_rows(csv_path, node.expand.bind, node.expand.normalize):
                        b = {**ob, **crow}
                        key = str(crow[_primary_var(node.expand.bind)])
                        rnodes.append(_expand_one(node, bid, b, row_tag, row_vs, row_runs, is_gc,
                                                  id_suffix=osuffix + f"__{key}", default_optional=True))
                else:
                    rnodes.append(_expand_one(node, bid, ob, row_tag, row_vs, row_runs, is_gc, id_suffix=osuffix))

    _ensure_unique(rnodes)
    return _lower_coalesce(rnodes)


def _ensure_unique(rnodes: List[RNode]) -> None:
    seen = set()
    for r in rnodes:
        if r.id in seen:
            raise ExpandError(f"duplicate resolved node id {r.id!r} after expansion")
        seen.add(r.id)


def _lower_coalesce(rnodes: List[RNode]) -> List[RNode]:
    """Replace each node with a coalesce input by ordered writers of the same output.

    ``coalesce([preferred, ..., fallback])`` -> writers in order [fallback (non-optional), ...,
    preferred (last, overwrites)]; every writer but the first is optional and ``after`` the previous,
    so the highest-priority *existing* input wins.
    """
    out: List[RNode] = []
    for r in rnodes:
        coal_slot = next((s for s, v in r.inputs.items() if isinstance(v, CoalesceSpec)), None)
        if coal_slot is None:
            out.append(r)
            continue
        candidates = r.inputs[coal_slot].coalesce  # [preferred, ..., fallback]
        ordered = list(reversed(candidates))  # [fallback, ..., preferred]
        prev_id: Optional[str] = None
        for k, cand in enumerate(ordered):
            wid = f"{r.id}__coal{k}"
            new_inputs = dict(r.inputs)
            new_inputs[coal_slot] = cand
            after = list(r.after)
            if prev_id is not None:
                after = after + [prev_id]
            out.append(RNode(id=wid, op=r.op, params=dict(r.params), inputs=new_inputs, out=dict(r.out),
                             optional=(k != 0) or r.optional, after=after, needs=dict(r.needs),
                             scope=r.scope, row_tag=r.row_tag, runs=r.runs, vs=r.vs, is_gc=r.is_gc))
            prev_id = wid
    return out
