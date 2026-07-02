"""Planner unit tests: edge inference and the voxel-spacing axis."""

import pytest

from copick_dag.ir.model import DiscoverSpec, GraphSpec
from copick_dag.planner import PlanError, plan
from copick_dag.planner.axes import AxisError, resolve_vs_axis


def _spec(nodes, **top):
    base = {"version": 1, "project": {"config": "${CFG}"}, "defaults": {"voxel_spacing": 20.0}, "nodes": nodes}
    base.update(top)
    return GraphSpec.from_dict(base)


def _dep_ids(rp, node_id):
    task = next(t for t in rp.tasks if t.node_id == node_id)
    return {rp.by_id()[d].node_id for d in task.deps}


def test_linear_chain_edges():
    spec = _spec(
        [
            {"id": "a", "op": "logical.segop", "params": {"operation": "union"}, "in": {"i": ["m:dp/*"]}, "out": {"o": "m:t/a"}},
            {"id": "b", "op": "logical.enclosed", "params": {"margin": 1}, "in": {"i1": "m:t/a", "i2": "c:u/s"}, "out": {"o": "m:t/b"}},
        ]
    )
    rp = plan(spec)
    assert _dep_ids(rp, "b") == {"a"}  # b reads a's output; c:u/s is external
    assert _dep_ids(rp, "a") == set()


def test_sibling_independence():
    spec = _spec(
        [
            {"id": "a", "op": "logical.segop", "params": {"operation": "difference"}, "in": {"i": ["x:u/s", "y:u/s"]}, "out": {"o": "x:t/a"}},
            {"id": "b", "op": "logical.segop", "params": {"operation": "difference"}, "in": {"i": ["p:u/s", "q:u/s"]}, "out": {"o": "p:t/b"}},
        ]
    )
    rp = plan(spec)
    assert _dep_ids(rp, "a") == set()
    assert _dep_ids(rp, "b") == set()


def test_glob_fanin_depends_on_all_producers():
    spec = _spec(
        [
            {"id": "a", "op": "logical.segop", "params": {"operation": "difference"}, "in": {"i": ["x:u/s", "y:u/s"]}, "out": {"o": "seg:filt/one"}},
            {"id": "b", "op": "logical.segop", "params": {"operation": "difference"}, "in": {"i": ["x:u/s", "y:u/s"]}, "out": {"o": "seg:filt/two"}},
            {"id": "c", "op": "logical.segop", "params": {"operation": "union"}, "in": {"i": ["seg:filt/*"]}, "out": {"o": "seg:comb/c"}},
        ]
    )
    rp = plan(spec)
    assert _dep_ids(rp, "c") == {"a", "b"}


def test_same_uri_overwrite_requires_after():
    # two nodes writing the identical URI, consumer reads it, but no `after` -> ambiguous -> error
    nodes = [
        {"id": "w1", "op": "core.cp", "in": {"src": "x:u/s"}, "out": {"dst": "z:t/z"}},
        {"id": "w2", "op": "logical.segop", "params": {"operation": "difference"}, "in": {"i": ["a:u/s", "b:u/s"]}, "out": {"o": "z:t/z"}},
        {"id": "r", "op": "logical.segop", "params": {"operation": "difference"}, "in": {"i": ["z:t/z", "q:u/s"]}, "out": {"o": "out:t/r"}},
    ]
    with pytest.raises(PlanError):
        plan(_spec(nodes))


def test_same_uri_overwrite_with_after_ok():
    nodes = [
        {"id": "w1", "op": "core.cp", "in": {"src": "x:u/s"}, "out": {"dst": "z:t/z"}},
        {"id": "w2", "op": "logical.segop", "params": {"operation": "difference"}, "after": ["w1"], "in": {"i": ["a:u/s", "b:u/s"]}, "out": {"o": "z:t/z"}},
        {"id": "r", "op": "logical.segop", "params": {"operation": "difference"}, "in": {"i": ["z:t/z", "q:u/s"]}, "out": {"o": "out:t/r"}},
    ]
    rp = plan(_spec(nodes))
    assert _dep_ids(rp, "r") == {"w2"}  # consumer depends on the last writer
    assert _dep_ids(rp, "w2") == {"w1"}  # last writer depends on the earlier one via `after`


def test_unknown_op_raises():
    with pytest.raises(PlanError):
        plan(_spec([{"id": "a", "op": "logical.nope", "in": {}, "out": {}}]))


# ---------------------------------------------------------------------------
# voxel-spacing axis
# ---------------------------------------------------------------------------


class _StubProject:
    def __init__(self, present):
        self._present = sorted(present)

    def voxel_spacings(self, run):
        return list(self._present)

    def all_voxel_spacings(self):
        return list(self._present)


def _discover_spec(match=None, tol=0.0):
    return _spec([], voxel_spacings=DiscoverSpec(discover=True, match=match, tol=tol))


def test_vs_discover_binds_true_float_within_tol():
    spec = _discover_spec(match=10.0, tol=0.05)
    proj = _StubProject([10.006, 20.012])
    assert resolve_vs_axis(spec, "run1", proj) == [10.006]


def test_vs_discover_zero_tol_raises_on_ugly_spacing():
    spec = _discover_spec(match=10.0, tol=0.0)
    proj = _StubProject([10.006, 20.012])
    with pytest.raises(AxisError):
        resolve_vs_axis(spec, "run1", proj)


def test_vs_discover_no_match_raises():
    spec = _discover_spec(match=10.0, tol=0.001)
    proj = _StubProject([20.012])
    with pytest.raises(AxisError):
        resolve_vs_axis(spec, "run1", proj)


def test_vs_discover_ambiguous_raises():
    spec = _discover_spec(match=10.0, tol=0.5)
    proj = _StubProject([9.8, 10.2])  # both within 0.5
    with pytest.raises(AxisError):
        resolve_vs_axis(spec, "run1", proj)


def test_vs_explicit_list_sweeps():
    spec = _spec([], voxel_spacings=[10.006, 20.012])
    assert resolve_vs_axis(spec, "run1", None) == [10.006, 20.012]


def test_vs_discover_requires_project():
    spec = _discover_spec(match=10.0, tol=0.1)
    with pytest.raises(AxisError):
        resolve_vs_axis(spec, "run1", None)


# ---------------------------------------------------------------------------
# symmetric edge overlap (M3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("membrane:temp/x@20", "membrane:temp/x@20", True),
        ("*:split/final@20", "microtubules:split/final@20", True),  # wildcard producer -> concrete consumer
        ("microtubules:split/final@20", "*:split/final@20", True),  # symmetric
        ("cytoplasm:filtcyto/final@20", "*:filt/final@20", False),  # user differs
        ("membrane:filt/final@20", "*:filt/final@20", True),
        ("cytoplasm:expsplit/final@20", "*:expsplit/final@20", True),
        ("membrane:filt/final@10.006", "*:filt/final@10.012", False),  # vs separates lists
    ],
)
def test_patterns_overlap(a, b, expected):
    from copick_dag.uris import patterns_overlap

    assert patterns_overlap(a, b) is expected


def test_wildcard_producer_never_overwrite():
    # a wildcard producer matching a wildcard consumer must be a fan-in edge, not an overwrite
    from copick_dag.plan.concrete import ConcreteTask
    from copick_dag.planner.edges import build_producer_index, find_producers

    wild = ConcreteTask("F", "process.split", "F", "r", 20.0, outputs={"o": "*:expsplit/final@20"})
    conc = ConcreteTask("E", "logical.segop", "E", "r", 20.0, outputs={"o": "cytoplasm:expsplit/final@20"})
    consumer = ConcreteTask("G", "process.combine", "G", "r", 20.0, inputs={"i": "*:expsplit/final@20"})
    idx = build_producer_index([wild, conc, consumer])
    exact, glob = find_producers("*:expsplit/final@20", idx, exclude=consumer)
    assert exact == []  # neither producer is a concrete-identical overwrite
    assert {t.task_id for t in glob} == {"F", "E"}  # combine depends on BOTH
