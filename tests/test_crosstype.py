"""Phase 0 engine tests: the object-type generalization proven with *synthetic* OpSpecs, before any
real cross-type op exists. Also covers the `?multilabel` URI round-trip fix. These lock the
seg-preserving refactor: cross-type edges never false-match, vs-free ops don't ride the vs sweep, and
vs-transform ops keep distinct in/out spacings."""

import pytest

from copick_dag.ir.model import GraphSpec
from copick_dag.planner import plan
from copick_dag.registry.overlay import OPS
from copick_dag.registry.spec import VS_URI, OpSpec, SlotSpec
from copick_dag.uris import get_vs, patterns_overlap, split_vs, strip_vs, with_vs


@pytest.fixture
def synth_ops():
    """Register throwaway ops covering mesh (vs-free), seg, and a vs-transform; remove after."""
    added = []

    def reg(op: OpSpec) -> None:
        OPS[op.op_id] = op
        added.append(op.op_id)

    reg(OpSpec(op_id="t.meshop", cli_path=("t", "meshop"), vs_style=VS_URI,
               inputs=[SlotSpec("i", object_type="mesh", flag="-i")],
               outputs=[SlotSpec("o", object_type="mesh", flag="-o")]))
    reg(OpSpec(op_id="t.segop", cli_path=("t", "segop"), vs_style=VS_URI,
               inputs=[SlotSpec("i", object_type="segmentation", flag="-i")],
               outputs=[SlotSpec("o", object_type="segmentation", flag="-o")]))
    reg(OpSpec(op_id="t.rescale", cli_path=("t", "rescale"), vs_style=VS_URI, vs_transform=True,
               inputs=[SlotSpec("i", object_type="segmentation", flag="-i")],
               outputs=[SlotSpec("o", object_type="segmentation", flag="-o")]))
    yield
    for k in added:
        OPS.pop(k, None)


def _plan_offline(nodes, voxel_spacings=None):
    d = {"version": 1, "project": {"config": "x"}, "runs": ["r1"], "nodes": nodes}
    if voxel_spacings is not None:
        d["voxel_spacings"] = voxel_spacings
    return plan(GraphSpec.from_dict(d))


def test_no_false_cross_type_match(synth_ops):
    # A writes a MESH x:u/s ; B writes a SEG x:u/s@10 ; C reads a MESH x:u/s.
    # C must depend on A (mesh) and NOT on B (seg) -- same identity string, different object type.
    rp = _plan_offline([
        {"id": "A", "op": "t.meshop", "in": {"i": "src:u/s"}, "out": {"o": "x:u/s"}},
        {"id": "B", "op": "t.segop", "in": {"i": "seed:p/q@10.0"}, "out": {"o": "x:u/s@10.0"}},
        {"id": "C", "op": "t.meshop", "in": {"i": "x:u/s"}, "out": {"o": "y:u/s"}},
    ])
    by_node = {t.node_id: t for t in rp.tasks}
    dep_nodes = {rp.by_id()[d].node_id for d in by_node["C"].deps}
    assert "A" in dep_nodes and "B" not in dep_nodes


def test_vs_free_op_builds_once_under_multi_vs(synth_ops):
    # a mesh op under a 2-value vs sweep must produce ONE task per run, not one per spacing.
    rp = _plan_offline(
        [{"id": "m", "op": "t.meshop", "in": {"i": "a:u/s"}, "out": {"o": "b:u/s"}}],
        voxel_spacings=[10.0, 20.0],
    )
    tasks = [t for t in rp.tasks if t.node_id == "m"]
    assert len(tasks) == 1 and tasks[0].voxel_spacing is None


def test_vs_transform_keeps_distinct_in_out_vs(synth_ops):
    # rescale: in@10 -> out@20, built once, with both spacings preserved on their URIs.
    rp = _plan_offline(
        [{"id": "r", "op": "t.rescale", "in": {"i": "s:a/b@10.0"}, "out": {"o": "s:resc/0@20.0"}}],
        voxel_spacings=[10.0, 20.0],
    )
    tasks = [t for t in rp.tasks if t.node_id == "r"]
    assert len(tasks) == 1
    assert tasks[0].inputs["i"] == "s:a/b@10.0"
    assert tasks[0].outputs["o"] == "s:resc/0@20.0"


def test_multilabel_query_roundtrip():
    u = "membrane:pred/1@10.0?multilabel=true"
    assert get_vs(u) == "10.0"  # not "10.0?multilabel=true"
    assert split_vs(u) == ("membrane:pred/1?multilabel=true", "10.0")
    assert strip_vs(u) == "membrane:pred/1?multilabel=true"
    # attaching a spacing to a query-only URI inserts @vs before the query
    assert with_vs("membrane:pred/1?multilabel=true", 10.0) == "membrane:pred/1@10.0?multilabel=true"


def test_multilabel_query_does_not_break_overlap():
    # two multilabel seg URIs at the same spacing still overlap (vs parsed cleanly, not corrupted)
    assert patterns_overlap(
        "membrane:pred/1@10.0?multilabel=true", "membrane:pred/1@10.0?multilabel=true", "segmentation",
    )


# ---------------------------------------------------------------------------
# Real registered ops (planning only; no execution). Mirror the phase acceptance checks.
# ---------------------------------------------------------------------------

from fixtures import make_fixture  # noqa: E402
from copick_dag.planner import ProjectView  # noqa: E402


def _plan_fixture(tmp_path, nodes, runs=("run1",), defaults=None):
    cfg = make_fixture(str(tmp_path / "p"), list(runs))
    d = {"version": 1, "project": {"config": cfg}, "nodes": nodes}
    if defaults:
        d["defaults"] = defaults
    rp = plan(GraphSpec.from_dict(d), ProjectView.open(cfg))
    return rp, {t.node_id: t for t in rp.tasks}, {t.task_id: t for t in rp.tasks}


def test_real_mesh_chain_edges_and_no_vs_multiply(tmp_path):
    rp, by_node, by_id = _plan_fixture(tmp_path, [
        {"id": "to_mesh", "op": "convert.seg2mesh", "in": {"i": "cell:napari/manual-scaled"}, "out": {"o": "cell:m/s"}},
        {"id": "merge", "op": "logical.meshop", "params": {"operation": "union"}, "in": {"i": ["cell:m/s"]}, "out": {"o": "cell:mm/s"}},
        {"id": "back", "op": "convert.mesh2seg", "params": {"tomo_type": "wbp"}, "in": {"i": "cell:mm/s"}, "out": {"o": "cell:v/0@20.0"}},
    ], defaults={"voxel_spacing": 20.0})
    assert {by_id[d].node_id for d in by_node["merge"].deps} == {"to_mesh"}
    assert {by_id[d].node_id for d in by_node["back"].deps} == {"merge"}
    assert by_node["merge"].voxel_spacing is None  # mesh op is vs-free -> not swept
    assert by_node["to_mesh"].voxel_spacing == 20.0


def test_tomogram_primary_and_reference_edges(tmp_path):
    rp, by_node, by_id = _plan_fixture(tmp_path, [
        {"id": "ds", "op": "process.downsample", "in": {"i": "wbp@10.0"}, "out": {"o": "wbp@20.0"}},
        {"id": "vb", "op": "process.validbox", "in": {"t": "wbp@20.0"}, "out": {"o": "b:vb/x"}},
        {"id": "mp", "op": "convert.mesh2picks", "params": {"sampling_type": "surface"},
         "in": {"i": "b:vb/x", "t": "wbp@20.0"}, "out": {"o": "r:mp/x"}},
    ])
    assert {by_id[d].node_id for d in by_node["vb"].deps} == {"ds"}          # tomogram producer -> tomo ref
    assert {by_id[d].node_id for d in by_node["mp"].deps} == {"vb", "ds"}    # mesh chain + tomo ref
    assert by_node["ds"].voxel_spacing == 10.0 and by_node["ds"].outputs["o"] == "wbp@20.0"


def test_rescale_vs_change_and_templated_fanin(tmp_path):
    rp, by_node, by_id = _plan_fixture(tmp_path, [
        {"id": "rs", "op": "process.rescale", "in": {"i": "cell:napari/manual-scaled@10.0"}, "out": {"o": "cell:r/0@20.0"}},
        {"id": "sep", "op": "process.separate-components", "in": {"i": "cell:r/0@20.0"}, "out": {"o": "m:sc/{instance_id}@20.0"}},
        {"id": "comb", "op": "process.combine", "in": {"i": "m:sc/*@20.0"}, "out": {"o": "ml:rc/f@20.0"}},
    ])
    assert by_node["rs"].inputs["i"] == "cell:napari/manual-scaled@10.0"
    assert by_node["rs"].outputs["o"] == "cell:r/0@20.0"  # distinct in/out spacings preserved
    assert {by_id[d].node_id for d in by_node["sep"].deps} == {"rs"}
    assert {by_id[d].node_id for d in by_node["comb"].deps} == {"sep"}  # {instance_id} -> * fan-in


def test_export_is_a_self_wiring_barrier(tmp_path):
    rp, by_node, by_id = _plan_fixture(tmp_path, [
        {"id": "seg", "op": "logical.segop", "params": {"operation": "union"},
         "in": {"i": ["membrane:data-portal/*"]}, "out": {"o": "membrane:m/f"}},
        {"id": "exp", "op": "export.segmentation", "params": {"output_format": "mrc", "output_dir": "/tmp/o"},
         "in": {"i": "membrane:m/f@20.0"}},
    ], runs=("run1", "run2"), defaults={"voxel_spacing": 20.0})
    exp = by_node["exp"]
    assert exp.scope == "barrier" and exp.run is None and sorted(exp.run_scope) == ["run1", "run2"]
    assert {by_id[d].run for d in exp.deps} == {"run1", "run2"}


def test_finalize_selects_cp_rm_by_object_type(tmp_path):
    """finalize copies/gc pick the cp/rm variant matching their declared object_type (mesh here)."""
    from copick_dag.backends.bash.emitter import task_argv

    cfg = make_fixture(str(tmp_path / "p"), ["run1"])
    graph = {
        "version": 1, "project": {"config": cfg},
        "nodes": [{"id": "seed", "op": "logical.meshop", "params": {"operation": "union"},
                   "in": {"i": ["m:a/b"]}, "out": {"o": "m:c/d"}}],
        "finalize": {"copies": [{"from": "m:c/d", "to": "m:final/0", "object_type": "mesh"}],
                     "gc": {"remove": ["m:a/b"], "object_type": "mesh"}},
    }
    rp = plan(GraphSpec.from_dict(graph), ProjectView.open(cfg))
    ops = {t.node_id: t.op_id for t in rp.tasks}
    assert ops["finalize_cp_0"] == "core.cp-mesh" and ops["gc_rm_0"] == "core.rm-mesh"
    cp = next(t for t in rp.tasks if t.node_id == "finalize_cp_0")
    assert task_argv(cp, cfg)[:2] == ["cp", "mesh"]


@pytest.mark.slow
def test_crosstype_bash_roundtrip(tmp_path):
    """Real execution (bash backend, no nextflow needed): seg2mesh -> mesh2seg round-trips a
    segmentation through a mesh and back, recovering the seed. Exercises the cross-type edges,
    tomogram grid reference, and per-slot @vs on actually-run copick commands."""
    import subprocess

    import numpy as np

    from fixtures import make_crosstype_fixture, read_meshes, read_segmentations
    from copick_dag.backends.bash import render_bash

    cfg = make_crosstype_fixture(str(tmp_path / "p"), ["run1"])
    graph = {
        "version": 1, "project": {"config": cfg}, "defaults": {"voxel_spacing": 20.0},
        "nodes": [
            {"id": "to_mesh", "op": "convert.seg2mesh", "params": {"level": 0.5, "step_size": 1},
             "in": {"i": "cell:napari/manual-scaled"}, "out": {"o": "cell:seg2mesh/surf"}},
            {"id": "to_seg", "op": "convert.mesh2seg", "params": {"tomo_type": "wbp"},
             "in": {"i": "cell:seg2mesh/surf"}, "out": {"o": "cell:mesh2seg/vol@20.0"}},
        ],
    }
    rp = plan(GraphSpec.from_dict(graph), ProjectView.open(cfg))
    proc = subprocess.run(["bash", "-c", render_bash(rp)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]

    assert "run1|cell" in read_meshes(cfg, ["run1"], "seg2mesh", "surf")
    rt = read_segmentations(cfg, ["run1"], "mesh2seg", "vol", 20.0)
    orig = read_segmentations(cfg, ["run1"], "napari", "manual-scaled", 20.0)
    assert "run1|cell" in rt
    a, b = rt["run1|cell"].astype(bool), orig["run1|cell"].astype(bool)
    assert (a & b).sum() / max(1, (a | b).sum()) > 0.3
