"""M3 (dynamic frontier): reproduce the per-run example_3 chain on both backends, voxel-identical,
with data-dependent split fan-out, glob fan-in, CSV expansion, and microtubules content-routing."""

import os
import subprocess

import numpy as np
import pytest

from copick_dag.backends.bash import render_bash
from copick_dag.backends.memory import run_memory
from copick_dag.ir.model import GraphSpec
from copick_dag.planner import ProjectView, plan
from fixtures import M3A_RUNS, M3A_VS, make_m3a_csv, make_m3a_fixture, read_segmentations

GRAPH = os.path.join(os.path.dirname(__file__), "data", "example_3_perrun.yaml")


def _load(config, csv):
    spec = GraphSpec.from_yaml(GRAPH)
    spec.project.config = config
    spec.sweep.over[0]["csv"] = csv
    spec.sweep.over[0]["runs"] = M3A_RUNS
    return spec


def test_m3a_plan_structure(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "exclusion.csv"))
    cfg = make_m3a_fixture(str(tmp_path / "proj"))
    rp = plan(_load(cfg, csv), ProjectView.open(cfg))
    by_node = {t.node_id: t for t in rp.tasks if t.run == "run_mt"}
    # coalesce lowered to two ordered writers of membrane:filt/final
    assert "thick_mem__L1__coal0" in by_node and "thick_mem__L1__coal1" in by_node
    # final2 (combine over *:expsplit) depends on BOTH E (cyto_exp) and F (split_expand)
    final2 = by_node["final2__L1"]
    dep_nodes = {rp.by_id()[d].node_id for d in final2.deps}
    assert "cyto_exp__L1" in dep_nodes and "split_expand__L1" in dep_nodes
    # CSV filter nodes are optional (a label may be absent from a run)
    assert by_node["filt__membrane__L1"].optional


@pytest.mark.slow
def test_m3a_bash_and_memory_voxel_identical(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "exclusion.csv"))
    cfg_bash = make_m3a_fixture(str(tmp_path / "bash"))
    cfg_mem = make_m3a_fixture(str(tmp_path / "mem"))

    # bash
    pa = plan(_load(cfg_bash, csv), ProjectView.open(cfg_bash))
    script = tmp_path / "run.sh"
    script.write_text(render_bash(pa))
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]

    # memory
    stats = run_memory(plan(_load(cfg_mem, csv), ProjectView.open(cfg_mem)), workers=1)
    assert not [e for e in stats.errors if e.startswith("[ERROR]")], stats.errors

    # final2 (with cytoplasm) and final2nocyto voxel-identical across backends, both runs
    for user in ("final2", "final2nocyto"):
        fa = read_segmentations(cfg_bash, M3A_RUNS, user, "final", M3A_VS)
        fb = read_segmentations(cfg_mem, M3A_RUNS, user, "final", M3A_VS)
        assert set(fa) == set(fb) and len(fa) == 2
        for k in fa:
            assert np.array_equal(fa[k], fb[k]), f"{user} {k} differs"

    # routing: the MT-present run and the MT-absent run diverge on membrane:filt/final
    mf = read_segmentations(cfg_mem, M3A_RUNS, "filt", "final", M3A_VS)
    assert not np.array_equal(mf["run_mt|membrane"], mf["run_nomt|membrane"])


def _stats_graph(config, csv, stats_path):
    from copick_dag.ir.model import GraphNode

    spec = _load(config, csv)
    spec.nodes.append(
        GraphNode(
            id="stats",
            op="process.seg-stats",
            inputs={"i": "multilabel:final2/final"},
            params={"output_format": "csv", "output_path": stats_path},
        ),
    )
    return spec


def test_m3b_seg_stats_is_a_barrier(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "exclusion.csv"))
    cfg = make_m3a_fixture(str(tmp_path / "proj"))
    rp = plan(_stats_graph(cfg, csv, str(tmp_path / "s.csv")), ProjectView.open(cfg))
    bt = next(t for t in rp.tasks if t.node_id == "stats__L1")
    assert bt.scope == "barrier" and bt.run is None
    assert sorted(bt.run_scope) == sorted(M3A_RUNS)
    # depends on the per-run final2 producers of both runs
    dep_runs = {rp.by_id()[d].run for d in bt.deps}
    assert dep_runs == set(M3A_RUNS)


@pytest.mark.slow
def test_m3b_seg_stats_bash_and_memory_match(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "exclusion.csv"))
    cfg_bash = make_m3a_fixture(str(tmp_path / "bash"))
    cfg_mem = make_m3a_fixture(str(tmp_path / "mem"))
    bstats = str(tmp_path / "bash_stats.csv")
    mstats = str(tmp_path / "mem_stats.csv")

    pa = plan(_stats_graph(cfg_bash, csv, bstats), ProjectView.open(cfg_bash))
    script = tmp_path / "run.sh"
    script.write_text(render_bash(pa))
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]

    stats = run_memory(plan(_stats_graph(cfg_mem, csv, mstats), ProjectView.open(cfg_mem)), workers=1)
    assert not [e for e in stats.errors if e.startswith("[ERROR]")], stats.errors

    assert os.path.exists(bstats) and os.path.exists(mstats)
    with open(bstats) as fh:
        brows = sorted(fh.read().splitlines())
    with open(mstats) as fh:
        mrows = sorted(fh.read().splitlines())
    assert brows == mrows and len(brows) > 1  # header + at least one component row
