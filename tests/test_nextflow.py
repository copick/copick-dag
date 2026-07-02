"""M4: structural tests for the Nextflow (emit-only) backend.

Correctness is anchored to *command-equivalence with the bash backend*: every emitted process runs
exactly `copick ` + `task_argv(task)`, which is already proven voxel-identical to the memory backend
(M3). The graph is planned offline (no copick project needed) from the example_3 per-run spec.
"""

import os
import re

import pytest

from copick_dag.backends.bash.emitter import task_argv
from copick_dag.backends.nextflow import render_nextflow, render_nextflow_config
from copick_dag.backends.nextflow.emitter import _name_map
from copick_dag.ir.model import GraphNode, GraphSpec
from copick_dag.planner import plan
from fixtures import M3A_RUNS, make_m3a_csv

GRAPH = os.path.join(os.path.dirname(__file__), "data", "example_3_perrun.yaml")


def _sq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _cmd(task, config) -> str:
    return "copick " + " ".join(_sq(a) for a in task_argv(task, config))


def _plan(tmp_path, with_stats=False):
    csv = make_m3a_csv(str(tmp_path / "exclusion.csv"))
    spec = GraphSpec.from_yaml(GRAPH)
    spec.project.config = "/abs/config.json"
    spec.sweep.over[0]["csv"] = csv
    spec.sweep.over[0]["runs"] = M3A_RUNS
    if with_stats:
        spec.nodes.append(
            GraphNode(
                id="stats",
                op="process.seg-stats",
                inputs={"i": "multilabel:final2/final"},
                params={"output_format": "csv", "output_path": "/tmp/stats.csv"},
            ),
        )
    return plan(spec)  # offline (sweep row supplies runs + vs)


def _wf_line(nf, cvar):
    return next(ln for ln in nf.splitlines() if ln.strip().startswith(f"{cvar} = "))


def test_command_equivalence_to_bash(tmp_path):
    rp = _plan(tmp_path)
    nf = render_nextflow(rp)
    assert "nextflow.enable.dsl = 2" in nf
    for t in rp.tasks:
        assert _cmd(t, rp.config) in nf, f"missing command for {t.task_id}"


def test_optional_has_or_true_and_always_emits_token(tmp_path):
    rp = _plan(tmp_path)
    nf = render_nextflow(rp)
    # every task (optional or not) emits its done token
    for t in rp.tasks:
        assert f"val '{t.task_id}'" in nf
    opt = [t for t in rp.tasks if t.optional]
    nonopt = [t for t in rp.tasks if not t.optional]
    assert opt and nonopt
    for t in opt:  # optional -> || true
        assert _cmd(t, rp.config) + " || true" in nf
    for t in nonopt:  # non-optional -> no || true
        assert _cmd(t, rp.config) + " || true" not in nf


def test_dependency_wiring(tmp_path):
    rp = _plan(tmp_path)
    nf = render_nextflow(rp)
    names = _name_map(rp.topo_order())
    for t in rp.tasks:
        wl = _wf_line(nf, f"c_{names[t.task_id]}")
        if not t.deps:
            assert "(start)" in wl
        elif len(t.deps) == 1:
            assert f"(c_{names[t.deps[0]]})" in wl
        else:
            assert ".mix(" in wl and ".collect()" in wl
            for d in t.deps:
                assert f"c_{names[d]}" in wl


def test_barrier_multi_r_and_collect(tmp_path):
    rp = _plan(tmp_path, with_stats=True)
    nf = render_nextflow(rp)
    bt = next(t for t in rp.tasks if t.scope == "barrier")
    cmd = _cmd(bt, rp.config)
    assert cmd.count("'-r'") == len(M3A_RUNS) == 2  # aggregates over the row's runs
    assert cmd in nf
    names = _name_map(rp.topo_order())
    assert ".collect()" in _wf_line(nf, f"c_{names[bt.task_id]}")


def test_process_names_unique_and_valid(tmp_path):
    rp = _plan(tmp_path, with_stats=True)
    nf = render_nextflow(rp)
    procs = re.findall(r"^process (\S+) \{", nf, re.M)
    assert len(procs) == len(rp.tasks)
    assert len(procs) == len(set(procs))  # unique
    for p in procs:
        assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p), p


def test_config_has_local_and_slurm_profiles():
    cfg = render_nextflow_config()
    assert "process.executor = 'local'" in cfg
    assert "'slurm'" in cfg and "process.queue" in cfg


@pytest.mark.slow
def test_nextflow_lint_if_available(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("nextflow"):
        pytest.skip("nextflow not installed (emit-only scope)")
    rp = _plan(tmp_path, with_stats=True)
    nf_path = tmp_path / "pipeline.nf"
    nf_path.write_text(render_nextflow(rp))
    (tmp_path / "nextflow.config").write_text(render_nextflow_config())
    proc = subprocess.run(
        ["nextflow", "run", str(nf_path), "-stub-run", "-preview"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
