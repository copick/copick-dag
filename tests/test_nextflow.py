"""M4: structural tests for the Nextflow (emit-only) backend.

Correctness is anchored to *command-equivalence with the bash backend*: every emitted process runs
exactly `copick ` + `task_argv(task)`, which is already proven voxel-identical to the memory backend
(M3). The graph is planned offline (no copick project needed) from the example_3 per-run spec.
"""

import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from copick_dag.backends.bash import render_bash
from copick_dag.backends.bash.emitter import task_argv
from copick_dag.backends.memory import run_memory
from copick_dag.backends.nextflow import render_nextflow, render_nextflow_config, render_subworkflow
from copick_dag.backends.nextflow.emitter import _name_map
from copick_dag.ir.model import GraphNode, GraphSpec
from copick_dag.planner import ProjectView, plan
from fixtures import M3A_RUNS, M3A_VS, make_m3a_csv, make_m3a_fixture, read_segmentations

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


def test_render_subworkflow_composable(tmp_path):
    """render_subworkflow emits a named take/main/emit workflow a host pipeline can include+call."""
    rp = _plan(tmp_path)
    wf = render_subworkflow(rp, "COPICK_DAG_CLEANUP", label="copick_cpu")

    # A named, composable sub-workflow -- NOT a standalone top-level `workflow {}`.
    assert "workflow COPICK_DAG_CLEANUP {" in wf
    assert "  take:\n    gate" in wf
    assert "  main:" in wf
    assert "  emit:\n    done = " in wf
    assert "Channel.value('start')" not in wf  # roots gate on the take input, not a local start

    # Per-process label (host config maps it to a container/resources) + per-DAG name prefix.
    assert "label 'copick_cpu'" in wf
    assert wf.count("process COPICK_DAG_CLEANUP_p_") == len(rp.tasks)  # one process per task

    # Same commands as the standalone/bash backends (command-equivalence preserved).
    for t in rp.tasks:
        assert _cmd(t, rp.config) in wf

    # Root tasks (no deps) gate on the `gate` take-input; non-roots on their dep channels.
    names = _name_map(rp.topo_order())
    roots = [t for t in rp.tasks if not t.deps]
    assert roots
    for t in roots:
        assert _wf_line(wf, f"c_{names[t.task_id]}").endswith("(gate).first()")


def test_render_subworkflow_unique_prefix_across_dags(tmp_path):
    """Two DAGs rendered into one module get disjoint process names (no collisions)."""
    rp = _plan(tmp_path)
    a = render_subworkflow(rp, "COPICK_DAG_A", label="copick_cpu")
    b = render_subworkflow(rp, "COPICK_DAG_B", label="copick_cpu")
    procs = lambda s: {ln.split()[1] for ln in s.splitlines() if ln.startswith("process ")}
    assert procs(a).isdisjoint(procs(b))
    assert "workflow COPICK_DAG_A {" in a and "workflow COPICK_DAG_B {" in b


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


# ---------------------------------------------------------------------------
# Real-execution tests: actually run the emitted workflow (local executor) and diff the resulting
# zarr against the already-proven bash/memory backends. These close the "emit-only" gap (M4).
# ---------------------------------------------------------------------------

requires_nextflow = pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow not installed")


def _plan_with_project(config, csv):
    """Plan the example_3 per-run chain against a real copick project (run-set enumerated)."""
    spec = GraphSpec.from_yaml(GRAPH)
    spec.project.config = config
    spec.sweep.over[0]["csv"] = csv
    spec.sweep.over[0]["runs"] = M3A_RUNS
    return plan(spec, ProjectView.open(config))


def _emit_nf(rp, nfdir, extra_config=""):
    os.makedirs(nfdir, exist_ok=True)
    with open(os.path.join(nfdir, "pipeline.nf"), "w") as fh:
        fh.write(render_nextflow(rp))
    with open(os.path.join(nfdir, "nextflow.config"), "w") as fh:
        fh.write(render_nextflow_config() + extra_config)


def _run_nf(nfdir, profile="standard", resume=False, extra=()):
    cmd = ["nextflow", "run", "pipeline.nf", "-profile", profile, "-ansi-log", "false"]
    if resume:
        cmd.append("-resume")
    cmd += list(extra)
    return subprocess.run(cmd, cwd=nfdir, capture_output=True, text=True)


def _assert_finals_equal(cfg_ref, cfg_nf):
    """Assert final2 (with cytoplasm) and final2nocyto are voxel-identical across two projects."""
    for user in ("final2", "final2nocyto"):
        fa = read_segmentations(cfg_ref, M3A_RUNS, user, "final", M3A_VS)
        fb = read_segmentations(cfg_nf, M3A_RUNS, user, "final", M3A_VS)
        assert set(fa) == set(fb) and len(fa) == len(M3A_RUNS), f"{user}: {sorted(fa)} vs {sorted(fb)}"
        for k in fa:
            assert np.array_equal(fa[k], fb[k]), f"{user} {k} differs between backends"


@pytest.mark.slow
@requires_nextflow
def test_nextflow_local_equivalence(tmp_path):
    """Test 1 (flagship): nextflow local executor == bash == memory, voxel-for-voxel.

    Three identical fixtures; bash on one, memory on another, nextflow (local, default maxForks)
    on the third. The nextflow run exercises *concurrent* same-run writes (different objects), a
    pattern bash/memory never hit -- so this doubles as the parallel half of the maxForks check.
    """
    csv = make_m3a_csv(str(tmp_path / "excl.csv"))
    cfg_bash = make_m3a_fixture(str(tmp_path / "bash"))
    cfg_mem = make_m3a_fixture(str(tmp_path / "mem"))
    cfg_nf = make_m3a_fixture(str(tmp_path / "nf"))

    proc = subprocess.run(["bash", "-c", render_bash(_plan_with_project(cfg_bash, csv))],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]

    stats = run_memory(_plan_with_project(cfg_mem, csv), workers=1)
    assert not [e for e in stats.errors if e.startswith("[ERROR]")], stats.errors

    nfdir = str(tmp_path / "nfwork")
    _emit_nf(_plan_with_project(cfg_nf, csv), nfdir)
    r = _run_nf(nfdir)
    assert r.returncode == 0, r.stdout[-2000:] + "\n" + r.stderr[-3000:]

    _assert_finals_equal(cfg_bash, cfg_nf)  # nextflow == bash
    _assert_finals_equal(cfg_mem, cfg_nf)   # nextflow == memory (three-way)


@pytest.mark.slow
@requires_nextflow
def test_nextflow_serial_equals_parallel(tmp_path):
    """Test 2: forcing fully-serial execution yields the same outputs as bash.

    NOTE: ``process.maxForks = 1`` does *not* serialize this workflow -- the emitter renders one
    process per task, so each process already has a single instance and maxForks (a per-process
    cap) is a no-op across distinct processes. True global serialization needs
    ``executor.queueSize = 1``; we assert equivalence under that. Combined with the default-parallel
    run in test 1, this shows copick's concurrent same-run writes are race-free (parallel == serial).
    """
    csv = make_m3a_csv(str(tmp_path / "excl.csv"))
    cfg_bash = make_m3a_fixture(str(tmp_path / "bash"))
    cfg_nf = make_m3a_fixture(str(tmp_path / "nf"))

    proc = subprocess.run(["bash", "-c", render_bash(_plan_with_project(cfg_bash, csv))],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]

    nfdir = str(tmp_path / "nfwork")
    _emit_nf(_plan_with_project(cfg_nf, csv), nfdir, extra_config="\nexecutor.queueSize = 1\n")
    r = _run_nf(nfdir)
    assert r.returncode == 0, r.stdout[-2000:] + "\n" + r.stderr[-3000:]

    _assert_finals_equal(cfg_bash, cfg_nf)


@pytest.mark.slow
@requires_nextflow
def test_nextflow_resume_all_cached(tmp_path):
    """Test 3: ``-resume`` against an unchanged project re-runs nothing (crash-recovery only).

    Run once, then again with ``-resume`` in the same work dir; assert the second run marks every
    task CACHED (via the trace file) and leaves outputs bit-identical. (Nextflow is blind to the
    external zarr store, so a *changed* upstream would not invalidate downstreams -- a documented
    limitation; this only exercises crash recovery.)
    """
    csv = make_m3a_csv(str(tmp_path / "excl.csv"))
    cfg_nf = make_m3a_fixture(str(tmp_path / "nf"))
    rp = _plan_with_project(cfg_nf, csv)
    n_tasks = len(rp.tasks)

    nfdir = str(tmp_path / "nfwork")
    _emit_nf(rp, nfdir)
    r1 = _run_nf(nfdir, extra=("-with-trace", "trace1.txt"))
    assert r1.returncode == 0, r1.stdout[-2000:] + "\n" + r1.stderr[-3000:]
    first = read_segmentations(cfg_nf, M3A_RUNS, "final2", "final", M3A_VS)

    r2 = _run_nf(nfdir, resume=True, extra=("-with-trace", "trace2.txt"))
    assert r2.returncode == 0, r2.stdout[-2000:] + "\n" + r2.stderr[-3000:]

    # every task cached on resume
    with open(os.path.join(nfdir, "trace2.txt")) as fh:
        rows = [ln.split("\t") for ln in fh.read().splitlines()[1:] if ln.strip()]
    header = "status"
    with open(os.path.join(nfdir, "trace2.txt")) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
    si = cols.index(header)
    statuses = [row[si] for row in rows]
    assert len(statuses) == n_tasks, f"trace has {len(statuses)} rows, expected {n_tasks}"
    assert all(s == "CACHED" for s in statuses), f"non-cached tasks on resume: {statuses}"

    # outputs unchanged
    second = read_segmentations(cfg_nf, M3A_RUNS, "final2", "final", M3A_VS)
    assert set(first) == set(second)
    for k in first:
        assert np.array_equal(first[k], second[k]), f"{k} changed after resume"


@pytest.mark.slow
@requires_nextflow
@pytest.mark.skipif(shutil.which("sbatch") is None, reason="no SLURM (sbatch not found)")
@pytest.mark.skipif(
    not os.environ.get("COPICK_DAG_SLURM_WORKDIR"),
    reason="set COPICK_DAG_SLURM_WORKDIR to a SHARED (cluster-visible) path to run the SLURM test",
)
def test_nextflow_slurm_profile():
    """Test 4: run the workflow through the SLURM executor; assert voxel-identical to bash.

    SLURM jobs run on compute nodes, so BOTH the nextflow work dir and the copick overlay store must
    live on a cluster-shared filesystem -- NOT pytest's node-local ``tmp_path``. Point
    ``COPICK_DAG_SLURM_WORKDIR`` at a shared dir to enable this test. Queue/account come from
    ``COPICK_DAG_SLURM_QUEUE`` (default ``cpu``) and ``COPICK_DAG_SLURM_CLUSTEROPTS`` (extra sbatch
    flags, e.g. ``--account=... --qos=...``).
    """
    import tempfile

    base = tempfile.mkdtemp(prefix="cgnf_slurm_", dir=os.environ["COPICK_DAG_SLURM_WORKDIR"])
    queue = os.environ.get("COPICK_DAG_SLURM_QUEUE", "cpu")
    clusteropts = os.environ.get("COPICK_DAG_SLURM_CLUSTEROPTS", "")

    csv = make_m3a_csv(os.path.join(base, "excl.csv"))
    cfg_bash = make_m3a_fixture(os.path.join(base, "bash"))
    cfg_nf = make_m3a_fixture(os.path.join(base, "nf"))

    proc = subprocess.run(["bash", "-c", render_bash(_plan_with_project(cfg_bash, csv))],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]

    nfdir = os.path.join(base, "nfwork")
    _emit_nf(_plan_with_project(cfg_nf, csv), nfdir)
    extra = ["--queue", queue]
    if clusteropts:
        extra += ["--clusterOptions", clusteropts]
    r = _run_nf(nfdir, profile="slurm", extra=extra)
    assert r.returncode == 0, r.stdout[-2000:] + "\n" + r.stderr[-3000:]

    _assert_finals_equal(cfg_bash, cfg_nf)
