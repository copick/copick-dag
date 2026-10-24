"""End-to-end: the in-memory backend is voxel-identical to the bash backend on example_1."""

import os
import subprocess

import numpy as np
import pytest

from copick_dag.backends.bash import render_bash
from copick_dag.backends.memory import run_memory
from copick_dag.ir.model import GraphSpec
from copick_dag.planner import ProjectView, plan
from fixtures import make_fixture, read_final_segmentations

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "example_1.yaml")
RUNS = ["run1", "run2"]


def _plan_for(config):
    spec = GraphSpec.from_yaml(EXAMPLE)
    spec.project.config = config
    return spec, plan(spec, ProjectView.open(config))


def test_bash_render_structure(tmp_path):
    cfg = make_fixture(str(tmp_path / "proj"), RUNS)
    _, rp = _plan_for(cfg)
    script = render_bash(rp)
    # every logical op group and the core object ops must appear
    for token in [
        "copick logical segop",
        "copick logical clipseg",
        "copick logical enclosed",
        "copick cp segmentation",
        "copick mv segmentation",
        "copick rm segmentation",
    ]:
        assert token in script, token
    # flag-style ops strip @vs and pass -vs; uri-style keep @vs
    assert "-vs 20.0" in script
    assert "@20.0" in script
    # optional inclusion subtraction is tolerated
    assert "|| true" in script


def test_memory_rejects_run_agnostic_plan(tmp_path):
    # plan without a project -> run is None -> memory backend must refuse
    spec = GraphSpec.from_yaml(EXAMPLE)
    spec.project.config = "${CFG}"
    rp = plan(spec)  # offline
    with pytest.raises(ValueError):
        run_memory(rp)


@pytest.mark.slow
def test_bash_and_memory_are_voxel_identical(tmp_path):
    cfg_bash = make_fixture(str(tmp_path / "bash"), RUNS)
    cfg_mem = make_fixture(str(tmp_path / "mem"), RUNS)

    # --- bash backend ---
    _, plan_bash = _plan_for(cfg_bash)
    script = render_bash(plan_bash)
    script_path = tmp_path / "run.sh"
    script_path.write_text(script)
    proc = subprocess.run(["bash", str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]

    # --- memory backend (sequential: deterministic, single-parse, cache-served) ---
    _, plan_mem = _plan_for(cfg_mem)
    stats = run_memory(plan_mem, workers=1)
    assert not [e for e in stats.errors if e.startswith("[ERROR]")], stats.errors
    assert stats.project_parses == 1  # parsed exactly once
    assert stats.total_hits > 0  # intermediates served from memory (no zarr re-read)

    # --- compare final-training segmentations ---
    fa = read_final_segmentations(cfg_bash, RUNS)
    fb = read_final_segmentations(cfg_mem, RUNS)
    assert set(fa) == set(fb) and len(fa) == 12  # 6 objects x 2 runs
    for key in fa:
        a, b = fa[key], fb[key]
        assert a.shape == b.shape and np.array_equal(a, b), f"{key} differs"
