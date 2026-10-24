"""OME-Zarr 0.4/v2 and 0.5/v3 orchestration acceptance tests."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import zarr

from copick_dag.backends.bash import render_bash
from copick_dag.backends.memory import run_memory
from copick_dag.ir.model import GraphSpec
from copick_dag.plan.concrete import ConcreteTask, ResolvedPlan
from copick_dag.planner import ProjectView, plan
from fixtures import make_fixture

RUN = "run1"
SOURCE = "cell:napari/manual-scaled@20.0"
SECOND_SOURCE = "cytosole:napari/manual-scaled@20.0"
OUTPUT = "cell:migrated/final@20.0"
AXES = [
    {"name": "z", "type": "space", "unit": "angstrom"},
    {"name": "y", "type": "space", "unit": "angstrom"},
    {"name": "x", "type": "space", "unit": "angstrom"},
]


def _resolve(config: str, uri: str):
    import copick
    from copick.util.uri import resolve_copick_objects

    matches = resolve_copick_objects(uri, copick.from_file(config), "segmentation", RUN)
    assert len(matches) == 1
    return matches[0]


def _multiscale(path: str, version: str) -> dict:
    return {
        "version": version,
        "name": "/",
        "axes": AXES,
        "datasets": [
            {
                "path": path,
                "coordinateTransformations": [{"type": "scale", "scale": [20.0, 20.0, 20.0]}],
            },
        ],
    }


def _rewrite_source(config: str, layout: str) -> np.ndarray:
    source = _resolve(config, SOURCE)
    data = source.numpy().copy()
    if layout == "canonical-v3":
        return data
    if layout in {"v2-numeric", "v2-s0"}:
        path = "0" if layout == "v2-numeric" else "s0"
        group = zarr.open_group(source.zarr(), mode="w", zarr_format=2)
        group.create_array(path, data=data, chunks=(8, 8, 8), compressor=None)
        group.attrs["multiscales"] = [_multiscale(path, "0.4")]
        return data
    if layout == "noncanonical-v3":
        group = zarr.open_group(source.zarr(), mode="w", zarr_format=3)
        group.create_array("s0", data=data, chunks=(7, 8, 9), dimension_names=("z", "y", "x"))
        group.attrs["ome"] = {"version": "0.5", "multiscales": [_multiscale("s0", "0.5")]}
        return data
    raise AssertionError(layout)


def _semantic_graph(config: str) -> GraphSpec:
    return GraphSpec.from_dict(
        {
            "version": 1,
            "project": {"config": config},
            "runs": [RUN],
            "nodes": [
                {
                    "id": "union",
                    "op": "logical.segop",
                    "params": {"operation": "union"},
                    "in": {"i": [SOURCE, SECOND_SOURCE]},
                    "out": {"o": OUTPUT},
                },
            ],
        },
    )


def _assert_canonical_output(config: str) -> None:
    group = zarr.open_group(_resolve(config, OUTPUT).zarr(), mode="r")
    assert group.metadata.zarr_format == 3
    assert group.attrs["ome"]["version"] == "0.5"
    multiscale = group.attrs["ome"]["multiscales"][0]
    assert [dataset["path"] for dataset in multiscale["datasets"]] == ["0"]
    assert [axis["name"] for axis in multiscale["axes"]] == ["z", "y", "x"]
    array = group["0"]
    assert array.metadata.dimension_names == ("z", "y", "x")
    assert array.chunks == (128, 128, 128)
    assert array.shards == (128, 128, 128)
    assert type(array.metadata.chunk_key_encoding).__name__ == "V2ChunkKeyEncoding"


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["v2-numeric", "v2-s0", "canonical-v3", "noncanonical-v3"])
@pytest.mark.parametrize("backend", ["memory", "bash"])
def test_supported_source_layouts_produce_same_canonical_output(tmp_path, layout, backend):
    config = make_fixture(str(tmp_path / f"{layout}-{backend}"), [RUN])
    source = _rewrite_source(config, layout)
    expected = np.logical_or(source, _resolve(config, SECOND_SOURCE).numpy()).astype(np.uint8)
    resolved = plan(_semantic_graph(config), ProjectView.open(config))

    if backend == "memory":
        stats = run_memory(resolved, workers=1)
        assert not [error for error in stats.errors if error.startswith("[ERROR]")]
    else:
        proc = subprocess.run(["bash", "-c", render_bash(resolved)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]

    assert np.array_equal(_resolve(config, OUTPUT).numpy(), expected)
    _assert_canonical_output(config)


def test_planning_does_not_read_array_payload(monkeypatch, tmp_path):
    from copick.models import CopickSegmentation

    config = make_fixture(str(tmp_path / "planner"), [RUN])

    def fail_if_read(*args, **kwargs):
        raise AssertionError("planner read a segmentation payload")

    monkeypatch.setattr(CopickSegmentation, "numpy", fail_if_read)
    resolved = plan(_semantic_graph(config), ProjectView.open(config), validate_preconditions=True)
    assert len(resolved.tasks) == 1


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["v2-s0", "noncanonical-v3"])
@pytest.mark.parametrize("backend", ["memory", "bash"])
def test_copy_contract_is_decoded_equivalence_not_representation(tmp_path, layout, backend):
    config = make_fixture(str(tmp_path / f"copy-{layout}-{backend}"), [RUN])
    expected = _rewrite_source(config, layout)
    destination = "cell:copy/layout@20.0"
    task = ConcreteTask(
        task_id=f"copy::{RUN}",
        op_id="core.cp",
        node_id="copy",
        run=RUN,
        voxel_spacing=20.0,
        params={"overwrite": False},
        inputs={"src": SOURCE},
        outputs={"dst": destination},
    )
    resolved = ResolvedPlan(config=config, tasks=[task])

    if backend == "memory":
        run_memory(resolved, workers=1)
    else:
        proc = subprocess.run(["bash", "-c", render_bash(resolved)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]

    copied = _resolve(config, destination)
    assert (copied.name, copied.user_id, copied.session_id, copied.voxel_size) == ("cell", "copy", "layout", 20.0)
    assert np.array_equal(copied.numpy(), expected)
    # Deliberately do not assert the destination's format, raw keys, codecs, chunks, or shards.
