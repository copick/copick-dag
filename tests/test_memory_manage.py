"""Durability and decoded-copy contracts for the memory backend."""

from __future__ import annotations

import numpy as np
import pytest

from copick_dag.backends.memory import run_memory
from copick_dag.backends.memory.ops import ExecError, _remove, exec_task
from copick_dag.backends.memory.overlay import CacheSeg, RunCache
from copick_dag.plan.concrete import ConcreteTask, ResolvedPlan
from fixtures import make_fixture

RUN = "run1"
SOURCE = "cell:napari/manual-scaled@20.0"


def _task(op_id: str, *, src: str | None = None, dst: str | None = None, overwrite: bool = False) -> ConcreteTask:
    kind = op_id.removeprefix("core.").split("-", 1)[0]
    inputs = {"i": src} if kind == "rm" else {"src": src}
    outputs = {} if kind == "rm" else {"dst": dst}
    return ConcreteTask(
        task_id=f"{kind}::{RUN}",
        op_id=op_id,
        node_id=kind,
        run=RUN,
        voxel_spacing=20.0,
        params={"overwrite": overwrite},
        inputs=inputs,
        outputs=outputs,
    )


def _segmentation(config: str, uri: str):
    import copick
    from copick.util.uri import resolve_copick_objects

    matches = resolve_copick_objects(uri, copick.from_file(config), "segmentation", RUN)
    return matches[0] if matches else None


def test_cache_records_only_after_durable_write():
    class FailingSeg:
        def from_numpy(self, data):
            raise OSError("durable write failed")

    cache = RunCache()
    proxy = CacheSeg(FailingSeg(), "cell:test/fail@20.0", cache)
    with pytest.raises(OSError, match="durable write failed"):
        proxy.from_numpy(np.ones((2, 2, 2), dtype=np.uint8))
    assert cache.arrays == {}
    assert cache.writes == 0


def test_copy_honors_overwrite_false(tmp_path):
    import copick

    config = make_fixture(str(tmp_path / "project"), [RUN])
    root = copick.from_file(config)
    existing = root.get_run(RUN).new_segmentation(
        name="cell",
        user_id="copy",
        session_id="result",
        is_multilabel=False,
        voxel_size=20.0,
        exist_ok=True,
    )
    sentinel = np.full((24, 24, 24), 7, dtype=np.uint8)
    existing.from_numpy(sentinel)

    plan = ResolvedPlan(
        config=config,
        tasks=[_task("core.cp", src=SOURCE, dst="cell:copy/result@20.0", overwrite=False)],
    )
    with pytest.raises(ValueError):
        run_memory(plan, root=root)
    assert np.array_equal(_segmentation(config, "cell:copy/result@20.0").numpy(), sentinel)


def test_copy_overwrite_and_move_are_decoded_equivalent(tmp_path):
    import copick

    config = make_fixture(str(tmp_path / "project"), [RUN])
    root = copick.from_file(config)
    source = _segmentation(config, SOURCE).numpy()
    existing = root.get_run(RUN).new_segmentation(
        name="cell",
        user_id="copy",
        session_id="result",
        is_multilabel=False,
        voxel_size=20.0,
        exist_ok=True,
    )
    existing.from_numpy(np.zeros_like(source))

    copy = _task("core.cp", src=SOURCE, dst="cell:copy/result@20.0", overwrite=True)
    move = _task("core.mv", src="cell:copy/result@20.0", dst="cell:moved/result@20.0")
    move.task_id = "mv::run1"
    move.deps = [copy.task_id]
    stats = run_memory(ResolvedPlan(config=config, tasks=[copy, move]), root=root)

    assert stats.total_hits >= 1
    assert _segmentation(config, "cell:copy/result@20.0") is None
    assert np.array_equal(_segmentation(config, "cell:moved/result@20.0").numpy(), source)


def test_remove_structured_errors_are_task_failures(monkeypatch):
    import copick.ops.manage

    monkeypatch.setattr(
        copick.ops.manage,
        "remove_copick_objects_batch",
        lambda **kwargs: {RUN: {"deleted": 0, "objects": [], "errors": ["backend refused deletion"]}},
    )
    with pytest.raises(ExecError, match="backend refused deletion"):
        _remove(object(), SOURCE, RUN)


def test_non_segmentation_manage_is_rejected_before_resolution():
    task = _task("core.cp-mesh", src="cell:a/b", dst="cell:c/d")
    with pytest.raises(ExecError, match="does not support cp for 'mesh'"):
        exec_task(None, None, task, RunCache())
