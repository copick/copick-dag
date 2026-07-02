"""Cross-run barrier adapters for the in-memory backend.

Barriers run after every per-run subgraph has completed and been written through to zarr, so they
read durable objects via ``root`` (per-run caches are gone by now). ``seg-stats`` is a MAP over the
barrier's ``run_scope`` (per-run component analysis) followed by a single REDUCE that writes one
CSV/PDF file.
"""

from __future__ import annotations

import os

from copick_dag.plan.concrete import ConcreteTask
from copick_dag.registry import get_op


class BarrierError(RuntimeError):
    pass


def exec_barrier(root, task: ConcreteTask) -> None:
    op = get_op(task.op_id)
    if op.op_id == "process.seg-stats":
        _exec_seg_stats(root, task)
    else:
        raise BarrierError(f"no in-memory barrier adapter for op {op.op_id!r}")


def _exec_seg_stats(root, task: ConcreteTask) -> None:
    from copick_utils.process.seg_stats import export_stats_csv, export_stats_plot, seg_stats_batch

    input_uri = task.inputs["i"]
    run_names = task.run_scope
    output_path = task.params["output_path"]
    fmt = task.params.get("output_format", "csv")
    connectivity = task.params.get("connectivity", "all")
    # match the copick CLI default (--include-background/--no-include-background defaults to True)
    include_background = bool(task.params.get("include_background", True))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results = seg_stats_batch(
        root=root,
        input_uri=input_uri,
        connectivity=connectivity,
        include_background=include_background,
        run_names=run_names,
        workers=1,
    )
    if fmt == "plot":
        export_stats_plot(results, output_path, root=root)
    else:
        export_stats_csv(results, output_path)
