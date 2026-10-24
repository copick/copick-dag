"""Per-op adapters for the in-memory backend.

Each adapter resolves a task's input URIs to (cache-aware) segmentation objects and calls the
*existing* copick per-run converter with the write-through overlay run, so no array algorithm is
reimplemented here. Segmentation ``cp``/``mv`` decode and rewrite through entity APIs, while ``rm``
uses ``copick.ops.manage`` and checks its structured result.
"""

from __future__ import annotations

from typing import List, Optional

from copick_dag.backends.memory.overlay import CacheSeg, MemoryOverlayRun, RunCache, seg_uri
from copick_dag.plan.concrete import ConcreteTask
from copick_dag.registry import get_op
from copick_dag.registry.introspect import MissingOperationDependency, require_operation
from copick_dag.uris import SegURI


def _resolve_segs(overlay: MemoryOverlayRun, uri: str, cache: RunCache) -> List[CacheSeg]:
    """Resolve a (possibly wildcard) segmentation URI to cache-aware proxies in this run."""
    from copick.util.uri import get_copick_objects_by_type

    s = SegURI.parse(uri)
    vs = None
    if s.voxel_spacing not in (None, "*"):
        try:
            vs = float(s.voxel_spacing)
        except (TypeError, ValueError):
            vs = None
    objs = get_copick_objects_by_type(
        root=overlay.root,
        object_type="segmentation",
        run_name=overlay.name,
        name=s.name,
        user_id=s.user_id,
        session_id=s.session_id,
        voxel_spacing=vs,
        pattern_type=s.pattern_type,
    )
    return [CacheSeg(o, seg_uri(o.name, o.user_id, o.session_id, o.voxel_size), cache) for o in objs]


def _resolve_one(overlay: MemoryOverlayRun, uri: str, cache: RunCache) -> Optional[CacheSeg]:
    segs = _resolve_segs(overlay, uri, cache)
    return segs[0] if segs else None


def _out_params(uri: str):
    o = SegURI.parse(uri)
    return {
        "object_name": o.name,
        "user_id": o.user_id,
        "session_id": o.session_id,
        "voxel_spacing": float(o.voxel_spacing),
    }


class ExecError(RuntimeError):
    pass


def exec_task(overlay: MemoryOverlayRun, root, task: ConcreteTask, cache: RunCache) -> None:
    op = get_op(task.op_id)
    try:
        require_operation(op)
    except MissingOperationDependency as e:
        raise ExecError(str(e)) from None
    if op.op_id == "logical.segop":
        _exec_segop(overlay, task, cache)
    elif op.op_id == "logical.clipseg":
        _exec_clipseg(overlay, task, cache)
    elif op.op_id == "logical.enclosed":
        _exec_enclosed(overlay, task, cache)
    elif op.op_id == "process.split":
        _exec_split(overlay, task, cache)
    elif op.op_id == "process.combine":
        _exec_combine(overlay, task, cache)
    elif op.op_id == "process.filter-components":
        _exec_filter(overlay, task, cache)
    elif op.op_id == "process.expand-labels":
        _exec_expand(overlay, task, cache)
    elif op.op_id == "process.thickness-filter":
        _exec_thickness(overlay, task, cache)
    elif op.kind in ("cp", "mv", "rm"):
        _exec_manage(overlay, root, task, cache)
    else:
        raise ExecError(f"no in-memory adapter for op {op.op_id!r}")


def _exec_segop(overlay: MemoryOverlayRun, task: ConcreteTask, cache: RunCache) -> None:
    from copick_utils.logical.segmentation_operations import (
        segmentation_boolean_operation,
        segmentation_multi_union,
    )

    operation = task.params["operation"]
    out = _out_params(task.outputs["o"])
    i_val = task.inputs["i"]
    i_uris = i_val if isinstance(i_val, list) else [i_val]

    if operation == "union":
        segs: List[CacheSeg] = []
        for u in i_uris:
            segs.extend(_resolve_segs(overlay, u, cache))
        if not segs:
            raise ExecError(f"union: no segmentations matched {i_uris} in run {overlay.name}")
        if len(segs) == 1:
            # union of one == itself; write through unchanged
            arr = segs[0].numpy()
            out_seg = overlay.new_segmentation(
                out["object_name"],
                out["user_id"],
                out["session_id"],
                False,
                out["voxel_spacing"],
                exist_ok=True,
            )
            out_seg.from_numpy(arr)
            return
        segmentation_multi_union(
            segmentations=segs,
            run=overlay,
            object_name=out["object_name"],
            session_id=out["session_id"],
            user_id=out["user_id"],
            voxel_spacing=out["voxel_spacing"],
        )
        return

    # difference / intersection / exclusion: exactly two inputs
    if len(i_uris) != 2:
        raise ExecError(f"{operation}: expected 2 inputs, got {len(i_uris)}")
    seg1 = _resolve_one(overlay, i_uris[0], cache)
    seg2 = _resolve_one(overlay, i_uris[1], cache)
    if seg1 is None or seg2 is None:
        raise ExecError(f"{operation}: could not resolve inputs {i_uris} in run {overlay.name}")
    segmentation_boolean_operation(
        segmentation1=seg1,
        segmentation2=seg2,
        run=overlay,
        object_name=out["object_name"],
        session_id=out["session_id"],
        user_id=out["user_id"],
        operation=operation,
        voxel_spacing=out["voxel_spacing"],
    )


def _exec_clipseg(overlay: MemoryOverlayRun, task: ConcreteTask, cache: RunCache) -> None:
    from copick_utils.logical.distance_operations import limit_segmentation_by_distance

    out = _out_params(task.outputs["o"])
    in_seg = _resolve_one(overlay, task.inputs["i"], cache)
    if in_seg is None:
        raise ExecError(f"clipseg: input {task.inputs['i']} not found in run {overlay.name}")

    ref_seg = None
    if "rs" in task.inputs:
        ref_seg = _resolve_one(overlay, task.inputs["rs"], cache)
        if ref_seg is None:
            raise ExecError(f"clipseg: reference {task.inputs['rs']} not found in run {overlay.name}")
    elif "rm" in task.inputs or "rt" in task.inputs:
        raise ExecError("clipseg: mesh/tomogram references not yet supported in the in-memory backend")

    limit_segmentation_by_distance(
        segmentation=in_seg,
        run=overlay,
        object_name=out["object_name"],
        session_id=out["session_id"],
        user_id=out["user_id"],
        reference_segmentation=ref_seg,
        max_distance=float(task.params["max_distance"]),
        voxel_spacing=out["voxel_spacing"],
        invert=bool(task.params.get("invert", False)),
    )


def _exec_enclosed(overlay: MemoryOverlayRun, task: ConcreteTask, cache: RunCache) -> None:
    from copick_utils.logical.enclosed_operations import segmentation_enclosed

    out = _out_params(task.outputs["o"])
    seg1 = _resolve_one(overlay, task.inputs["i1"], cache)
    seg2 = _resolve_one(overlay, task.inputs["i2"], cache)
    if seg1 is None or seg2 is None:
        raise ExecError(f"enclosed: inputs {task.inputs} not found in run {overlay.name}")

    kw = {}
    if "margin" in task.params:
        kw["margin"] = int(task.params["margin"])
    if "connectivity" in task.params:
        kw["connectivity"] = task.params["connectivity"]
    if task.params.get("min_size") is not None:
        kw["min_size"] = float(task.params["min_size"])
    if task.params.get("max_size") is not None:
        kw["max_size"] = float(task.params["max_size"])

    segmentation_enclosed(
        segmentation1=seg1,
        segmentation2=seg2,
        run=overlay,
        object_name=out["object_name"],
        session_id=out["session_id"],
        user_id=out["user_id"],
        voxel_spacing=out["voxel_spacing"],
        **kw,
    )


def _exec_split(overlay: MemoryOverlayRun, task: ConcreteTask, cache: RunCache) -> None:
    from copick_utils.process.split_labels import split_multilabel_segmentation

    in_seg = _resolve_one(overlay, task.inputs["i"], cache)
    if in_seg is None:
        raise ExecError(f"split: input {task.inputs['i']} not found in run {overlay.name}")
    output_user_id = task.params.get("output_user_id", "split")
    labels = task.params.get("labels")  # None => enumerate from np.unique + config
    split_multilabel_segmentation(segmentation=in_seg, run=overlay, output_user_id=output_user_id, labels=labels)


def _exec_combine(overlay: MemoryOverlayRun, task: ConcreteTask, cache: RunCache) -> None:
    from copick_utils.process.combine_labels import combine_labels

    out = _out_params(task.outputs["o"])
    segs = _resolve_segs(overlay, task.inputs["i"], cache)
    if not segs:
        raise ExecError(f"combine: no segmentations matched {task.inputs['i']} in run {overlay.name}")
    combine_labels(
        segmentations=segs,
        run=overlay,
        object_name=out["object_name"],
        session_id=out["session_id"],
        user_id=out["user_id"],
    )


def _exec_filter(overlay: MemoryOverlayRun, task: ConcreteTask, cache: RunCache) -> None:
    from copick_utils.process.filter_components import filter_segmentation_components

    out = _out_params(task.outputs["o"])
    seg = _resolve_one(overlay, task.inputs["i"], cache)
    if seg is None:
        raise ExecError(f"filter-components: input {task.inputs['i']} not found in run {overlay.name}")
    vs = task.voxel_spacing
    min_size = task.params.get("min_size")
    max_size = task.params.get("max_size")
    if task.params.get("size_unit") == "voxel":  # CLI wrapper converts voxel^3 -> angstrom^3
        vv = vs**3
        min_size = None if min_size is None else float(min_size) * vv
        max_size = None if max_size is None else float(max_size) * vv
    filter_segmentation_components(
        segmentation=seg,
        run=overlay,
        object_name=out["object_name"],
        session_id=out["session_id"],
        user_id=out["user_id"],
        connectivity=task.params.get("connectivity", "all"),
        min_size=None if min_size is None else float(min_size),
        max_size=None if max_size is None else float(max_size),
        keep_largest=(int(task.params["keep_largest"]) if task.params.get("keep_largest") is not None else None),
    )


def _exec_expand(overlay: MemoryOverlayRun, task: ConcreteTask, cache: RunCache) -> None:
    from copick_utils.process.expand_labels import expand_labels_segmentation

    out = _out_params(task.outputs["o"])
    seg = _resolve_one(overlay, task.inputs["i"], cache)
    if seg is None:
        raise ExecError(f"expand-labels: input {task.inputs['i']} not found in run {overlay.name}")
    vs = task.voxel_spacing
    distance = float(task.params["distance"])
    max_hole = task.params.get("max_hole_size")
    max_hole = None if max_hole is None else float(max_hole)
    if task.params.get("size_unit") == "voxel":  # CLI: distance*vs (linear), hole*vs^3 (cubic)
        distance *= vs
        if max_hole is not None:
            max_hole *= vs**3
    expand_labels_segmentation(
        segmentation=seg,
        run=overlay,
        object_name=out["object_name"],
        session_id=out["session_id"],
        user_id=out["user_id"],
        distance=distance,
        max_hole_size=max_hole,
        connectivity=task.params.get("connectivity", "all"),
    )


def _exec_thickness(overlay: MemoryOverlayRun, task: ConcreteTask, cache: RunCache) -> None:
    from copick_utils.process.thickness_filter import thickness_filter_segmentation

    out = _out_params(task.outputs["o"])
    seg = _resolve_one(overlay, task.inputs["i"], cache)
    if seg is None:
        raise ExecError(f"thickness-filter: input {task.inputs['i']} not found in run {overlay.name}")
    min_thickness = float(task.params["min_thickness"])
    if task.params.get("thickness_unit") == "voxel":  # CLI: min_thickness*vs (linear)
        min_thickness *= task.voxel_spacing
    thickness_filter_segmentation(
        segmentation=seg,
        run=overlay,
        object_name=out["object_name"],
        session_id=out["session_id"],
        user_id=out["user_id"],
        min_thickness=min_thickness,
    )


def _exec_manage(overlay: MemoryOverlayRun, root, task: ConcreteTask, cache: RunCache) -> None:
    """cp/mv/rm. cp/mv go through the write-through overlay (read the source from the per-run cache,
    write the destination the same way as every other op) -- avoiding a stale read from the shared,
    heavily-mutated root that ``copy_copick_objects_batch`` is prone to. rm deletes via copick.
    """
    op = get_op(task.op_id)
    object_type = op.object_type_positional
    if object_type != "segmentation":
        raise ExecError(
            f"the memory backend does not support {op.kind} for {object_type!r}; " "use the bash or Nextflow backend",
        )

    if op.kind in ("cp", "mv"):
        src_seg = _resolve_one(overlay, task.inputs["src"], cache)
        if src_seg is None:
            raise ExecError(f"{op.kind}: source {task.inputs['src']} not found in run {overlay.name}")
        arr = src_seg.numpy()
        dst = _out_params(task.outputs["dst"])
        out_seg = overlay.new_segmentation(
            dst["object_name"],
            dst["user_id"],
            dst["session_id"],
            bool(getattr(src_seg, "is_multilabel", False)),
            dst["voxel_spacing"],
            exist_ok=bool(task.params.get("overwrite", False)),
        )
        out_seg.from_numpy(arr)
        if op.kind == "mv":
            result = _remove(root, task.inputs["src"], task.run)
            if result["deleted"] < 1:
                raise ExecError(f"mv: destination was written but source {task.inputs['src']!r} was not removed")
            cache.evict(src_seg.cache_uri)
    elif op.kind == "rm":
        dry_run = bool(task.params.get("dry_run", False))
        _remove(root, task.inputs["i"], task.run, dry_run=dry_run)
        if not dry_run:
            # rm accepts patterns, so conservatively invalidate the per-run cache after durable success.
            cache.clear()


def _remove(root, uri: str, run: Optional[str], dry_run: bool = False) -> dict:
    from copick.ops.manage import remove_copick_objects_batch

    results = remove_copick_objects_batch(
        root=root,
        object_type="segmentation",
        uri=uri,
        run_names=[run] if run is not None else None,
        dry_run=dry_run,
        workers=1,
    )
    errors: List[str] = []
    deleted = 0
    objects: List[str] = []
    if not isinstance(results, dict):
        raise ExecError(f"remove returned an invalid result for {uri!r}: {results!r}")
    for run_name, result in results.items():
        if result is None:
            errors.append(f"run {run_name!r} failed completely")
            continue
        if not isinstance(result, dict):
            errors.append(f"run {run_name!r} returned {result!r}")
            continue
        deleted += int(result.get("deleted", 0))
        objects.extend(str(item) for item in result.get("objects", []))
        errors.extend(f"run {run_name!r}: {error}" for error in result.get("errors", []))
    if errors:
        raise ExecError(f"remove failed for {uri!r}: {'; '.join(errors)}")
    return {"deleted": deleted, "objects": objects, "dry_run": dry_run}
