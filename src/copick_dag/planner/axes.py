"""Resolve the run and voxel-spacing axes of a graph.

Voxel spacing is a first-class sweep axis, not a literal. Because copick matches spacings by *exact*
float equality (a stored ``10.006`` is never matched by ``10.0``), the ``discover`` mode resolves a
requested (round) spacing against the spacings actually present in each run, within a tolerance, and
raises at build time on a miss or an ambiguity.
"""

from __future__ import annotations

from typing import List, Optional

from copick_dag.ir.model import DiscoverSpec, GraphSpec
from copick_dag.planner.context import ProjectView


class AxisError(ValueError):
    pass


def resolve_runs(spec: GraphSpec, project: Optional[ProjectView]) -> List[Optional[str]]:
    """Return the concrete run-set, or ``[None]`` (a single "all runs" pseudo-run) when unknown."""
    if spec.runs:
        return list(spec.runs)
    if project is not None:
        runs = project.run_names()
        if not runs:
            raise AxisError("project contains no runs")
        return runs
    return [None]


def resolve_vs_axis(
    spec: GraphSpec,
    run: Optional[str],
    project: Optional[ProjectView],
) -> List[Optional[float]]:
    """Return the voxel spacings to sweep for ``run``.

    - ``None``            -> ``[defaults.voxel_spacing]`` if set, else ``[None]`` (per-node vs).
    - scalar / list       -> those exact spacings.
    - discover            -> spacings present in the project (per run), optionally matched to a
                             requested value within ``tol``.
    """
    v = spec.voxel_spacings
    if v is None:
        if spec.defaults.voxel_spacing is not None:
            return [float(spec.defaults.voxel_spacing)]
        return [None]
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, list):
        return [float(x) for x in v]

    # DiscoverSpec
    assert isinstance(v, DiscoverSpec)
    if project is None:
        raise AxisError("voxel_spacings.discover requires a project config to query")
    present = project.voxel_spacings(run) if run is not None else project.all_voxel_spacings()
    if not present:
        raise AxisError(f"no voxel spacings present for run {run!r}")
    if v.match is None:
        return list(present)
    cands = [s for s in present if abs(s - v.match) <= v.tol]
    if not cands:
        raise AxisError(
            f"no voxel spacing within tol={v.tol} of {v.match} for run {run!r}; present={present}",
        )
    if len(cands) > 1:
        raise AxisError(
            f"ambiguous voxel spacing for run {run!r}: {cands} all match {v.match}±{v.tol}",
        )
    return [cands[0]]
