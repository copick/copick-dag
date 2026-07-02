"""The planner: compile an abstract ``GraphSpec`` into a concrete ``ResolvedPlan``.

Stage 1 opens the copick project once (optional), resolves the run and voxel-spacing axes, expands
one task per ``(node, run, voxel_spacing)``, and infers data-dependency edges from URI matching.
"""

from copick_dag.planner.build import PlanError, plan
from copick_dag.planner.context import ProjectView

__all__ = ["plan", "PlanError", "ProjectView"]
