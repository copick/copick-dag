"""The concrete DAG (``ResolvedPlan`` / ``ConcreteTask``) -- the single lowering target that all
backends consume."""

from copick_dag.plan.concrete import ConcreteTask, ResolvedPlan

__all__ = ["ConcreteTask", "ResolvedPlan"]
