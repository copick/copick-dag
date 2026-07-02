"""Bash backend: lower a ``ResolvedPlan`` to a reproducible ``copick`` CLI script."""

from copick_dag.backends.bash.emitter import render_bash, task_argv

__all__ = ["render_bash", "task_argv"]
