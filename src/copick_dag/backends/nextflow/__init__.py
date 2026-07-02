"""Nextflow backend: emit a DSL2 workflow (+ config) from a ``ResolvedPlan`` (M4, emit-only).

One process per concrete task, running the same ``copick`` command as the bash backend, wired by
done-signal channels that mirror the task ``deps`` graph. Process isolation (one OS process per
task, own ``CopickRoot``) is what makes this the safe out-of-core / HPC-parallel backend.
"""

from copick_dag.backends.nextflow.emitter import render_nextflow, render_nextflow_config

__all__ = ["render_nextflow", "render_nextflow_config"]
