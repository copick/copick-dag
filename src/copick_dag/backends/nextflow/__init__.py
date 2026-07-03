"""Nextflow backend: emit a DSL2 workflow (+ config) from a ``ResolvedPlan`` (M4, emit-only).

One process per concrete task, running the same ``copick`` command as the bash backend, wired by
done-signal channels that mirror the task ``deps`` graph. Process isolation (one OS process per
task, own ``CopickRoot``) is what makes this the safe out-of-core / HPC-parallel backend.

:func:`render_subworkflow` emits a **named, composable** sub-workflow instead of a standalone one,
so a host Nextflow pipeline can ``include`` and run the DAG inside its own single ``nextflow run``.
"""

from copick_dag.backends.nextflow.emitter import (
    render_nextflow,
    render_nextflow_config,
    render_subworkflow,
)

__all__ = ["render_nextflow", "render_nextflow_config", "render_subworkflow"]
