"""copick-dag: an abstract DAG-pipeline IR for copick with pluggable execution backends.

A copick pipeline is authored once as a declarative graph (``GraphSpec``), compiled by a
project-querying planner into a backend-agnostic ``ResolvedPlan`` (the concrete DAG), and then
materialized by an emitter/executor:

- ``bash``      -- emit a ``copick`` CLI script (reproducible, same behaviour as today).
- ``memory``    -- run in-process with a write-through memory overlay, parallel across runs
                   and along the graph, keeping arrays in memory between ops.
- ``nextflow``  -- per-run process parallelism for local and SLURM execution.

See ``copick_dag.ir`` (the IR), ``copick_dag.registry`` (the operation registry),
``copick_dag.planner`` (abstract -> concrete), and ``copick_dag.backends`` (emitters).
"""

__version__ = "2.0.0-alpha.1"

from copick_dag.ir.model import GraphSpec, GraphNode  # noqa: E402
from copick_dag.plan.concrete import ConcreteTask, ResolvedPlan  # noqa: E402

__all__ = [
    "__version__",
    "GraphSpec",
    "GraphNode",
    "ConcreteTask",
    "ResolvedPlan",
]
