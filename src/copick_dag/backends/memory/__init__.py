"""In-memory backend: execute a ``ResolvedPlan`` in-process with a write-through memory overlay.

The project is parsed once; per-run subgraphs stream through their whole chain (intermediate arrays
served from memory, no zarr re-read) while runs run in parallel. Every intermediate is also written
through to zarr, so results are durable/resumable and identical to the bash backend.
"""

from copick_dag.backends.memory.executor import MemoryStats, run_memory

__all__ = ["run_memory", "MemoryStats"]
