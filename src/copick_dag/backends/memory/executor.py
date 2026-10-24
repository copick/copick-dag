"""Drive a ``ResolvedPlan`` in-process over the write-through memory overlay."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List

from copick.util.log import get_logger

from copick_dag.backends.memory.ops import exec_task
from copick_dag.backends.memory.overlay import MemoryOverlayRun, RunCache
from copick_dag.plan.concrete import ConcreteTask, ResolvedPlan

logger = get_logger(__name__)


@dataclass
class MemoryStats:
    """Instrumentation from an in-memory run."""

    runs: List[str] = field(default_factory=list)
    project_parses: int = 0
    per_run: Dict[str, dict] = field(default_factory=dict)

    @property
    def total_hits(self) -> int:
        return sum(r["hits"] for r in self.per_run.values())

    @property
    def total_misses(self) -> int:
        return sum(r["misses"] for r in self.per_run.values())

    @property
    def total_writes(self) -> int:
        return sum(r["writes"] for r in self.per_run.values())

    @property
    def errors(self) -> List[str]:
        out: List[str] = []
        for r in self.per_run.values():
            out.extend(r["errors"])
        return out


def run_memory(
    plan: ResolvedPlan,
    workers: int = 1,
    root=None,
) -> MemoryStats:
    """Execute ``plan`` in-memory.

    Args:
        plan: a concrete plan built WITH a project (run-set enumerated). ``run is None`` per-run tasks
            are rejected -- the in-memory backend needs concrete runs. (Barrier tasks, added in M3b,
            carry a ``run_scope`` instead and run in a second phase.)
        workers: max runs to execute concurrently. **Default 1 (sequential).** copick's ``CopickRoot``
            is not safe to mutate from multiple threads, so when ``workers > 1`` each run thread opens
            its OWN root (isolation); the shared-root fast path is used only when ``workers == 1``.
            Sequential execution still delivers the core wins: the project is parsed once and every
            intermediate is served from the per-run memory cache (no zarr re-reads).
        root: optional pre-opened ``CopickRoot`` (single-parse fast path; used only when workers==1).
    """
    import copick

    order = plan.topo_order()
    per_run_order = [t for t in order if t.scope != "barrier"]
    barrier_order = [t for t in order if t.scope == "barrier"]
    if any(t.run is None for t in per_run_order):
        raise ValueError(
            "in-memory backend requires a planned run-set; build the plan with a project (config).",
        )

    isolate = workers > 1
    parses = 0
    shared = root
    if shared is None:
        shared = copick.from_file(plan.config)
        parses = 1

    by_run: "OrderedDict[str, List[ConcreteTask]]" = OrderedDict()
    for t in per_run_order:
        by_run.setdefault(t.run, []).append(t)

    def run_one(run_name: str, tasks: List[ConcreteTask]) -> dict:
        r = copick.from_file(plan.config) if isolate else shared
        real = r.get_run(run_name)
        if real is None:
            return {"hits": 0, "misses": 0, "writes": 0, "errors": [f"run {run_name!r} not found in project"]}
        cache = RunCache()
        overlay = MemoryOverlayRun(real, cache)
        errors: List[str] = []
        for t in tasks:
            try:
                exec_task(overlay, r, t, cache)
            except Exception as e:  # noqa: BLE001
                if t.optional:
                    errors.append(f"[optional skipped] {t.task_id}: {e}")
                    logger.debug("optional task %s skipped: %s", t.task_id, e)
                else:
                    errors.append(f"[ERROR] {t.task_id}: {e}")
                    raise
        return {"hits": cache.hits, "misses": cache.misses, "writes": cache.writes, "errors": errors}

    per_run: Dict[str, dict] = {}
    effective_workers = max(1, min(workers, len(by_run)))
    if isolate and effective_workers > 1 and len(by_run) > 1:
        with ThreadPoolExecutor(max_workers=effective_workers) as ex:
            futs = {ex.submit(run_one, rn, ts): rn for rn, ts in by_run.items()}
            for f in as_completed(futs):
                per_run[futs[f]] = f.result()
        parses += len(by_run)
    else:
        for rn, ts in by_run.items():
            per_run[rn] = run_one(rn, ts)

    # ---- barrier phase (M3b): all per-run work is done and durably on zarr ----
    barrier_errors = _run_barriers(barrier_order, shared)

    stats = MemoryStats(runs=list(by_run), project_parses=parses, per_run=per_run)
    if barrier_errors:
        stats.per_run.setdefault("__barriers__", {"hits": 0, "misses": 0, "writes": 0, "errors": []})
        stats.per_run["__barriers__"]["errors"].extend(barrier_errors)
    return stats


def _run_barriers(barrier_tasks: List[ConcreteTask], root) -> List[str]:
    """Run cross-run barrier tasks (e.g. seg-stats) after all per-run subgraphs complete."""
    from copick_dag.backends.memory.barrier import exec_barrier

    errors: List[str] = []
    for t in barrier_tasks:
        try:
            exec_barrier(root, t)
        except Exception as e:  # noqa: BLE001
            if t.optional:
                errors.append(f"[optional skipped] {t.task_id}: {e}")
            else:
                errors.append(f"[ERROR] {t.task_id}: {e}")
                raise
    return errors
