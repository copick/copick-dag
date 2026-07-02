"""The write-through memory overlay.

A :class:`MemoryOverlayRun` wraps a real ``CopickRun`` and is passed to the existing copick per-run
converter functions unmodified. It intercepts the two I/O points those converters use:

- ``run.new_segmentation(...).from_numpy(data)`` -> persist to zarr *and* cache ``data`` in memory;
- ``segmentation.numpy()`` -> serve the cached array when present (no zarr re-read).

Because writes go through to zarr, the in-memory cache is a pure read-locality optimization: it can
be dropped at any time and the next read falls back to zarr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class RunCache:
    """Per-run array cache + hit/miss counters (for instrumentation)."""

    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def get(self, uri: str):
        return self.arrays.get(uri)

    def put(self, uri: str, data: np.ndarray) -> None:
        self.arrays[uri] = np.asarray(data)
        self.writes += 1

    def evict(self, uri: str) -> None:
        self.arrays.pop(uri, None)


def seg_uri(name: str, user_id: str, session_id: str, voxel_size) -> str:
    return f"{name}:{user_id}/{session_id}@{voxel_size}"


class CacheSeg:
    """Proxy over a real ``CopickSegmentation`` that serves ``.numpy()`` from the cache when present
    and (for outputs) captures ``.from_numpy()`` into the cache while still writing zarr."""

    def __init__(self, real, uri: str, cache: RunCache):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_uri", uri)
        object.__setattr__(self, "_cache", cache)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def numpy(self, *args, **kwargs):
        if not args and not kwargs:
            cached = self._cache.get(self._uri)
            if cached is not None:
                self._cache.hits += 1
                return cached
            self._cache.misses += 1
        return self._real.numpy(*args, **kwargs)

    def from_numpy(self, data, *args, **kwargs):
        self._cache.put(self._uri, data)
        return self._real.from_numpy(data, *args, **kwargs)


class MemoryOverlayRun:
    """Wrap a real ``CopickRun``; intercept ``new_segmentation`` for write-through caching."""

    def __init__(self, real_run, cache: RunCache):
        object.__setattr__(self, "_run", real_run)
        object.__setattr__(self, "_cache", cache)

    def __getattr__(self, name):
        return getattr(self._run, name)

    @property
    def cache(self) -> RunCache:
        return self._cache

    def new_segmentation(self, name, user_id, session_id, is_multilabel, voxel_size, exist_ok=False, **kwargs):
        real = self._run.new_segmentation(
            name=name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=is_multilabel,
            voxel_size=voxel_size,
            exist_ok=exist_ok,
            **kwargs,
        )
        return CacheSeg(real, seg_uri(name, user_id, session_id, voxel_size), self._cache)
