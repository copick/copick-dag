"""Pydantic models for the abstract IR (``GraphSpec``).

A ``GraphSpec`` is deliberately CLI-shaped: each node maps ~1:1 to a ``copick <group> <cmd>``
invocation. Data-dependency edges are *not* written explicitly -- they are inferred by the planner
from matching output URIs to input URIs -- but ``needs``/``after`` allow explicit overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# voxel-spacing axis
# ---------------------------------------------------------------------------


class DiscoverSpec(BaseModel):
    """Discover the voxel-spacing axis from the project rather than hard-coding it.

    copick matches voxel spacings by *exact float equality* (a stored ``10.006`` is never matched
    by ``10.0``). ``match`` + ``tol`` let the graph be written in round numbers while the planner
    binds the true per-run float.
    """

    model_config = ConfigDict(extra="forbid")

    discover: bool = True
    match: Optional[float] = None  # requested (possibly round) spacing; None => take all present
    tol: float = 0.0  # absolute tolerance when resolving ``match`` against actual spacings


# scalar (single spacing) | explicit list | discover-from-project
VoxelSpacingsSpec = Union[float, int, List[float], DiscoverSpec]


# ---------------------------------------------------------------------------
# dynamic-frontier value types (M3)
# ---------------------------------------------------------------------------


class CoalesceSpec(BaseModel):
    """An input that resolves to the first *existing* URI at run time (first = preferred)."""

    model_config = ConfigDict(extra="forbid")
    coalesce: List[str]


class SwitchSpec(BaseModel):
    """An output URI chosen by a bound variable (``by``, default ``name``); ``default`` is fallback."""

    model_config = ConfigDict(extra="forbid")
    switch: Dict[str, str]
    by: str = "name"


class ExpandSpec(BaseModel):
    """Expand a node into one concrete node per row of a CSV.

    ``bind`` maps a variable name to a 1-based CSV column; ``normalize`` rewrites values of the
    first bound column (fixing source typos, e.g. ``roug-er`` -> ``rough-er``).
    """

    model_config = ConfigDict(extra="forbid")
    from_csv: str
    bind: Dict[str, int]
    normalize: Dict[str, str] = Field(default_factory=dict)


class SweepSpec(BaseModel):
    """A named matrix: expand the whole node set once per row. Reserved row keys: ``tag`` (label),
    ``vs`` (voxel spacing), ``runs`` (per-row run-set); all other keys are free variables usable as
    ``${var}``."""

    model_config = ConfigDict(extra="forbid")
    over: List[Dict[str, Any]]


InputValue = Union[str, List[str], CoalesceSpec]
OutputValue = Union[str, SwitchSpec]


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A single operation in the graph."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: Optional[str] = None
    op: str  # registry key, e.g. "logical.segop"
    params: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, InputValue] = Field(default_factory=dict, alias="in")
    out: Dict[str, OutputValue] = Field(default_factory=dict)
    scope: Optional[str] = None  # override registry default ("per_run" | "barrier")
    optional: Optional[bool] = None  # try-if-exists (None => registry/planner default)
    after: List[str] = Field(default_factory=list)  # explicit ordering override
    needs: Dict[str, str] = Field(default_factory=dict)  # explicit edge binding (slot -> node id)
    when: Optional[str] = None  # content-dependent predicate (dynamic frontier, M3)
    expand: Optional[ExpandSpec] = None  # CSV expansion (M3)
    over: Optional[List[Dict[str, Any]]] = None  # per-node local sweep (e.g. seg-stats variants)


class ProjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: str


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voxel_spacing: Optional[float] = None


# ---------------------------------------------------------------------------
# finalize block (moves / copies / gc)
# ---------------------------------------------------------------------------


class MoveCopy(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    src: str = Field(alias="from")
    dst: str = Field(alias="to")
    object_type: str = "segmentation"
    optional: bool = False
    overwrite: bool = False


class GcSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    remove: List[str] = Field(default_factory=list)
    object_type: str = "segmentation"


class FinalizeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    moves: List[MoveCopy] = Field(default_factory=list)
    copies: List[MoveCopy] = Field(default_factory=list)
    gc: Optional[GcSpec] = None


# ---------------------------------------------------------------------------
# top-level graph
# ---------------------------------------------------------------------------


class GraphSpec(BaseModel):
    """The full pipeline specification."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    version: int = 1
    project: ProjectSpec
    defaults: Defaults = Field(default_factory=Defaults)
    voxel_spacings: Optional[VoxelSpacingsSpec] = None
    runs: Optional[List[str]] = None  # None => all runs in the project
    sweep: Optional[SweepSpec] = None  # named matrix: expand the node set per row (M3)
    nodes: List[GraphNode] = Field(default_factory=list)
    finalize: Optional[FinalizeSpec] = None

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphSpec":
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "GraphSpec":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)
