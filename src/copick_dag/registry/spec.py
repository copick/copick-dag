"""Dataclasses describing an operation in the registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# scope
SCOPE_PER_RUN = "per_run"
SCOPE_BARRIER = "barrier"

# how voxel spacing reaches the command
VS_FLAG = "flag"  # passed as a separate -vs option; URIs must NOT carry @vs
VS_URI = "uri"  # carried inside each URI as @voxel_spacing
VS_NONE = "none"  # op does not take a voxel spacing distinct from its URIs (cp/mv/rm)


@dataclass(frozen=True)
class SlotSpec:
    """One input or output URI slot of an op."""

    slot: str  # key used in node.in / node.out
    object_type: str = "segmentation"
    role: str = "input"  # "input" | "reference" | "output"
    flag: Optional[str] = None  # CLI flag (e.g. "-i", "-rs", "-o"); None => positional/declared
    multiple: bool = False  # accepts a list of URIs
    positional: bool = False  # rendered as a positional CLI argument (cp/mv/rm)
    order: int = 0  # positional order
    required: bool = True
    declared: bool = False  # edges-only (e.g. split's wildcard output); never emitted or linted
    is_file: bool = False  # a filesystem-path output (seg-stats -op), not a copick URI


@dataclass(frozen=True)
class ParamSpec:
    """A scalar parameter mapped to a CLI option."""

    name: str  # key in node.params
    flag: str  # CLI flag (e.g. "--operation", "--max-distance", "-vs", "-m")
    kind: str = "value"  # "value" | "bool" (boolean flag, emitted only when true)
    required: bool = False
    default: Any = None


@dataclass(frozen=True)
class OpSpec:
    op_id: str
    cli_path: Tuple[str, ...]  # e.g. ("logical", "segop") or ("cp",)
    scope: str = SCOPE_PER_RUN
    vs_style: str = VS_URI
    inputs: List[SlotSpec] = field(default_factory=list)
    outputs: List[SlotSpec] = field(default_factory=list)
    params: List[ParamSpec] = field(default_factory=list)
    object_type_positional: Optional[str] = None  # cp/mv/rm: the object-type positional arg
    streamable: bool = True
    kind: str = "op"  # "op" | "cp" | "mv" | "rm"

    # -- lookups -----------------------------------------------------------

    def input_slot(self, slot: str) -> Optional[SlotSpec]:
        for s in self.inputs:
            if s.slot == slot:
                return s
        return None

    def output_slot(self, slot: str) -> Optional[SlotSpec]:
        for s in self.outputs:
            if s.slot == slot:
                return s
        return None

    def param(self, name: str) -> Optional[ParamSpec]:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def data_input_slots(self) -> List[SlotSpec]:
        """Input slots that participate in data-dependency edge inference (segmentation ins/refs)."""
        return [s for s in self.inputs if s.object_type == "segmentation"]
