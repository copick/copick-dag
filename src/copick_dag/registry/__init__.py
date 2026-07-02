"""The operation registry: one source of truth per copick op.

Each :class:`~copick_dag.registry.spec.OpSpec` links, for a single ``op_id``:

- the CLI command path + how each slot/param maps to flags (used by the bash emitter);
- the input/output slot roles, arities, and object types (used by validation + edge inference);
- ``scope`` (per-run vs barrier) and ``vs_style`` (how voxel spacing is passed);
- whether the op is ``streamable`` in the in-memory backend.

The hand-authored overlay lives in :mod:`copick_dag.registry.overlay`; the
:mod:`copick_dag.registry.introspect` module cross-checks it against the *installed* copick CLI.
"""

from typing import Dict, List

from copick_dag.registry.overlay import OPS
from copick_dag.registry.spec import OpSpec, ParamSpec, SlotSpec

__all__ = ["OpSpec", "SlotSpec", "ParamSpec", "get_op", "all_ops", "op_ids"]


def get_op(op_id: str) -> OpSpec:
    try:
        return OPS[op_id]
    except KeyError:
        raise KeyError(
            f"unknown op {op_id!r}; known ops: {', '.join(sorted(OPS))}",
        ) from None


def all_ops() -> Dict[str, OpSpec]:
    return dict(OPS)


def op_ids() -> List[str]:
    return sorted(OPS)
