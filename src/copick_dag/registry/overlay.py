"""Hand-authored operation metadata (the ~4 things Click introspection cannot know).

This overlay is the *self-contained* source of truth for copick-dag (no edits to copick-utils).
Flags here are cross-checked against the installed copick CLI by
:func:`copick_dag.registry.introspect.lint_registry`.

Scope for M0-M2: the ops used by ``example_1`` (segop, clipseg, enclosed) plus the core object
ops (cp/mv/rm). More ops (process.*, convert.*, octopi.*) are added in later milestones.
"""

from __future__ import annotations

from typing import Dict

from copick_dag.registry.spec import (
    SCOPE_BARRIER,
    SCOPE_PER_RUN,
    VS_FLAG,
    VS_URI,
    OpSpec,
    ParamSpec,
    SlotSpec,
)

OPS: Dict[str, OpSpec] = {}


def _register(op: OpSpec) -> None:
    OPS[op.op_id] = op


# ---------------------------------------------------------------------------
# logical.segop -- boolean operations between segmentations
#   voxel spacing via -vs (URIs carry no @vs); union takes N inputs, the rest exactly 2.
# ---------------------------------------------------------------------------
_register(
    OpSpec(
        op_id="logical.segop",
        cli_path=("logical", "segop"),
        scope=SCOPE_PER_RUN,
        vs_style=VS_FLAG,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i", multiple=True)],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[ParamSpec("operation", "--operation", required=True)],
        streamable=True,
    ),
)

# ---------------------------------------------------------------------------
# logical.clipseg -- limit segmentation voxels by distance to a reference
#   voxel spacing carried in the URIs; reference is one of -rs / -rm / -rt.
# ---------------------------------------------------------------------------
_register(
    OpSpec(
        op_id="logical.clipseg",
        cli_path=("logical", "clipseg"),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        inputs=[
            SlotSpec("i", object_type="segmentation", role="input", flag="-i"),
            SlotSpec("rs", object_type="segmentation", role="reference", flag="-rs", required=False),
            SlotSpec("rm", object_type="mesh", role="reference", flag="-rm", required=False),
            SlotSpec("rt", object_type="tomogram", role="reference", flag="-rt", required=False),
        ],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("max_distance", "--max-distance", required=True),
            ParamSpec("invert", "--invert", kind="bool"),
            ParamSpec("mesh_voxel_spacing", "--mesh-voxel-spacing"),
        ],
        streamable=True,
    ),
)

# ---------------------------------------------------------------------------
# logical.enclosed -- remove enclosed components (voxel spacing via -vs)
# ---------------------------------------------------------------------------
_register(
    OpSpec(
        op_id="logical.enclosed",
        cli_path=("logical", "enclosed"),
        scope=SCOPE_PER_RUN,
        vs_style=VS_FLAG,
        inputs=[
            SlotSpec("i1", object_type="segmentation", role="input", flag="-i1"),
            SlotSpec("i2", object_type="segmentation", role="input", flag="-i2"),
        ],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("margin", "--margin"),
            ParamSpec("connectivity", "--connectivity"),
            ParamSpec("min_size", "--min-size"),
            ParamSpec("max_size", "--max-size"),
        ],
        streamable=True,
    ),
)

# ---------------------------------------------------------------------------
# core object ops: cp / mv / rm  (positional: OBJECT_TYPE SRC [DST]; single -r; not streamable)
# ---------------------------------------------------------------------------
_register(
    OpSpec(
        op_id="core.cp",
        cli_path=("cp",),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        kind="cp",
        object_type_positional="segmentation",
        inputs=[SlotSpec("src", object_type="segmentation", role="input", positional=True, order=1)],
        outputs=[SlotSpec("dst", object_type="segmentation", role="output", positional=True, order=2)],
        params=[ParamSpec("overwrite", "--overwrite", kind="bool")],
        streamable=False,
    ),
)

_register(
    OpSpec(
        op_id="core.mv",
        cli_path=("mv",),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        kind="mv",
        object_type_positional="segmentation",
        inputs=[SlotSpec("src", object_type="segmentation", role="input", positional=True, order=1)],
        outputs=[SlotSpec("dst", object_type="segmentation", role="output", positional=True, order=2)],
        params=[ParamSpec("overwrite", "--overwrite", kind="bool")],
        streamable=False,
    ),
)

_register(
    OpSpec(
        op_id="core.rm",
        cli_path=("rm",),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        kind="rm",
        object_type_positional="segmentation",
        inputs=[SlotSpec("i", object_type="segmentation", role="input", positional=True, order=1)],
        outputs=[],
        params=[ParamSpec("force", "--force", kind="bool"), ParamSpec("dry_run", "--dry-run", kind="bool")],
        streamable=False,
    ),
)

# ---------------------------------------------------------------------------
# process ops (M3): split (data-dependent fan-out), combine (glob fan-in),
# filter-components, expand-labels, thickness-filter. All carry vs in the URI.
# ---------------------------------------------------------------------------

# split: multilabel -> one binary per present label. No -o; output is a DECLARED wildcard
# (`*:<output_user_id>/<session>`) used only for edge inference. output_user_id is a param.
_register(
    OpSpec(
        op_id="process.split",
        cli_path=("process", "split"),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag=None, declared=True)],
        params=[ParamSpec("output_user_id", "--output-user-id"), ParamSpec("labels", "--labels")],
        streamable=True,
    ),
)

# combine: `*`-pattern of binaries -> one multilabel.
_register(
    OpSpec(
        op_id="process.combine",
        cli_path=("process", "combine"),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[],
        streamable=True,
    ),
)

_register(
    OpSpec(
        op_id="process.filter-components",
        cli_path=("process", "filter-components"),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("min_size", "--min-size"),
            ParamSpec("max_size", "--max-size"),
            ParamSpec("size_unit", "--size-unit"),
            ParamSpec("keep_largest", "--keep-largest"),
            ParamSpec("connectivity", "--connectivity"),
        ],
        streamable=True,
    ),
)

_register(
    OpSpec(
        op_id="process.expand-labels",
        cli_path=("process", "expand-labels"),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("distance", "--distance", required=True),
            ParamSpec("max_hole_size", "--max-hole-size"),
            ParamSpec("size_unit", "--size-unit"),
            ParamSpec("connectivity", "--connectivity"),
        ],
        streamable=True,
    ),
)

_register(
    OpSpec(
        op_id="process.thickness-filter",
        cli_path=("process", "thickness-filter"),
        scope=SCOPE_PER_RUN,
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("min_thickness", "--min-thickness", required=True),
            ParamSpec("thickness_unit", "--thickness-unit"),
        ],
        streamable=True,
    ),
)

# seg-stats: cross-run BARRIER. Reads one segmentation URI across many runs and writes a single
# file (CSV/plot). The output is a filesystem path (a param), not a copick object -- terminal sink.
_register(
    OpSpec(
        op_id="process.seg-stats",
        cli_path=("process", "seg-stats"),
        scope=SCOPE_BARRIER,
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[],
        params=[
            ParamSpec("output_path", "--output-path", required=True),
            ParamSpec("output_format", "--output-format"),
            ParamSpec("connectivity", "--connectivity"),
            ParamSpec("include_background", "--include-background", kind="bool"),
        ],
        streamable=False,
    ),
)
