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
    VS_NONE,
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

# ===========================================================================
# convert.* -- cross-object-type converters (M6). Canonical flags (normalized CLI).
# vs_style is per the PRIMARY object: VS_URI when a segmentation slot is involved (carries @vs),
# VS_NONE for pure picks<->mesh converters. `-im` fans out to `{instance_id}`-templated meshes.
# ===========================================================================

# clustering options shared by the picks->mesh converters
_CLUSTER = [
    ParamSpec("use_clustering", "--use-clustering", kind="bool"),
    ParamSpec("clustering_method", "--clustering-method"),
    ParamSpec("clustering_eps", "--clustering-eps"),
    ParamSpec("clustering_min_samples", "--clustering-min-samples"),
    ParamSpec("clustering_n_clusters", "--clustering-n-clusters"),
    ParamSpec("all_clusters", "--all-clusters", kind="bool"),
]

_register(
    OpSpec(
        op_id="convert.seg2mesh",
        cli_path=("convert", "seg2mesh"),
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("level", "--level"),
            ParamSpec("step_size", "--step-size"),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
        ],
    ),
)

_register(
    OpSpec(
        op_id="convert.seg2picks",
        cli_path=("convert", "seg2picks"),
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="picks", role="output", flag="-o")],
        params=[
            ParamSpec("segmentation_idx", "--segmentation-idx", required=True),
            ParamSpec("maxima_filter_size", "--maxima-filter-size"),
            ParamSpec("min_particle_size", "--min-particle-size"),
            ParamSpec("max_particle_size", "--max-particle-size"),
        ],
    ),
)

_register(
    OpSpec(
        op_id="convert.mesh2seg",
        cli_path=("convert", "mesh2seg"),
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="mesh", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("mode", "--mode"),
            ParamSpec("boundary_sampling_density", "--boundary-sampling-density"),
            ParamSpec("invert", "--invert", kind="bool"),
            ParamSpec("tomo_type", "--tomo-type"),
        ],
    ),
)

_register(
    OpSpec(
        op_id="convert.picks2seg",
        cli_path=("convert", "picks2seg"),
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="picks", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[ParamSpec("radius", "--radius"), ParamSpec("tomo_type", "--tomo-type")],
    ),
)

_register(
    OpSpec(
        op_id="convert.picks2mesh",
        cli_path=("convert", "picks2mesh"),
        vs_style=VS_NONE,
        inputs=[SlotSpec("i", object_type="picks", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("mesh_type", "--mesh-type"),
            ParamSpec("alpha", "--alpha"),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
            *_CLUSTER,
        ],
    ),
)

_register(
    OpSpec(
        op_id="convert.picks2sphere",
        cli_path=("convert", "picks2sphere"),
        vs_style=VS_NONE,
        inputs=[SlotSpec("i", object_type="picks", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("subdivisions", "--subdivisions"),
            ParamSpec("deduplicate_spheres", "--deduplicate-spheres", kind="bool"),
            ParamSpec("min_sphere_distance", "--min-sphere-distance"),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
            *_CLUSTER,
        ],
    ),
)

_register(
    OpSpec(
        op_id="convert.picks2ellipsoid",
        cli_path=("convert", "picks2ellipsoid"),
        vs_style=VS_NONE,
        inputs=[SlotSpec("i", object_type="picks", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("subdivisions", "--subdivisions"),
            ParamSpec("deduplicate_ellipsoids", "--deduplicate-ellipsoids", kind="bool"),
            ParamSpec("min_ellipsoid_distance", "--min-ellipsoid-distance"),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
            *_CLUSTER,
        ],
    ),
)

_register(
    OpSpec(
        op_id="convert.picks2plane",
        cli_path=("convert", "picks2plane"),
        vs_style=VS_NONE,
        inputs=[SlotSpec("i", object_type="picks", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("padding", "--padding"),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
            *_CLUSTER,
        ],
    ),
)

_register(
    OpSpec(
        op_id="convert.picks2surface",
        cli_path=("convert", "picks2surface"),
        vs_style=VS_NONE,
        inputs=[SlotSpec("i", object_type="picks", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("surface_method", "--surface-method"),
            ParamSpec("grid_resolution", "--grid-resolution"),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
            *_CLUSTER,
        ],
    ),
)

_register(
    OpSpec(
        op_id="convert.mesh2caps",
        cli_path=("convert", "mesh2caps"),
        vs_style=VS_NONE,
        inputs=[SlotSpec("i", object_type="mesh", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("axis", "--axis"),
            ParamSpec("angle_threshold", "--angle-threshold"),
            ParamSpec("surface", "--surface"),
            ParamSpec("auto_axis", "--auto-axis", kind="bool"),
        ],
    ),
)

# logical.meshop -- boolean operations on meshes (the mesh analog of logical.segop; N inputs).
_register(
    OpSpec(
        op_id="logical.meshop",
        cli_path=("logical", "meshop"),
        vs_style=VS_NONE,
        inputs=[SlotSpec("i", object_type="mesh", role="input", flag="-i", multiple=True)],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("operation", "--operation", required=True),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
        ],
    ),
)

# convert.seg2slab (copick_torch plugin) -- fit a slab mesh to a segmentation.
_register(
    OpSpec(
        op_id="convert.seg2slab",
        cli_path=("convert", "seg2slab"),
        vs_style=VS_URI,
        plugin="copick_torch",
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("label", "--label", required=True),
            ParamSpec("method", "--method"),
            ParamSpec("grid_resolution", "--grid-resolution"),
            ParamSpec("regularization", "--regularization"),
            ParamSpec("num_iterations", "--num-iterations"),
        ],
    ),
)

# ===========================================================================
# Phase 2: mixed-type references + tomogram as a primary I/O type.
# vs_style is VS_URI throughout (a seg/tomo reference or tomo primary carries @vs); the mesh/picks
# primary URIs stay vs-free automatically (per-slot @vs handling in the planner/emitter).
# ===========================================================================

# convert.mesh2picks -- sample points on a mesh within a tomogram's bounds -> picks.
_register(
    OpSpec(
        op_id="convert.mesh2picks",
        cli_path=("convert", "mesh2picks"),
        vs_style=VS_URI,
        inputs=[
            SlotSpec("i", object_type="mesh", role="input", flag="-i"),
            SlotSpec("t", object_type="tomogram", role="reference", flag="-t"),
        ],
        outputs=[SlotSpec("o", object_type="picks", role="output", flag="-o")],
        params=[
            ParamSpec("sampling_type", "--sampling-type", required=True),
            ParamSpec("n_points", "--n-points"),
            ParamSpec("min_dist", "--min-dist"),
            ParamSpec("edge_dist", "--edge-dist"),
            ParamSpec("include_normals", "--include-normals", kind="bool"),
            ParamSpec("random_orientations", "--random-orientations", kind="bool"),
            ParamSpec("seed", "--seed"),
        ],
    ),
)

# logical.clipmesh -- limit mesh by distance to a reference (mesh/seg/tomo). Mirror of clipseg.
_register(
    OpSpec(
        op_id="logical.clipmesh",
        cli_path=("logical", "clipmesh"),
        vs_style=VS_URI,
        inputs=[
            SlotSpec("i", object_type="mesh", role="input", flag="-i"),
            SlotSpec("rm", object_type="mesh", role="reference", flag="-rm", required=False),
            SlotSpec("rs", object_type="segmentation", role="reference", flag="-rs", required=False),
            SlotSpec("rt", object_type="tomogram", role="reference", flag="-rt", required=False),
        ],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("max_distance", "--max-distance", required=True),
            ParamSpec("mesh_voxel_spacing", "--mesh-voxel-spacing"),
            ParamSpec("invert", "--invert", kind="bool"),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
        ],
    ),
)

# logical.clippicks -- limit picks by distance to a reference (mesh/seg/tomo).
_register(
    OpSpec(
        op_id="logical.clippicks",
        cli_path=("logical", "clippicks"),
        vs_style=VS_URI,
        inputs=[
            SlotSpec("i", object_type="picks", role="input", flag="-i"),
            SlotSpec("rm", object_type="mesh", role="reference", flag="-rm", required=False),
            SlotSpec("rs", object_type="segmentation", role="reference", flag="-rs", required=False),
            SlotSpec("rt", object_type="tomogram", role="reference", flag="-rt", required=False),
        ],
        outputs=[SlotSpec("o", object_type="picks", role="output", flag="-o")],
        params=[
            ParamSpec("max_distance", "--max-distance", required=True),
            ParamSpec("mesh_voxel_spacing", "--mesh-voxel-spacing"),
            ParamSpec("invert", "--invert", kind="bool"),
        ],
    ),
)

# logical.picksin / picksout -- keep picks inside / outside a reference region (mesh/seg).
for _pid, _cli in (("logical.picksin", ("logical", "picksin")), ("logical.picksout", ("logical", "picksout"))):
    _register(
        OpSpec(
            op_id=_pid,
            cli_path=_cli,
            vs_style=VS_URI,
            inputs=[
                SlotSpec("i", object_type="picks", role="input", flag="-i"),
                SlotSpec("rm", object_type="mesh", role="reference", flag="-rm", required=False),
                SlotSpec("rs", object_type="segmentation", role="reference", flag="-rs", required=False),
            ],
            outputs=[SlotSpec("o", object_type="picks", role="output", flag="-o")],
            params=[],
        ),
    )

# process.validbox -- a box mesh of a tomogram's valid (reconstructed) region. No copick data input.
_register(
    OpSpec(
        op_id="process.validbox",
        cli_path=("process", "validbox"),
        vs_style=VS_URI,
        inputs=[SlotSpec("t", object_type="tomogram", role="reference", flag="-t")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[ParamSpec("angle", "--angle")],
    ),
)

# convert.picks2slab (copick_torch) -- fit a slab mesh between two pick layers within a tomogram.
_register(
    OpSpec(
        op_id="convert.picks2slab",
        cli_path=("convert", "picks2slab"),
        vs_style=VS_URI,
        plugin="copick_torch",
        inputs=[
            SlotSpec("i1", object_type="picks", role="input", flag="-i1"),
            SlotSpec("i2", object_type="picks", role="input", flag="-i2"),
            SlotSpec("t", object_type="tomogram", role="reference", flag="-t"),
        ],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("method", "--method"),
            ParamSpec("grid_resolution", "--grid-resolution"),
            ParamSpec("regularization", "--regularization"),
            ParamSpec("fit_resolution", "--fit-resolution"),
            ParamSpec("num_iterations", "--num-iterations"),
            ParamSpec("learning_rate", "--learning-rate"),
        ],
    ),
)

# process.downsample (copick_torch) -- Fourier-rescale a tomogram to a new spacing (VS-CHANGING).
_register(
    OpSpec(
        op_id="process.downsample",
        cli_path=("process", "downsample"),
        vs_style=VS_URI,
        vs_transform=True,
        plugin="copick_torch",
        inputs=[SlotSpec("i", object_type="tomogram", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="tomogram", role="output", flag="-o")],
        params=[ParamSpec("delete_source", "--delete-source", kind="bool")],
    ),
)

# process.bandpass (copick_torch) -- band-pass filter a tomogram in place (same spacing).
_register(
    OpSpec(
        op_id="process.bandpass",
        cli_path=("process", "bandpass"),
        vs_style=VS_URI,
        plugin="copick_torch",
        inputs=[SlotSpec("i", object_type="tomogram", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="tomogram", role="output", flag="-o")],
        params=[
            ParamSpec("lp_freq", "--lp-freq"),
            ParamSpec("lp_decay", "--lp-decay"),
            ParamSpec("hp_freq", "--hp-freq"),
            ParamSpec("hp_decay", "--hp-decay"),
            ParamSpec("show_filter", "--show-filter", kind="bool"),
        ],
    ),
)

# ===========================================================================
# Phase 3: process specials -- vs-transform (rescale), cross-type (fit-spline), templated fan-out
# (separate-components/skeletonize), mesh fan-out (hull).
# ===========================================================================

# process.rescale -- resample a segmentation to a different voxel spacing (in@vs_in -> out@vs_out).
_register(
    OpSpec(
        op_id="process.rescale",
        cli_path=("process", "rescale"),
        vs_style=VS_URI,
        vs_transform=True,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("target_voxel_spacing", "--target-voxel-spacing"),
            ParamSpec("tomo_type", "--tomo-type"),
            ParamSpec("order", "--order"),
        ],
    ),
)

# process.fit-spline -- fit a spline through a skeleton segmentation -> picks. vs from the input URI
# (no separate -vs emitted).
_register(
    OpSpec(
        op_id="process.fit-spline",
        cli_path=("process", "fit-spline"),
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="picks", role="output", flag="-o")],
        params=[
            ParamSpec("spacing_distance", "--spacing-distance", required=True),
            ParamSpec("smoothing_factor", "--smoothing-factor"),
            ParamSpec("degree", "--degree"),
            ParamSpec("connectivity_radius", "--connectivity-radius"),
        ],
    ),
)

# process.separate-components -- one output segmentation per connected component (session template
# ``{instance_id}``). The templated output is a session-wildcard producer for downstream fan-in.
_register(
    OpSpec(
        op_id="process.separate-components",
        cli_path=("process", "separate-components"),
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("connectivity", "--connectivity"),
            ParamSpec("min_size", "--min-size"),
            ParamSpec("multilabel", "--multilabel", kind="bool"),
        ],
    ),
)

# process.skeletonize -- medial skeleton of a segmentation (session template ``{input_session_id}``).
_register(
    OpSpec(
        op_id="process.skeletonize",
        cli_path=("process", "skeletonize"),
        vs_style=VS_URI,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="segmentation", role="output", flag="-o")],
        params=[
            ParamSpec("method", "--method"),
            ParamSpec("min_object_size", "--min-object-size"),
            ParamSpec("min_branch_length", "--min-branch-length"),
            ParamSpec("remove_noise", "--remove-noise", kind="bool"),
            ParamSpec("remove_short_branches", "--remove-short-branches", kind="bool"),
        ],
    ),
)

# process.hull -- convex hull of a mesh (``-im`` fans out one mesh per input instance).
_register(
    OpSpec(
        op_id="process.hull",
        cli_path=("process", "hull"),
        vs_style=VS_NONE,
        inputs=[SlotSpec("i", object_type="mesh", role="input", flag="-i")],
        outputs=[SlotSpec("o", object_type="mesh", role="output", flag="-o")],
        params=[
            ParamSpec("hull_type", "--hull-type"),
            ParamSpec("individual_meshes", "--individual-meshes", kind="bool"),
        ],
    ),
)

# ===========================================================================
# Phase 4: export terminal sinks -- cross-run BARRIER ops writing files (not copick objects). The
# type-specific ``--*-uri`` input filter self-wires to upstream producers; run selection uses the same
# repeatable ``--run-names/-r`` as the existing seg-stats barrier (no special emit path).
# ===========================================================================

_register(
    OpSpec(
        op_id="export.segmentation",
        cli_path=("export", "segmentation"),
        scope=SCOPE_BARRIER,
        vs_style=VS_URI,
        streamable=False,
        inputs=[SlotSpec("i", object_type="segmentation", role="input", flag="--segmentation-uri")],
        outputs=[],
        params=[
            ParamSpec("output_format", "--output-format", required=True),
            ParamSpec("output_dir", "--output-dir"),
            ParamSpec("level", "--level"),
            ParamSpec("compression", "--compression"),
            ParamSpec("copy_all_levels", "--copy-all-levels", kind="bool"),
        ],
    ),
)

_register(
    OpSpec(
        op_id="export.picks",
        cli_path=("export", "picks"),
        scope=SCOPE_BARRIER,
        vs_style=VS_URI,
        streamable=False,
        inputs=[SlotSpec("i", object_type="picks", role="input", flag="--picks-uri")],
        outputs=[],
        params=[
            ParamSpec("output_format", "--output-format", required=True),
            ParamSpec("output_dir", "--output-dir"),
            ParamSpec("output_file", "--output-file"),
            ParamSpec("voxel_size", "--voxel-size"),
            ParamSpec("index_map", "--index-map"),
            ParamSpec("include_optics", "--include-optics", kind="bool"),
        ],
    ),
)

_register(
    OpSpec(
        op_id="export.tomogram",
        cli_path=("export", "tomogram"),
        scope=SCOPE_BARRIER,
        vs_style=VS_URI,
        streamable=False,
        inputs=[SlotSpec("i", object_type="tomogram", role="input", flag="--tomogram-uri")],
        outputs=[],
        params=[
            ParamSpec("output_format", "--output-format", required=True),
            ParamSpec("output_dir", "--output-dir"),
            ParamSpec("level", "--level"),
            ParamSpec("compression", "--compression"),
            ParamSpec("copy_all_levels", "--copy-all-levels", kind="bool"),
        ],
    ),
)

# ===========================================================================
# cp / mv / rm for the other object types the CLI accepts (segmentation variants are above). Same
# positional shape (OBJECT_TYPE SRC [DST]); mesh/picks carry no @vs (VS_NONE), tomogram/feature do
# (VS_URI). Used directly and by the object-type-aware finalize block (expand.py).
# ===========================================================================

for _ot in ("mesh", "picks"):
    _register(
        OpSpec(
            op_id=f"core.cp-{_ot}", cli_path=("cp",), vs_style=VS_NONE, kind="cp",
            object_type_positional=_ot,
            inputs=[SlotSpec("src", object_type=_ot, role="input", positional=True, order=1)],
            outputs=[SlotSpec("dst", object_type=_ot, role="output", positional=True, order=2)],
            params=[ParamSpec("overwrite", "--overwrite", kind="bool")],
            streamable=False,
        ),
    )
    _register(
        OpSpec(
            op_id=f"core.mv-{_ot}", cli_path=("mv",), vs_style=VS_NONE, kind="mv",
            object_type_positional=_ot,
            inputs=[SlotSpec("src", object_type=_ot, role="input", positional=True, order=1)],
            outputs=[SlotSpec("dst", object_type=_ot, role="output", positional=True, order=2)],
            params=[ParamSpec("overwrite", "--overwrite", kind="bool")],
            streamable=False,
        ),
    )

for _ot, _vs in (("mesh", VS_NONE), ("picks", VS_NONE), ("tomogram", VS_URI), ("feature", VS_URI)):
    _register(
        OpSpec(
            op_id=f"core.rm-{_ot}", cli_path=("rm",), vs_style=_vs, kind="rm",
            object_type_positional=_ot,
            inputs=[SlotSpec("i", object_type=_ot, role="input", positional=True, order=1)],
            outputs=[],
            params=[ParamSpec("force", "--force", kind="bool"), ParamSpec("dry_run", "--dry-run", kind="bool")],
            streamable=False,
        ),
    )
