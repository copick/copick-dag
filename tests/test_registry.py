"""Registry lints clean against the installed copick CLI, and command validation works."""

import pytest

from copick_dag.registry import all_ops, get_op, op_ids
from copick_dag.registry.introspect import lint_registry, validate_argv

TORCH_OPS = {"convert.seg2slab", "convert.picks2slab", "process.downsample", "process.bandpass"}


def test_registry_has_example1_ops():
    ids = op_ids()
    for expected in ["logical.segop", "logical.clipseg", "logical.enclosed", "core.cp", "core.mv", "core.rm"]:
        assert expected in ids


def test_registry_lint_clean():
    issues = lint_registry()
    assert issues == [], f"registry drift vs installed CLI: {issues}"


def test_every_op_cli_path_resolves():
    from copick_dag.registry.introspect import resolve_command

    for op in all_ops().values():
        if op.op_id in TORCH_OPS:
            continue
        assert resolve_command(op.cli_path) is not None, op.op_id


def test_validate_argv_rejects_bad_operation(tmp_path):
    # a real config is needed because Click validates -c exists; content can be minimal-invalid,
    # so we only assert the *operation* choice is what fails.
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    ok, err = validate_argv(
        ["logical", "segop", "-c", str(cfg), "--operation", "banana", "-vs", "20.0", "-i", "a:b/c", "-o", "d:e/f"],
    )
    assert not ok
    assert "banana" in err or "operation" in err.lower()


def test_get_op_unknown():
    with pytest.raises(KeyError):
        get_op("logical.nope")


# ---------------------------------------------------------------------------
# M6: the 29 added transformation ops (convert / logical mesh-picks / process specials / export).
# ---------------------------------------------------------------------------

NEW_OPS = [
    "convert.seg2mesh",
    "convert.seg2picks",
    "convert.mesh2seg",
    "convert.picks2seg",
    "convert.picks2mesh",
    "convert.picks2sphere",
    "convert.picks2ellipsoid",
    "convert.picks2plane",
    "convert.picks2surface",
    "convert.mesh2caps",
    "convert.mesh2picks",
    "convert.seg2slab",
    "convert.picks2slab",
    "logical.meshop",
    "logical.clipmesh",
    "logical.clippicks",
    "logical.picksin",
    "logical.picksout",
    "process.validbox",
    "process.downsample",
    "process.bandpass",
    "process.rescale",
    "process.fit-spline",
    "process.separate-components",
    "process.skeletonize",
    "process.hull",
    "export.segmentation",
    "export.picks",
    "export.tomogram",
    "core.cp-mesh",
    "core.cp-picks",
    "core.mv-mesh",
    "core.mv-picks",
    "core.rm-mesh",
    "core.rm-picks",
    "core.rm-tomogram",
    "core.rm-feature",
]


def test_all_new_ops_registered():
    ids = set(op_ids())
    assert [o for o in NEW_OPS if o not in ids] == []


def test_new_ops_validate_argv(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    cases = [
        ["convert", "seg2mesh", "-c", str(cfg), "-r", "x", "-i", "a:b/c@10.0", "-o", "d:e/f"],
        ["convert", "mesh2seg", "-c", str(cfg), "-r", "x", "-i", "a:b/c", "--tomo-type", "wbp", "-o", "d:e/f@10.0"],
        [
            "convert",
            "mesh2picks",
            "-c",
            str(cfg),
            "-r",
            "x",
            "-i",
            "a:b/c",
            "-t",
            "wbp@10.0",
            "--sampling-type",
            "surface",
            "-o",
            "d:e/f",
        ],
        ["logical", "meshop", "-c", str(cfg), "-r", "x", "-i", "a:b/c", "--operation", "union", "-o", "d:e/f"],
        [
            "logical",
            "clipmesh",
            "-c",
            str(cfg),
            "-r",
            "x",
            "-i",
            "a:b/c",
            "-rt",
            "wbp@10.0",
            "--max-distance",
            "50",
            "-o",
            "d:e/f",
        ],
        ["process", "rescale", "-c", str(cfg), "-r", "x", "-i", "a:b/c@10.0", "-o", "a:d/e@20.0"],
        [
            "process",
            "fit-spline",
            "-c",
            str(cfg),
            "-r",
            "x",
            "-i",
            "a:b/c@10.0",
            "--spacing-distance",
            "4.0",
            "-o",
            "d:e/f",
        ],
        [
            "export",
            "segmentation",
            "-c",
            str(cfg),
            "-r",
            "x",
            "--segmentation-uri",
            "a:b/c@10.0",
            "--output-format",
            "mrc",
            "--output-dir",
            str(tmp_path),
        ],
        ["cp", "mesh", "a:b/c", "d:e/f", "-c", str(cfg), "-r", "x"],
        ["rm", "picks", "a:b/c", "-c", str(cfg), "-r", "x", "--force"],
    ]
    for argv in cases:
        ok, err = validate_argv(argv)
        assert ok, f"{argv[:2]} rejected: {err}"


def test_plugin_gating_tolerates_missing_command():
    """A plugin-gated op whose command isn't installed must NOT fail lint; a non-plugin one must."""
    from copick_dag.registry.introspect import lint_op
    from copick_dag.registry.spec import OpSpec, SlotSpec

    ghost = OpSpec(
        op_id="ghost.op",
        cli_path=("ghost", "op"),
        plugin="nonexistent_plugin",
        inputs=[SlotSpec("i", flag="-i")],
        outputs=[SlotSpec("o", role="output", flag="-o")],
    )
    assert lint_op(ghost) == []
    hard = OpSpec(op_id="ghost.op2", cli_path=("ghost", "op2"), inputs=[SlotSpec("i", flag="-i")])
    assert lint_op(hard) != []


def test_lean_install_reports_missing_torch_operation():
    from copick_dag.registry.introspect import MissingOperationDependency, operation_available, require_operation

    op = get_op("process.downsample")
    if operation_available(op):
        pytest.skip("lean-install contract is covered in the torch-free CI job")
    with pytest.raises(MissingOperationDependency, match=r"copick-dag\[torch\]"):
        require_operation(op)


@pytest.mark.torch_plugin
def test_torch_plugin_operations_resolve_and_validate(tmp_path):
    from copick_dag.registry.introspect import resolve_command

    for op_id in TORCH_OPS:
        op = get_op(op_id)
        command = resolve_command(op.cli_path)
        assert command is not None, op_id
        assert command.help or command.short_help, op_id

    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    cases = [
        ["convert", "seg2slab", "-c", str(cfg), "-r", "x", "-i", "a:b/c@10.0", "--label", "1", "-o", "a:d/e"],
        [
            "convert",
            "picks2slab",
            "-c",
            str(cfg),
            "-r",
            "x",
            "-i1",
            "a:b/c",
            "-i2",
            "a:b/d",
            "-t",
            "wbp@10.0",
            "-o",
            "a:d/e",
        ],
        ["process", "downsample", "-c", str(cfg), "-r", "x", "-i", "wbp@10.0", "-o", "wbp@20.0"],
        ["process", "bandpass", "-c", str(cfg), "-r", "x", "-i", "wbp@10.0", "-o", "filtered@10.0"],
    ]
    for argv in cases:
        ok, err = validate_argv(argv)
        assert ok, f"{argv[:2]} rejected: {err}"
