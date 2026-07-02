"""Registry lints clean against the installed copick CLI, and command validation works."""

from copick_dag.registry import all_ops, get_op, op_ids
from copick_dag.registry.introspect import lint_registry, validate_argv


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
    import pytest

    with pytest.raises(KeyError):
        get_op("logical.nope")
