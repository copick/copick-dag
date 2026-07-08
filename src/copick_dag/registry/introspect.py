"""Auto-layer: introspect the *installed* copick Click CLI and validate the overlay against it.

Adapted from the logic in ``copick-mcp``'s ``cli_introspection`` but kept local so copick-dag does
not depend on copick-mcp. Uses ``copick.cli.cli.add_core_commands`` / ``add_plugin_commands`` to
assemble the full command tree, then navigates a registry ``cli_path`` to the concrete command.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Sequence, Set, Tuple

import click

from copick_dag.registry.overlay import OPS
from copick_dag.registry.spec import OpSpec


@functools.lru_cache(maxsize=1)
def build_cli() -> click.Group:
    """Build the full copick CLI (core + plugin commands). Cached."""
    from copick.cli.cli import add_core_commands, add_plugin_commands

    @click.group()
    def cli() -> None:  # pragma: no cover - trivial
        pass

    cli = add_core_commands(cli)
    cli = add_plugin_commands(cli)
    return cli


def resolve_command(cli_path: Sequence[str]) -> Optional[click.Command]:
    """Navigate ``cli_path`` (e.g. ``("logical", "segop")``) to a concrete Click command."""
    cmd: click.Command = build_cli()
    for part in cli_path:
        if not isinstance(cmd, click.Group) or part not in cmd.commands:
            return None
        cmd = cmd.commands[part]
    return cmd


def command_flags(cmd: click.Command) -> Set[str]:
    """All real option flag strings of a command (ignoring optgroup ``fake_*`` separators)."""
    flags: Set[str] = set()
    for prm in cmd.params:
        opts = getattr(prm, "opts", None) or []
        if isinstance(prm, click.Option):
            flags.update(opts)
    return flags


def command_positionals(cmd: click.Command) -> List[str]:
    return [prm.name for prm in cmd.params if isinstance(prm, click.Argument)]


def lint_op(op: OpSpec) -> List[str]:
    """Return a list of drift issues for a single op (empty == clean)."""
    issues: List[str] = []
    cmd = resolve_command(op.cli_path)
    if cmd is None:
        if op.plugin is not None:
            # A plugin-provided op (e.g. copick_torch) whose command is not installed in this env:
            # tolerated, so copick-dag stays lint-clean on a lean install without that plugin.
            return []
        return [f"{op.op_id}: CLI path {op.cli_path} does not resolve to an installed command"]

    flags = command_flags(cmd)
    positionals = command_positionals(cmd)

    # every non-positional slot flag must exist (declared/edges-only slots are not emitted)
    for slot in [*op.inputs, *op.outputs]:
        if slot.positional or slot.declared:
            continue
        if slot.flag is None:
            issues.append(f"{op.op_id}: slot {slot.slot!r} is non-positional but has no flag")
        elif slot.flag not in flags:
            issues.append(f"{op.op_id}: slot {slot.slot!r} flag {slot.flag!r} not in command flags {sorted(flags)}")

    # every param flag must exist
    for p in op.params:
        if p.flag not in flags:
            issues.append(f"{op.op_id}: param {p.name!r} flag {p.flag!r} not in command flags {sorted(flags)}")

    # positional slots must line up with the command's arguments (object_type + slots)
    n_positional_slots = sum(1 for s in [*op.inputs, *op.outputs] if s.positional)
    expected_positionals = n_positional_slots + (1 if op.object_type_positional else 0)
    if op.object_type_positional or n_positional_slots:
        if len(positionals) != expected_positionals:
            issues.append(
                f"{op.op_id}: expected {expected_positionals} positional args "
                f"(object_type={bool(op.object_type_positional)} + {n_positional_slots} slots), "
                f"command has {len(positionals)}: {positionals}",
            )
    return issues


def lint_registry() -> List[str]:
    """Lint every op in the overlay against the installed CLI. Returns all issues."""
    issues: List[str] = []
    for op in OPS.values():
        issues.extend(lint_op(op))
    return issues


def validate_argv(argv: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """Validate an emitted ``copick …`` command (argv WITHOUT the leading ``copick``).

    Returns ``(ok, error_message)``. Uses Click's own parsing so option/URI validation matches the
    real CLI.
    """
    cli = build_cli()
    ctx = click.Context(cli)
    try:
        args = list(argv)
        cmd_name, cmd, rest = cli.resolve_command(ctx, args)
        if isinstance(cmd, click.Group) and rest:
            sub_ctx = click.Context(cmd, parent=ctx)
            sub_name, sub_cmd, sub_rest = cmd.resolve_command(sub_ctx, rest)
            final_ctx = click.Context(sub_cmd, parent=sub_ctx)
            sub_cmd.parse_args(final_ctx, list(sub_rest))
        else:
            final_ctx = click.Context(cmd, parent=ctx)
            cmd.parse_args(final_ctx, list(rest))
        return True, None
    except click.ClickException as e:
        return False, e.format_message()
    except Exception as e:  # pragma: no cover - defensive
        return False, str(e)
