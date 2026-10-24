"""``copick-dag`` command-line interface: validate / plan / render / run a graph spec."""

from __future__ import annotations

import os
import sys
from typing import Optional

import click

from copick_dag.backends.bash import render_bash
from copick_dag.ir.model import GraphSpec
from copick_dag.planner import PlanError, ProjectView, plan
from copick_dag.registry.introspect import lint_registry, selected_operation_issues


def _load(graph: str, config: Optional[str]):
    spec = GraphSpec.from_yaml(graph)
    if config:
        spec.project.config = config
    cfg = spec.project.config
    project = None
    if cfg and "${" not in cfg and os.path.exists(cfg):
        project = ProjectView.open(cfg)
    return spec, project


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="copick-dag")
def cli() -> None:
    """Compile and run copick pipelines defined as an abstract DAG."""


@cli.command(short_help="Validate a graph spec (schema + registry + planning).")
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
@click.option("-c", "--config", default=None, help="Override the project config path.")
def validate(graph: str, config: Optional[str]) -> None:
    issues = lint_registry()
    if issues:
        click.secho("registry drift:", fg="red")
        for i in issues:
            click.echo(f"  - {i}")
        sys.exit(1)
    try:
        spec, project = _load(graph, config)
        rp = plan(spec, project, validate_preconditions=project is not None)
    except (PlanError, ValueError) as e:
        click.secho(f"INVALID: {e}", fg="red")
        sys.exit(1)
    missing = selected_operation_issues(rp)
    if missing:
        click.secho("INVALID: selected operations are unavailable:", fg="red")
        for issue in missing:
            click.echo(f"  - {issue}")
        sys.exit(1)
    click.secho(f"OK: {len(rp.tasks)} tasks, {len(spec.nodes)} nodes", fg="green")
    for w in rp.warnings:
        click.secho(f"  warning: {w}", fg="yellow")


@cli.command(short_help="Print the concrete DAG (tasks + dependencies).")
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
@click.option("-c", "--config", default=None, help="Override the project config path.")
def show(graph: str, config: Optional[str]) -> None:
    spec, project = _load(graph, config)
    rp = plan(spec, project)
    click.echo(f"# {len(rp.tasks)} tasks over runs {sorted({t.run for t in rp.tasks})}")
    for t in rp.topo_order():
        deps = ", ".join(t.deps) if t.deps else "-"
        click.echo(f"{t.task_id}  [{t.op_id}]  deps=({deps})")


@cli.command(short_help="Render the plan to a bash script or a Nextflow workflow.")
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
@click.option("-c", "--config", default=None, help="Override the project config path.")
@click.option("--backend", type=click.Choice(["bash", "nextflow"]), default="bash")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None, help="Write to a file.")
def render(graph: str, config: Optional[str], backend: str, output: Optional[str]) -> None:
    spec, project = _load(graph, config)
    rp = plan(spec, project)
    if backend == "nextflow":
        from copick_dag.backends.nextflow import render_nextflow, render_nextflow_config

        script = render_nextflow(rp)
        if output:
            with open(output, "w") as fh:
                fh.write(script)
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(output)) or ".", "nextflow.config")
            with open(cfg_path, "w") as fh:
                fh.write(render_nextflow_config())
            click.secho(f"wrote {output} and {cfg_path}", fg="green")
        else:
            click.echo(script)
        return

    script = render_bash(rp)
    if output:
        with open(output, "w") as fh:
            fh.write(script)
        click.secho(f"wrote {output}", fg="green")
    else:
        click.echo(script)


@cli.command(short_help="Execute the plan (in-memory, bash, or nextflow).")
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
@click.option("-c", "--config", default=None, help="Project config path (required unless the spec has a concrete one).")
@click.option("--backend", type=click.Choice(["memory", "bash", "nextflow"]), default="memory")
@click.option(
    "-w",
    "--workers",
    type=int,
    default=1,
    help="Runs to execute concurrently (memory backend; >1 isolates a root per run).",
)
@click.option("--profile", default="standard", help="Nextflow profile (standard|slurm).")
@click.option("--resume", is_flag=True, default=False, help="Pass -resume to nextflow (crash-recovery only).")
def run(graph: str, config: Optional[str], backend: str, workers: int, profile: str, resume: bool) -> None:
    spec, project = _load(graph, config)
    if project is None:
        raise click.ClickException("a real project config is required to run; pass -c/--config")
    rp = plan(spec, project, validate_preconditions=True)
    for w in rp.warnings:
        click.secho(f"warning: {w}", fg="yellow")

    if backend == "bash":
        script = render_bash(rp)
        import subprocess

        proc = subprocess.run(["bash", "-c", script])
        sys.exit(proc.returncode)

    if backend == "nextflow":
        import shutil
        import subprocess
        import tempfile

        if not shutil.which("nextflow"):
            raise click.ClickException(
                "nextflow not found on PATH; install from https://get.nextflow.io, "
                "or use --backend memory / --backend bash.",
            )
        from copick_dag.backends.nextflow import render_nextflow, render_nextflow_config

        workdir = tempfile.mkdtemp(prefix="copick_dag_nf_")
        nf_path = os.path.join(workdir, "pipeline.nf")
        with open(nf_path, "w") as fh:
            fh.write(render_nextflow(rp))
        with open(os.path.join(workdir, "nextflow.config"), "w") as fh:
            fh.write(render_nextflow_config())
        cmd = ["nextflow", "run", nf_path, "-profile", profile]
        if resume:
            cmd.append("-resume")
        click.secho(f"running: {' '.join(cmd)} (in {workdir})", fg="green")
        proc = subprocess.run(cmd, cwd=workdir)
        sys.exit(proc.returncode)

    from copick_dag.backends.memory import run_memory

    stats = run_memory(rp, workers=workers, root=project.root)
    click.secho(
        f"done: {len(stats.runs)} runs, cache hits={stats.total_hits} misses={stats.total_misses} "
        f"writes={stats.total_writes}",
        fg="green",
    )
    if stats.errors:
        for e in stats.errors:
            click.secho(f"  {e}", fg="yellow")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
