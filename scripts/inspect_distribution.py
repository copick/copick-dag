"""Fail release builds whose artifacts do not expose the copick-dag 1.x contract."""

from __future__ import annotations

import argparse
import email
import tarfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

EXPECTED_CONSOLE_SCRIPT = "copick-dag = copick_dag.cli:main"
EXPECTED_RUNTIME_REQUIREMENTS = {
    "copick": ">=1.23.1,<2",
    "copick-utils": ">=1.7.0,<2",
}


def _requirements(metadata: email.message.Message) -> list[Requirement]:
    return [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]


def _requirement_for(requirements: list[Requirement], name: str, *, extra: str | None = None) -> Requirement:
    matches = [
        requirement
        for requirement in requirements
        if requirement.name == name
        and ((extra is None and requirement.marker is None) or (extra and extra in str(requirement.marker)))
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name!r} requirement for extra {extra!r}, found {matches!r}")
    return matches[0]


def inspect_distributions(dist_dir: Path) -> tuple[Path, Path]:
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"Expected one wheel and one source distribution, found {wheels!r} and {sdists!r}")

    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        entry_point_names = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
        if len(metadata_names) != 1 or len(entry_point_names) != 1:
            raise ValueError("Wheel must contain exactly one METADATA and one entry_points.txt file")
        if not any(name == "copick_dag/__init__.py" for name in archive.namelist()):
            raise ValueError("Wheel does not contain the copick_dag package")

        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        if metadata["Requires-Python"] != ">=3.10":
            raise ValueError(f"Unexpected Requires-Python: {metadata['Requires-Python']!r}")

        requirements = _requirements(metadata)
        for name, expected_specifier in EXPECTED_RUNTIME_REQUIREMENTS.items():
            requirement = _requirement_for(requirements, name)
            if requirement.specifier != SpecifierSet(expected_specifier):
                raise ValueError(f"Unexpected {name} requirement: {requirement}")

        runtime_names = {requirement.name for requirement in requirements if requirement.marker is None}
        forbidden = runtime_names & {"zarr", "numcodecs", "copick-torch"}
        if forbidden:
            raise ValueError(f"Lean runtime unexpectedly depends on {sorted(forbidden)}")

        for extra in ("torch", "all"):
            requirement = _requirement_for(requirements, "copick-torch", extra=extra)
            if requirement.specifier != SpecifierSet(">=1.2.0,<2"):
                raise ValueError(f"Unexpected copick-torch requirement for {extra}: {requirement}")

        entry_points = archive.read(entry_point_names[0]).decode()
        if EXPECTED_CONSOLE_SCRIPT not in entry_points:
            raise ValueError(f"Wheel is missing console script {EXPECTED_CONSOLE_SCRIPT!r}")

    sdist = sdists[0]
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        required_suffixes = (
            "/uv.lock",
            "/README.md",
            "/CHANGELOG.md",
            "/examples/example_1.yaml",
            "/src/copick_dag/__init__.py",
        )
        missing = [suffix for suffix in required_suffixes if not any(name.endswith(suffix) for name in names)]
        if missing:
            raise ValueError(f"Source distribution is missing required files: {missing}")

    return wheel, sdist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    wheel, sdist = inspect_distributions(args.dist_dir)
    print(f"Validated {wheel.name} and {sdist.name}")


if __name__ == "__main__":
    main()
