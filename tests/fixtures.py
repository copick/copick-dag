"""Build a small, self-contained copick filesystem project for tests.

Creates the leaf segmentations ``example_1`` consumes (membrane from two ``data-portal`` users,
plus cell/cytosole/flagellum/inclusion under ``napari/manual-scaled``) at voxel spacing 20.0, with
deterministic geometric content so operations produce stable, non-trivial results.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

VS = 20.0
SHAPE = (24, 24, 24)

# object name -> (is_particle, label)
OBJECTS: Dict[str, Tuple[bool, int]] = {
    "membrane": (False, 1),
    "cell": (False, 2),
    "cytosole": (False, 3),
    "flagellum": (False, 4),
    "inclusion": (False, 5),
    "intermembrane-space": (False, 6),
}


def _box(shape, z, y, x) -> np.ndarray:
    a = np.zeros(shape, dtype=np.uint8)
    a[z[0] : z[1], y[0] : y[1], x[0] : x[1]] = 1
    return a


def _leaf_arrays() -> Dict[str, np.ndarray]:
    """Deterministic leaf volumes. Keys are ``name|user|session``."""
    cell = _box(SHAPE, (4, 20), (4, 20), (4, 20))
    cytosole = _box(SHAPE, (7, 17), (7, 17), (7, 17))
    # two membrane annotations near the cell surface (overlapping slabs -> union is meaningful)
    mem1 = _box(SHAPE, (4, 20), (4, 20), (4, 6))
    mem2 = _box(SHAPE, (4, 20), (4, 6), (4, 20))
    flagellum = _box(SHAPE, (2, 6), (10, 14), (10, 14))
    inclusion = _box(SHAPE, (9, 12), (9, 12), (9, 12))
    return {
        "membrane|data-portal|auto-1": mem1,
        "membrane|data-portal|auto-2": mem2,
        "cell|napari|manual-scaled": cell,
        "cytosole|napari|manual-scaled": cytosole,
        "flagellum|napari|manual-scaled": flagellum,
        "inclusion|napari|manual-scaled": inclusion,
    }


def make_fixture(base_dir: str, runs: List[str] = ("run1", "run2")) -> str:
    """Create a copick project under ``base_dir`` and return the config path."""
    import copick
    from copick.models import PickableObject

    os.makedirs(base_dir, exist_ok=True)
    overlay_root = os.path.join(base_dir, "overlay")
    os.makedirs(overlay_root, exist_ok=True)
    config_path = os.path.join(base_dir, "config.json")

    pickable = [
        PickableObject(name=name, is_particle=is_particle, label=label)
        for name, (is_particle, label) in OBJECTS.items()
    ]
    # ``new_config`` prepends the ``local://`` protocol itself; pass a bare absolute path so the
    # stored ``overlay_root`` is a single, CWD-independent URL. (Pre-prefixing here yields a doubled
    # ``local://local://…`` that fsspec resolves relative to the process CWD -- which silently
    # fragments the store across the per-task working directories the nextflow backend runs in.)
    root = copick.new_config(
        config=config_path,
        overlay_root=overlay_root,
        proj_name="copick-dag test",
        pickable_objects=pickable,
    )

    arrays = _leaf_arrays()
    for run_name in runs:
        run = root.new_run(run_name, exist_ok=True)
        for key, arr in arrays.items():
            name, user, session = key.split("|")
            seg = run.new_segmentation(
                voxel_size=VS,
                name=name,
                session_id=session,
                is_multilabel=False,
                user_id=user,
                exist_ok=True,
            )
            seg.from_numpy(arr)
    return config_path


# ---------------------------------------------------------------------------
# M3a fixture: a `multilabel:pipeline/final` per run (one WITH microtubules, one WITHOUT)
# plus a `sample:rescale/0` leaf, for the dynamic-frontier (example_3) tests.
# ---------------------------------------------------------------------------

M3A_VS = 10.0
M3A_OBJECTS = {  # name -> (is_particle, label)
    "membrane": (False, 1),
    "cytoplasm": (False, 2),
    "microtubules": (False, 3),
    "vesicle": (False, 5),  # always present -> `*:filt` always has >=2 members (like real example_3)
    "sample": (False, 4),
}
M3A_RUNS = ["run_mt", "run_nomt"]


def _multilabel(with_mt: bool) -> np.ndarray:
    a = np.zeros(SHAPE, dtype=np.uint8)
    a[4:8, 4:20, 4:20] = 1  # membrane: a wall at low x
    a[12:20, 12:20, 12:20] = 2  # cytoplasm: interior box, away from membrane
    a[16:20, 4:8, 4:8] = 5  # vesicle: a corner blob, present in every run
    if with_mt:
        a[5:7, 5:9, 5:9] = 3  # microtubules: adjacent to the membrane wall
    return a


def make_m3a_fixture(base_dir: str) -> str:
    import copick
    from copick.models import PickableObject

    os.makedirs(base_dir, exist_ok=True)
    overlay_root = os.path.join(base_dir, "overlay")
    os.makedirs(overlay_root, exist_ok=True)
    config_path = os.path.join(base_dir, "config.json")

    pickable = [
        PickableObject(name=name, is_particle=is_particle, label=label)
        for name, (is_particle, label) in M3A_OBJECTS.items()
    ]
    # bare absolute path: new_config adds a single ``local://`` (see make_fixture note above).
    root = copick.new_config(
        config=config_path, overlay_root=overlay_root, proj_name="copick-dag m3a test", pickable_objects=pickable
    )

    sample = np.zeros(SHAPE, dtype=np.uint8)
    sample[2:22, 2:22, 2:22] = 1
    for run_name in M3A_RUNS:
        run = root.new_run(run_name, exist_ok=True)
        ml = run.new_segmentation(
            voxel_size=M3A_VS,
            name="multilabel",
            session_id="final",
            is_multilabel=True,
            user_id="pipeline",
            exist_ok=True,
        )
        ml.from_numpy(_multilabel(with_mt=(run_name == "run_mt")))
        sm = run.new_segmentation(
            voxel_size=M3A_VS, name="sample", session_id="0", is_multilabel=False, user_id="rescale", exist_ok=True
        )
        sm.from_numpy(sample)
    return config_path


def make_m3a_csv(path: str) -> str:
    with open(path, "w") as fh:
        fh.write("membrane,3\ncytoplasm,5\nmicrotubules,2\nvesicle,2\n")
    return path


def read_segmentations(
    config_path: str, runs: List[str], user_id: str, session: str, vs: float
) -> Dict[str, np.ndarray]:
    """Read every ``*:<user_id>/<session>@vs`` segmentation, keyed ``run|name``."""
    import copick

    root = copick.from_file(config_path)
    out: Dict[str, np.ndarray] = {}
    for run_name in runs:
        run = root.get_run(run_name)
        if run is None:
            continue
        for seg in run.get_segmentations(user_id=user_id, session_id=session, voxel_size=vs):
            out[f"{run_name}|{seg.name}"] = seg.numpy()
    return out


# ---------------------------------------------------------------------------
# Cross-type fixtures (M6): a tomogram per run so mesh/picks/tomogram ops have a grid reference.
# ---------------------------------------------------------------------------


def make_tomogram(run, tomo_type: str = "wbp", vs: float = VS, shape=SHAPE):
    """Attach a (zero-filled) tomogram of ``tomo_type`` at spacing ``vs`` to ``run``."""
    vso = run.new_voxel_spacing(voxel_size=vs, exist_ok=True)
    tomo = vso.new_tomogram(tomo_type, exist_ok=True)
    tomo.from_numpy(np.zeros(shape, dtype=np.float32))
    return tomo


def make_crosstype_fixture(base_dir: str, runs: List[str] = ("run1", "run2"), tomo_type: str = "wbp") -> str:
    """``make_fixture`` (segmentations) + a tomogram per run -- enough of a grid for the pipeline to
    *generate* meshes/picks via convert ops (preferred over hand-seeding mesh/picks leaves)."""
    import copick

    config_path = make_fixture(base_dir, list(runs))
    root = copick.from_file(config_path)
    for run_name in runs:
        make_tomogram(root.get_run(run_name), tomo_type=tomo_type, vs=VS, shape=SHAPE)
    return config_path


def read_meshes(config_path: str, runs: List[str], user_id: str, session: str) -> Dict[str, object]:
    """Read every ``*:<user_id>/<session>`` mesh, keyed ``run|object_name``."""
    import copick

    root = copick.from_file(config_path)
    out: Dict[str, object] = {}
    for run_name in runs:
        run = root.get_run(run_name)
        if run is None:
            continue
        for m in run.get_meshes(user_id=user_id, session_id=session):
            out[f"{run_name}|{m.pickable_object_name}"] = m
    return out


def read_picks(config_path: str, runs: List[str], user_id: str, session: str) -> Dict[str, object]:
    """Read every ``*:<user_id>/<session>`` pick set, keyed ``run|object_name``."""
    import copick

    root = copick.from_file(config_path)
    out: Dict[str, object] = {}
    for run_name in runs:
        run = root.get_run(run_name)
        if run is None:
            continue
        for p in run.get_picks(user_id=user_id, session_id=session):
            out[f"{run_name}|{p.pickable_object_name}"] = p
    return out


def read_final_segmentations(
    config_path: str, runs: List[str], session: str = "final-training"
) -> Dict[str, np.ndarray]:
    """Read every ``*:napari/<session>@20`` segmentation into a dict keyed ``run|name``."""
    import copick

    root = copick.from_file(config_path)
    out: Dict[str, np.ndarray] = {}
    for run_name in runs:
        run = root.get_run(run_name)
        if run is None:
            continue
        for seg in run.get_segmentations(user_id="napari", session_id=session, voxel_size=VS):
            out[f"{run_name}|{seg.name}"] = seg.numpy()
    return out
