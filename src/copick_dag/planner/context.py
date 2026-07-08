"""A thin read-only view over a copick project used by the planner.

Opens ``copick.from_file(config)`` exactly once and answers the queries the planner needs:
the run-set, the voxel spacings actually present (per run or across the project), and whether a
given segmentation already exists (for precondition checks).
"""

from __future__ import annotations

from typing import List, Optional

from copick_dag.uris import SegURI


class ProjectView:
    def __init__(self, root, config: Optional[str] = None):
        self.root = root
        self.config = config
        self._run_names: Optional[List[str]] = None

    @classmethod
    def open(cls, config: str) -> "ProjectView":
        import copick

        return cls(copick.from_file(config), config=config)

    # -- run axis ----------------------------------------------------------

    def run_names(self) -> List[str]:
        if self._run_names is None:
            self._run_names = [r.name for r in self.root.runs]
        return list(self._run_names)

    # -- voxel-spacing axis ------------------------------------------------

    def voxel_spacings(self, run_name: str) -> List[float]:
        run = self.root.get_run(run_name)
        if run is None:
            return []
        return sorted(float(vs.voxel_size) for vs in run.voxel_spacings)

    def all_voxel_spacings(self) -> List[float]:
        present = set()
        for name in self.run_names():
            present.update(self.voxel_spacings(name))
        return sorted(present)

    # -- existence checks --------------------------------------------------

    def segmentation_exists(self, uri: str, run_name: str) -> bool:
        """True if at least one segmentation matching ``uri`` exists in ``run_name``."""
        run = self.root.get_run(run_name)
        if run is None:
            return False
        s = SegURI.parse(uri)
        vs = None
        if s.voxel_spacing not in (None, "*"):
            try:
                vs = float(s.voxel_spacing)
            except (TypeError, ValueError):
                vs = None
        # Only use direct lookups for fully-concrete component values; patterns fall back to listing.
        kwargs = {}
        if s.name and "*" not in s.name and not s.name.startswith("re:"):
            kwargs["name"] = s.name
        if s.user_id and "*" not in s.user_id:
            kwargs["user_id"] = s.user_id
        if s.session_id and "*" not in s.session_id:
            kwargs["session_id"] = s.session_id
        if vs is not None:
            kwargs["voxel_size"] = vs
        try:
            segs = run.get_segmentations(**kwargs)
        except Exception:
            return False
        return len(segs) > 0

    def object_exists(self, uri: str, object_type: str, run_name: str) -> bool:
        """True if at least one object of ``object_type`` matching ``uri`` exists in ``run_name``.

        Segmentations keep the original fast path (byte-identical precondition behaviour); other
        object types dispatch to copick's own object-type-generic resolver."""
        if object_type == "segmentation":
            return self.segmentation_exists(uri, run_name)
        from copick.util.uri import resolve_copick_objects

        if self.root.get_run(run_name) is None:
            return False
        try:
            objs = resolve_copick_objects(uri, self.root, object_type, run_name=run_name)
        except Exception:
            return False
        return len(objs) > 0
