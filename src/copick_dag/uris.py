"""URI helpers for copick-dag.

Thin wrappers over :mod:`copick.util.uri` that add the operations the planner and emitters need but
copick's project-resolution API does not provide: the ``@voxel_spacing`` suffix manipulation the
emitters need, ``${var}`` substitution, and -- the real reason this module exists -- the *abstract*
producer<->consumer URI overlap used for offline edge inference (copick can only match a concrete
value against a pattern, or resolve a URI against a live project).

Every copick object type is addressed differently (picks/mesh carry no voxel spacing; segmentation,
tomogram, and feature do). We parse each URI *by its object type* via ``parse_copick_uri`` and match
component-wise: glob/regex per identity component AND numeric match on ``voxel_spacing`` for the
vs-carrying types. For segmentations this reproduces the historical ``SegURI`` behaviour exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from copick.util.uri import _matches_numeric_pattern, _matches_pattern, parse_copick_uri

_VAR_RE = re.compile(r"\$\{(\w+)\}")

# identity components (excluding voxel spacing) in the order they are matched, per object type.
_IDENTITY_KEYS: Dict[str, Tuple[str, ...]] = {
    "picks": ("object_name", "user_id", "session_id"),
    "mesh": ("object_name", "user_id", "session_id"),
    "segmentation": ("name", "user_id", "session_id"),
    "tomogram": ("tomo_type",),
    "feature": ("tomo_type", "feature_type"),
}
# object types whose URIs carry an ``@voxel_spacing`` component.
_VS_TYPES = frozenset({"segmentation", "tomogram", "feature"})


def substitute(text: str, mapping: Dict[str, object]) -> str:
    """Replace ``${var}`` placeholders in ``text`` using ``mapping``.

    A missing variable is left untouched (so a literal URI with no placeholders passes through).
    """

    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key in mapping:
            return _fmt(mapping[key])
        return m.group(0)

    return _VAR_RE.sub(repl, text)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        # avoid "20.0" -> keep as-is; copick voxel spacings are floats, keep precision
        return repr(value) if value != int(value) else str(value)
    return str(value)


def is_pattern(uri: str) -> bool:
    """True if the URI contains a glob/regex wildcard (``*``, ``?``, ``[``, an ``re:`` prefix, or a
    ``{placeholder}`` fan-out template). A trailing ``?query`` (e.g. ``?multilabel=true``) is ignored
    so it is never mistaken for a ``?`` glob."""
    body = uri.split("?", 1)[0]
    if body.startswith("re:"):
        return True
    return any(ch in body for ch in "*?[{")


# ---------------------------------------------------------------------------
# @voxel_spacing suffix manipulation (query-safe: a trailing ?query is preserved)
# ---------------------------------------------------------------------------


def split_vs(uri: str) -> tuple[str, Optional[str]]:
    """Split a URI into ``(uri_without_vs, vs_str_or_None)``.

    Only the last ``@`` *before* any ``?query`` is the voxel-spacing delimiter (object names never
    contain ``@``). The query, if present, stays attached to the returned base so round-trips are
    lossless -- this is what keeps a ``…@10?multilabel=true`` URI from corrupting the vs.
    """
    body, sep, query = uri.partition("?")
    if "@" not in body:
        return uri, None
    base, vs = body.rsplit("@", 1)
    return base + sep + query, vs


def get_vs(uri: str) -> Optional[str]:
    return split_vs(uri)[1]


def with_vs(uri: str, vs: object) -> str:
    """Ensure the URI carries an ``@voxel_spacing`` suffix, adding ``vs`` if absent (inserted before
    any ``?query``). Only meaningful for vs-carrying types; the planner gates the call by object type.
    """
    body, sep, query = uri.partition("?")
    if "@" in body:
        base, existing = body.rsplit("@", 1)
    else:
        base, existing = body, None
    if existing is not None and existing != "*":
        return uri
    return f"{base}@{_fmt(vs)}{sep}{query}"


def strip_vs(uri: str) -> str:
    """Return the URI without its ``@voxel_spacing`` suffix (for ``-vs``-flag style commands)."""
    body, sep, query = uri.partition("?")
    if "@" not in body:
        return uri
    return body.rsplit("@", 1)[0] + sep + query


# ---------------------------------------------------------------------------
# parsed URIs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjURI:
    """A parsed copick URI of any object type (components may be concrete or glob/regex patterns)."""

    raw: str
    object_type: str
    pattern_type: str  # "glob" | "regex"
    values: Tuple[str, ...]  # identity components in match order (excludes voxel spacing)
    voxel_spacing: Optional[str]  # None if the type carries no vs; else value or "*"

    @classmethod
    def parse(cls, uri: str, object_type: str) -> "ObjURI":
        p = parse_copick_uri(uri, object_type)
        keys = _IDENTITY_KEYS[object_type]
        values = tuple((p.get(k) or "*") for k in keys)
        vs = p.get("voxel_spacing") if object_type in _VS_TYPES else None
        return cls(uri, object_type, p.get("pattern_type", "glob"), values, vs)

    @property
    def has_vs(self) -> bool:
        return self.object_type in _VS_TYPES

    def is_concrete(self) -> bool:
        if any(is_pattern(v) for v in self.values):
            return False
        return not (self.has_vs and self.voxel_spacing in (None, "*"))


@dataclass(frozen=True)
class SegURI:
    """A parsed *segmentation* URI. Retained as a thin compatibility shim for the in-memory backend
    and the planner's precondition check, which are segmentation-specific by design."""

    raw: str
    name: str
    user_id: str
    session_id: str
    voxel_spacing: Optional[str]
    pattern_type: str  # "glob" | "regex"

    @classmethod
    def parse(cls, uri: str) -> "SegURI":
        p = parse_copick_uri(uri, "segmentation")
        return cls(
            raw=uri,
            name=p["name"],
            user_id=p["user_id"],
            session_id=p["session_id"],
            voxel_spacing=p.get("voxel_spacing"),
            pattern_type=p.get("pattern_type", "glob"),
        )

    def is_concrete(self) -> bool:
        return ObjURI.parse(self.raw, "segmentation").is_concrete()


# ---------------------------------------------------------------------------
# component-wise matching (object-type aware; default segmentation preserves call sites)
# ---------------------------------------------------------------------------


def seg_matches(producer_uri: str, consumer_pattern: str, object_type: str = "segmentation") -> bool:
    """Component-wise match: does concrete ``producer_uri`` satisfy ``consumer_pattern``?

    The consumer supplies the pattern-type (glob/regex); the producer supplies concrete values. Voxel
    spacing is compared numerically (only for vs-carrying types)."""
    prod = ObjURI.parse(producer_uri, object_type)
    cons = ObjURI.parse(consumer_pattern, object_type)
    pt = cons.pattern_type

    for pv, cv in zip(prod.values, cons.values):
        if not _matches_pattern(pv or "", cv or "*", pt):
            return False

    if not cons.has_vs:
        return True
    cons_vs = cons.voxel_spacing
    if cons_vs in (None, "*"):
        return True
    prod_vs = prod.voxel_spacing
    if prod_vs in (None, "*"):
        return False
    return _matches_numeric_pattern(prod_vs, cons_vs, pt)


def same_uri(a: str, b: str, object_type: str = "segmentation") -> bool:
    """True if two URIs address the *identical* concrete object (vs compared as float)."""
    sa, sb = ObjURI.parse(a, object_type), ObjURI.parse(b, object_type)
    if sa.values != sb.values:
        return False
    if not sa.has_vs:
        return True
    if sa.voxel_spacing in (None, "*") or sb.voxel_spacing in (None, "*"):
        return sa.voxel_spacing == sb.voxel_spacing
    try:
        return abs(float(sa.voxel_spacing) - float(sb.voxel_spacing)) < 1e-6
    except (TypeError, ValueError):
        return sa.voxel_spacing == sb.voxel_spacing


def is_concrete_uri(uri: str, object_type: str = "segmentation") -> bool:
    return ObjURI.parse(uri, object_type).is_concrete()


def _component_overlap(a: Optional[str], b: Optional[str], a_pt: str, b_pt: str) -> bool:
    """Do two URI components (each possibly a glob/regex) admit a common concrete value?"""
    if a in (None, "*") or b in (None, "*"):
        return True
    a_pat, b_pat = is_pattern(a), is_pattern(b)
    if not a_pat and not b_pat:
        return a == b
    if a_pat and not b_pat:
        return _matches_pattern(b, a, a_pt)
    if b_pat and not a_pat:
        return _matches_pattern(a, b, b_pt)
    # both partial patterns -- undecidable in general; M3 restricts fan-out wildcards to a full
    # '*' on a single component, so this branch is not exercised. Be conservative (assume overlap).
    return True


def _vs_overlap(a: Optional[str], b: Optional[str]) -> bool:
    if a in (None, "*") or b in (None, "*"):
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a == b


def patterns_overlap(a_uri: str, b_uri: str, object_type: str = "segmentation") -> bool:
    """Symmetric edge test: can two (possibly wildcard) URIs match a common object?

    Unlike :func:`seg_matches` (which assumes a concrete producer), this is order-independent, so a
    wildcard *producer* (e.g. split's declared ``*:split/final``) correctly matches a *concrete*
    consumer, and vice versa. Both URIs are parsed as ``object_type`` (edges only connect same-type
    producer->consumer)."""
    a = ObjURI.parse(a_uri, object_type)
    b = ObjURI.parse(b_uri, object_type)
    for av, bv in zip(a.values, b.values):
        if not _component_overlap(av, bv, a.pattern_type, b.pattern_type):
            return False
    if a.has_vs and not _vs_overlap(a.voxel_spacing, b.voxel_spacing):
        return False
    return True
