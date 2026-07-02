"""URI helpers for copick-dag.

Thin wrappers over :mod:`copick.util.uri` that add the operations the planner and emitters need:
splitting/attaching the ``@voxel_spacing`` suffix, ``${var}`` substitution, pattern detection, and
the component-wise producer->consumer match used for edge inference.

We match segmentation URIs by their four components ``(name, user_id, session_id, voxel_spacing)``.
Voxel spacing is compared numerically (copick uses *exact* float equality); everything else uses
copick's glob/regex matcher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from copick.util.uri import _matches_numeric_pattern, _matches_pattern, parse_copick_uri

_VAR_RE = re.compile(r"\$\{(\w+)\}")


def substitute(text: str, mapping: Dict[str, object]) -> str:
    """Replace ``${var}`` placeholders in ``text`` using ``mapping``.

    A missing variable is left untouched (so a literal URI with no placeholders passes through).
    Floats that are integer-valued are rendered without a trailing ``.0`` only when the mapping
    value is already a string; numeric values are rendered with ``repr``-free ``str``.
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
    """True if the URI contains a glob/regex wildcard (``*``, ``?``, ``[``, or an ``re:`` prefix)."""
    body = uri
    if body.startswith("re:"):
        return True
    return any(ch in body for ch in "*?[")


def split_vs(uri: str) -> tuple[str, Optional[str]]:
    """Split a segmentation URI into ``(uri_without_vs, vs_str_or_None)``.

    Only the *last* ``@`` is treated as the voxel-spacing delimiter (object names never contain
    ``@``). A trailing ``?multilabel=...`` query is preserved on the vs part.
    """
    if "@" not in uri:
        return uri, None
    base, vs = uri.rsplit("@", 1)
    return base, vs


def get_vs(uri: str) -> Optional[str]:
    return split_vs(uri)[1]


def with_vs(uri: str, vs: object) -> str:
    """Ensure the URI carries an ``@voxel_spacing`` suffix, adding ``vs`` if absent."""
    base, existing = split_vs(uri)
    if existing is not None and existing != "*":
        return uri
    return f"{base}@{_fmt(vs)}"


def strip_vs(uri: str) -> str:
    """Return the URI without its ``@voxel_spacing`` suffix (for ``-vs``-flag style commands)."""
    return split_vs(uri)[0]


@dataclass(frozen=True)
class SegURI:
    """A parsed segmentation URI (components may be concrete or glob/regex patterns)."""

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
        return not (
            is_pattern(self.name)
            or is_pattern(self.user_id or "")
            or is_pattern(self.session_id or "")
            or (self.voxel_spacing in (None, "*"))
        )


def seg_matches(producer_uri: str, consumer_pattern: str) -> bool:
    """Component-wise match: does concrete ``producer_uri`` satisfy ``consumer_pattern``?

    Used for edge inference. ``consumer_pattern`` supplies the pattern-type (glob/regex); the
    producer supplies concrete values. Voxel spacing is compared numerically.
    """
    prod = SegURI.parse(producer_uri)
    cons = SegURI.parse(consumer_pattern)
    pt = cons.pattern_type

    if not _matches_pattern(prod.name or "", cons.name or "*", pt):
        return False
    if not _matches_pattern(prod.user_id or "", cons.user_id or "*", pt):
        return False
    if not _matches_pattern(prod.session_id or "", cons.session_id or "*", pt):
        return False

    cons_vs = cons.voxel_spacing
    if cons_vs in (None, "*"):
        return True
    prod_vs = prod.voxel_spacing
    if prod_vs in (None, "*"):
        return False
    return _matches_numeric_pattern(prod_vs, cons_vs, pt)


def same_uri(a: str, b: str) -> bool:
    """True if two segmentation URIs address the *identical* concrete object (vs compared as float)."""
    sa, sb = SegURI.parse(a), SegURI.parse(b)
    if (sa.name, sa.user_id, sa.session_id) != (sb.name, sb.user_id, sb.session_id):
        return False
    if sa.voxel_spacing in (None, "*") or sb.voxel_spacing in (None, "*"):
        return sa.voxel_spacing == sb.voxel_spacing
    try:
        return abs(float(sa.voxel_spacing) - float(sb.voxel_spacing)) < 1e-6
    except (TypeError, ValueError):
        return sa.voxel_spacing == sb.voxel_spacing


def is_concrete_uri(uri: str) -> bool:
    return SegURI.parse(uri).is_concrete()


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


def patterns_overlap(a_uri: str, b_uri: str) -> bool:
    """Symmetric edge test: can two (possibly wildcard) segmentation URIs match a common object?

    Unlike :func:`seg_matches` (which assumes a concrete producer), this is order-independent, so a
    wildcard *producer* (e.g. split's declared ``*:split/final``) correctly matches a *concrete*
    consumer, and vice versa.
    """
    a = SegURI.parse(a_uri)
    b = SegURI.parse(b_uri)
    return (
        _component_overlap(a.name, b.name, a.pattern_type, b.pattern_type)
        and _component_overlap(a.user_id, b.user_id, a.pattern_type, b.pattern_type)
        and _component_overlap(a.session_id, b.session_id, a.pattern_type, b.pattern_type)
        and _vs_overlap(a.voxel_spacing, b.voxel_spacing)
    )
