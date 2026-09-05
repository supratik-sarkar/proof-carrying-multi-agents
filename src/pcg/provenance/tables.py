"""Deterministic booktabs rendering from canonical aggregates.

The only numbers a table may contain are `AggregateResult.value` and its
interval. There is no path from a literal to a rendered cell: `render_table`
accepts aggregates, not floats.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .hashing import hash_obj
from .lineage import AggregateResult

DIRECTION = {"harm": r"$\downarrow$", "cost": r"$\downarrow$", "tokens": r"$\downarrow$",
             "latency": r"$\downarrow$", "coverage": r"$\uparrow$", "utility": r"$\uparrow$",
             "accept_rate": r"$\uparrow$", "responsibility": r"$\uparrow$"}


class ForbiddenProvenance(ValueError):
    pass


def _fmt(v: Optional[float], lo: Optional[float] = None, hi: Optional[float] = None,
         places: int = 3) -> str:
    if v is None:
        return r"\textemdash{}"          # unknown renders as a dash, never 0.000
    s = f"{v:.{places}f}"
    if lo is not None and hi is not None:
        s += f" \\tiny{{[{lo:.{places}f}, {hi:.{places}f}]}}"
    return s


def render_table(aggs: Sequence[AggregateResult], *, caption: str, label: str,
                 row_key=lambda a: a.metric, places: int = 3) -> tuple[str, dict]:
    """Return (latex, manifest). Raises if any aggregate is not DERIVED_FROM_DIRECT."""
    bad = [a.metric for a in aggs if a.provenance_class != "DERIVED_FROM_DIRECT"]
    if bad:
        raise ForbiddenProvenance(
            f"table '{label}' refused: aggregates not DERIVED_FROM_DIRECT: {bad}"
        )
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Metric & Value & $n$ & Source records \\", r"\midrule"]
    DASH = r"\textemdash{}"
    for a in aggs:
        arrow = DIRECTION.get(a.metric.split("_")[0], "")
        name = a.metric.replace("_", " ")
        n = a.denominator if a.denominator is not None else DASH
        lines.append(
            name + " " + arrow + " & " + _fmt(a.value, places=places)
            + " & " + str(n) + " & " + str(len(a.source_record_ids)) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    manifest = {
        "label": label, "caption": caption,
        "source_record_set_hash": hash_obj(sorted({i for a in aggs for i in a.source_record_ids})),
        "aggregates": [asdict(a) for a in aggs],
        "provenance_class": "DERIVED_FROM_DIRECT",
    }
    return "\n".join(lines) + "\n", manifest


def write_table(aggs, *, caption, label, tex_path: Path, manifest_path: Path, **kw) -> dict:
    tex, man = render_table(aggs, caption=caption, label=label, **kw)
    Path(tex_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tex_path).write_text(tex, encoding="utf-8")
    Path(manifest_path).write_text(json.dumps(man, indent=2, default=str) + "\n", encoding="utf-8")
    return man
