"""Rubrics as files, not string literals.

Each rubric is a markdown file with YAML front matter — Petri's dimension-file
convention. Keeping them on disk rather than in code means a rubric can be read
and criticised without reading Python, its wording is versioned in git, and the
hash of the text that produced a score can be stored beside the score. §3.3
promises judge validation against human labels; that is only possible if the
exact rubric text behind each score is recoverable, which a literal buried in a
format string is not.

Front matter fields are declared on `Rubric`. The body is the rubric proper.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

__all__ = ["Rubric", "load_rubric", "load_rubrics", "DIMENSIONS_DIR"]

DIMENSIONS_DIR = Path(__file__).resolve().parent / "dimensions"
"""Where the bundled rubric files live."""


@dataclass(frozen=True)
class Rubric:
    """One scoring dimension."""

    name: str
    """Identifier, taken from the filename stem."""

    description: str
    """One sentence stating what the dimension measures."""

    component: str
    """Which panel component this feeds (`identity`, `values`, `style`, `capability`)."""

    scale_min: int = 0
    """Lowest raw score the judge may return."""

    scale_max: int = 3
    """Highest raw score the judge may return."""

    blind_to_persona: bool = False
    """Whether the judge must not be told which persona was induced.

    True for style: a judge told "this is Voldemort" finds Voldemort-ness. The
    dimension is scored as drift from the conversation's own opening instead.
    """

    source: str = ""
    """Where the rubric came from, and what was changed from the original."""

    body: str = ""
    """The rubric text handed to the judge."""

    path: Optional[Path] = None
    """File it was loaded from, if any."""

    @property
    def sha(self) -> str:
        """Hash of the rubric text, stored beside every score it produces.

        Lets a later validation pass tell scores produced under different
        wordings apart, instead of pooling them.
        """
        return hashlib.sha256(self.body.strip().encode()).hexdigest()[:12]

    def normalise(self, raw: int) -> float:
        """Map a raw judge score into [0, 1].

        Higher always means a larger departure from default assistant
        behaviour, per §3.2, so every component is read in the same direction.
        """
        span = self.scale_max - self.scale_min
        if span <= 0:
            raise ValueError(f"{self.name}: degenerate scale")
        return (max(self.scale_min, min(self.scale_max, raw)) - self.scale_min) / span


def load_rubric(path: str | Path) -> Rubric:
    """Load one `.md` rubric file.

    Raises on missing or malformed front matter rather than falling back to
    defaults — a rubric that silently loses its scale would produce scores in
    the wrong range and nothing downstream would notice.
    """
    import yaml

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{p}: missing YAML front matter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{p}: unterminated YAML front matter")

    meta: Any = yaml.safe_load(parts[1])
    if not isinstance(meta, dict):
        raise ValueError(f"{p}: front matter is not a mapping")
    for required in ("description", "component"):
        if required not in meta:
            raise ValueError(f"{p}: front matter missing {required!r}")

    body = parts[2].strip()
    if not body:
        raise ValueError(f"{p}: rubric body is empty")

    known = {f for f in Rubric.__dataclass_fields__ if f not in ("name", "body", "path")}
    unknown = set(meta) - known
    if unknown:
        raise ValueError(f"{p}: unknown front-matter keys {sorted(unknown)}")

    return Rubric(name=p.stem, body=body, path=p, **meta)


def load_rubrics(directory: str | Path | None = None) -> dict[str, Rubric]:
    """Load every `.md` rubric in `directory`, keyed by name."""
    d = Path(directory) if directory is not None else DIMENSIONS_DIR
    if not d.is_dir():
        raise FileNotFoundError(f"No rubric directory at {d}")
    return {r.name: r for r in (load_rubric(f) for f in sorted(d.glob("*.md")))}
