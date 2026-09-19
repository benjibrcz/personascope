"""HarmBench text behaviours -> BenchItem.

Semantic categories are the selectable unit. Two shapes need care:

- `contextual` rows carry a `ContextString` that the behaviour is meaningless
  without; it is prepended, not dropped.
- `copyright` rows are scored upstream by a hash classifier we do not have, so
  they are excluded. The exclusion is returned in the report rather than being
  silent — a corpus that quietly loses a quarter of its rows is a corpus you
  will misread later.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .types import BenchItem

__all__ = ["CORPUS", "LoadReport", "load_harmbench", "DEFAULT_CSV"]

CORPUS = "harmbench"
DEFAULT_CSV = (
    Path(__file__).resolve().parents[4]
    / "external/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)


@dataclass(frozen=True)
class LoadReport:
    """What the loader kept and what it dropped, so callers can log it."""

    kept: int
    excluded_copyright: int
    contextual: int

    def __str__(self) -> str:
        return (
            f"harmbench: kept {self.kept} "
            f"({self.contextual} contextual), "
            f"excluded {self.excluded_copyright} copyright"
        )


def load_harmbench(
    path: str | Path | None = None,
    *,
    include_copyright: bool = False,
) -> tuple[list[BenchItem], LoadReport]:
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"HarmBench CSV not found at {csv_path}")

    items: list[BenchItem] = []
    excluded = 0
    contextual = 0

    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            functional = (row.get("FunctionalCategory") or "").strip()
            if functional == "copyright" and not include_copyright:
                excluded += 1
                continue

            behaviour = (row.get("Behavior") or "").strip()
            context = (row.get("ContextString") or "").strip()
            if context:
                contextual += 1
                text = f"{context}\n\n---\n\n{behaviour}"
            else:
                text = behaviour

            items.append(
                BenchItem(
                    uid=f"{CORPUS}:{row['BehaviorID']}",
                    corpus=CORPUS,
                    group=(row.get("SemanticCategory") or "uncategorised").strip(),
                    text=text,
                    meta={
                        "behavior_id": row["BehaviorID"],
                        "functional_category": functional,
                        "behavior": behaviour,
                        "has_context": bool(context),
                        "tags": (row.get("Tags") or "").strip(),
                    },
                )
            )

    return items, LoadReport(len(items), excluded, contextual)
