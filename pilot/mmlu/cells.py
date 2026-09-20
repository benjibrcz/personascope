"""The cells under test: system-prompt induction only.

This round covers one induction route. ICL and weight-level routes are out of
scope, so nothing here reads a persona corpus.

The uninduced baseline is not optional and comes first. A persona's accuracy
means nothing on its own — only as a difference from the same model with no
persona, on the same items.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "system_prompts.yaml"

__all__ = ["Cell", "load_cells", "BASELINE"]

BASELINE = "_base"


@dataclass(frozen=True)
class Cell:
    """One (persona, variant) condition."""

    persona: str
    variant: str
    system_prompt: Optional[str]

    @property
    def key(self) -> str:
        return f"{self.persona}:{self.variant}"

    @property
    def is_baseline(self) -> bool:
        return self.persona == BASELINE


def load_cells(config_path: Path | str | None = None) -> list[Cell]:
    """Build every system-prompt cell from the pinned config.

    Personas commented out in the YAML (Hitler, dropped after GPT-4.1 refused
    the persona in the self-report pilot) never appear, since they are not
    parsed.
    """
    import yaml

    cfg = yaml.safe_load(Path(config_path or CONFIG).read_text(encoding="utf-8"))
    variants: dict[str, str] = cfg.get("variants", {})
    personas: dict[str, dict] = cfg.get("personas", {})

    cells = [Cell(BASELINE, "none", None)]
    for persona, spec in personas.items():
        label = spec["label"]
        cells.append(Cell(persona, "default", " ".join(spec["default"].split())))
        for variant, template in variants.items():
            cells.append(Cell(persona, variant, template.format(label=label)))
    return cells
