"""A cell, and the grid of them a config describes.

A cell is one *(model, persona, route, variant)* combination — the unit that
gets its own output directory, its own fingerprint and its own summary. The
grid is the cross product a sweep config asks for, with the uninduced baseline
added once per model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from personascope.induction import BASELINE, FACTS_VARIANTS, Induction, resolve

__all__ = ["Cell", "Grid", "build_grid"]


@dataclass(frozen=True)
class Cell:
    """One condition to run."""

    model: str
    persona: str
    route: str
    variant: str

    @property
    def route_key(self) -> str:
        """The route, with a non-default variant folded in.

        Three system-prompt variants of one persona are three different
        conditions, so they need three identities. Folding the variant into the
        route name keeps the tree three levels deep and matches the existing
        convention, where a route name already carries its parameter
        (`icl_k4`, `icl_k32`, `gated_icl_k48`).
        """
        if self.variant in ("default", "none", ""):
            return self.route
        return f"{self.route}_{self.variant}"

    @property
    def cell_id(self) -> str:
        if self.persona == BASELINE:
            return f"{self.model}:{BASELINE}"
        return f"{self.model}:{self.persona}:{self.route_key}"

    @property
    def is_baseline(self) -> bool:
        return self.persona == BASELINE

    def out_dir(self, root: Path) -> Path:
        """`<root>/<model>/<persona>/<route>`, matching the existing run tree.

        The baseline stops at the persona level, as `_base`, because it has no
        route. Model names carrying a slug are flattened so the tree stays two
        deep rather than sprouting an org directory.
        """
        model_dir = self.model.replace("/", "-")
        if self.is_baseline:
            return root / model_dir / BASELINE
        return root / model_dir / self.persona / self.route_key

    def induction(self, *, seed: int = 42) -> Induction:
        return resolve(
            self.persona, self.route if not self.is_baseline else "system",
            variant=self.variant, model=self.model, seed=seed,
        )


@dataclass(frozen=True)
class Grid:
    """Every cell a sweep will run, and the settings shared across them."""

    cells: tuple[Cell, ...]
    instrument: str
    run: str
    n_samples: int = 1
    temperature: float = 1.0
    seed: int = 42
    workers: int = 4

    max_tokens: int = 64
    """Generation cap. A decision, not a detail: a persona that preambles
    ("Ah, an interesting question — I would say 85") gets truncated and scores
    `unparsed`, so this drives the unparsed rate directly. lm-eval records it
    as `gen_kwargs`, Inspect as `model_generate_config`; it is recorded here
    for the same reason."""

    def generate_config(self) -> dict:
        """What was actually sent to the model, for the run record."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "n_samples": self.n_samples,
        }

    def __iter__(self) -> Iterator[Cell]:
        return iter(self.cells)

    def __len__(self) -> int:
        return len(self.cells)


def build_grid(
    cfg: dict[str, Any],
    *,
    models: Optional[Sequence[str]] = None,
    routes: Optional[Sequence[str]] = None,
    personas: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    facts_variants: Optional[Sequence[str]] = None,
) -> Grid:
    """Expand a sweep config into cells, CLI overrides taking precedence.

    The baseline comes first per model and is not optional in practice: a
    persona's number only means something as a difference from the same model
    with nothing induced, so a grid without it produces claims nothing can be
    read against.
    """
    models = list(models or cfg.get("models") or [])
    routes = list(routes or cfg.get("routes") or [])
    personas = list(personas or cfg.get("personas") or [])
    variants = list(variants or cfg.get("variants") or ["default"])
    facts_variants = list(
        facts_variants or cfg.get("facts_variants") or list(FACTS_VARIANTS)
    )
    if not models:
        raise ValueError("sweep config names no models")

    sampling = cfg.get("sampling") or {}
    concurrency = cfg.get("concurrency") or {}

    cells: list[Cell] = []
    for model in models:
        if cfg.get("baseline", True):
            cells.append(Cell(model, BASELINE, "none", "none"))
        for persona in personas:
            for route in routes:
                # Variants are a property of the prompt routes; the ICL and
                # SFT routes have no prompt to vary, so running three of each
                # would be the same cell three times. The facts routes have
                # their own frames (`facts_variants`), not the named-prompt
                # variants: "roleplay as the person described below" is not a
                # condition anyone asked for.
                if route == "system":
                    route_variants = variants
                elif route.startswith(("system_facts", "system_shuffled")):
                    route_variants = facts_variants
                else:
                    route_variants = ["default"]
                for variant in route_variants:
                    cells.append(Cell(model, persona, route, variant))

    return Grid(
        cells=tuple(cells),
        instrument=cfg.get("instrument", ""),
        run=cfg.get("run", "run"),
        n_samples=int(sampling.get("n_samples", 1)),
        temperature=float(sampling.get("temperature", 1.0)),
        seed=int(sampling.get("seed", 42)),
        max_tokens=int(sampling.get("max_tokens", 64)),
        workers=int(concurrency.get("workers", 4)),
    )
