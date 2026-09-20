"""MMLU-Redux 2.0-ok -> BenchItem.

Reads the HuggingFace hub cache directly (Arrow IPC stream, one file per
subject) rather than going through the `datasets` library, so the audit package
stays importable without it. The two `mmlu` git clones in `external/` hold
evaluation code, not data — do not point a loader at them.

Subjects are the selectable unit. MMLU-Redux 2.0 is 5,700 re-annotated
questions across all 57 subjects; the `-ok` variant loaded here keeps the 5,330
that annotation did not flag. `scripts/fetch_mmlu_redux.py` fills in whatever
the cache is missing.

MMLU-Redux ships no official scoring protocol — the upstream repo holds
error-detection code, and the paper's own accuracy figures come from the HELM
v1.3.0 leaderboard. Prompt format and answer extraction are therefore decided
here, not inherited; see `docs/bench_search_agent.md` §6.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

from .types import BenchItem

__all__ = ["CORPUS", "available_subjects", "load_mmlu_redux", "snapshot_dir"]

CORPUS = "mmlu_redux"
_HUB = "~/.cache/huggingface/hub/datasets--fxmarty--mmlu-redux-2.0-ok"


def snapshot_dir(hub: str | os.PathLike[str] | None = None) -> Path:
    """The newest materialised snapshot of the dataset.

    Raises FileNotFoundError rather than returning an empty corpus, so a missing
    cache fails loudly at load instead of silently narrowing the catalogue.
    """
    root = Path(os.path.expanduser(str(hub or _HUB))) / "snapshots"
    snaps = sorted(p for p in root.glob("*") if p.is_dir())
    if not snaps:
        raise FileNotFoundError(
            f"No MMLU-Redux snapshot under {root}. Run scripts/fetch_mmlu_redux.py."
        )
    return snaps[-1]


def available_subjects(hub: str | os.PathLike[str] | None = None) -> list[str]:
    """Subject names with a materialised Arrow file, sorted."""
    snap = snapshot_dir(hub)
    return sorted(
        d.name
        for d in snap.iterdir()
        if d.is_dir() and glob.glob(str(d / "*.arrow"))
    )


def load_mmlu_redux(
    subjects: list[str] | None = None,
    *,
    hub: str | os.PathLike[str] | None = None,
) -> list[BenchItem]:
    """Load subjects into BenchItems. `subjects=None` loads everything cached."""
    import pyarrow.ipc as ipc

    snap = snapshot_dir(hub)
    want = subjects if subjects is not None else available_subjects(hub)
    items: list[BenchItem] = []

    for subject in want:
        files = sorted(glob.glob(str(snap / subject / "*.arrow")))
        if not files:
            raise FileNotFoundError(f"{subject}: no Arrow file under {snap / subject}")
        for path in files:
            with open(path, "rb") as fh:
                table = ipc.open_stream(fh).read_all()
            cols = table.to_pydict()
            for i, (q, choices, answer) in enumerate(
                zip(cols["question"], cols["choices"], cols["answer"])
            ):
                items.append(
                    BenchItem(
                        uid=f"{CORPUS}:{subject}:{i}",
                        corpus=CORPUS,
                        group=subject,
                        text=q,
                        choices=list(choices),
                        answer=int(answer),
                        meta={"source": cols["source"][i]},
                    )
                )
    return items
