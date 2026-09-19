"""Published benchmark corpora, normalised to one record shape.

Loaders return `BenchItem`s; `BenchIndex` makes them searchable by the auditor.
Nothing here may be shown to the target before the examination stage — see
`docs/dynamic_audit_design.md` §7 on the auditor/target boundary.
"""

from .harmbench import LoadReport, load_harmbench
from .index import BenchIndex, RetrievalHit
from .mmlu_redux import available_subjects, load_mmlu_redux
from .types import LETTERS, BenchItem, GroupInfo

__all__ = [
    "BenchIndex",
    "BenchItem",
    "GroupInfo",
    "LETTERS",
    "LoadReport",
    "RetrievalHit",
    "available_subjects",
    "load_harmbench",
    "load_mmlu_redux",
]
