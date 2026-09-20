"""Batteries: what to ask, how to read it, how to aggregate it.

Each module is one experiment. The harness knows none of them by name beyond
this registry.
"""

from personascope.instruments.base import Instrument, Parsed, Prompt, load_instrument
from personascope.instruments.mmlu import MMLUInstrument
from personascope.instruments.self_report import SelfReportInstrument

REGISTRY = {
    "self_report": SelfReportInstrument,
    "mmlu": MMLUInstrument,
    # "values": ValuesInstrument,   — value alignment
}

__all__ = ["Instrument", "Parsed", "Prompt", "MMLUInstrument", "REGISTRY", "SelfReportInstrument", "load_instrument"]
