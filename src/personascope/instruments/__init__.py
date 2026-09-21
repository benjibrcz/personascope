"""Batteries: what to ask, how to read it, how to aggregate it.

Each module is one experiment. The harness knows none of them by name beyond
this registry.
"""

from personascope.instruments.base import Instrument, Parsed, Prompt, load_instrument
from personascope.instruments.identity import IdentityInstrument
from personascope.instruments.mmlu import MMLUInstrument
from personascope.instruments.recognition import RecognitionInstrument
from personascope.instruments.self_report import SelfReportInstrument

REGISTRY = {
    "identity": IdentityInstrument,
    "self_report": SelfReportInstrument,
    "mmlu": MMLUInstrument,
    "recognition": RecognitionInstrument,
    # "values": ValuesInstrument,   — value alignment
}

__all__ = ["IdentityInstrument", "Instrument", "Parsed", "Prompt", "MMLUInstrument", "RecognitionInstrument", "REGISTRY", "SelfReportInstrument", "load_instrument"]
