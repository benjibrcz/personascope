"""Batteries: what to ask, how to read it, how to aggregate it.

Each module is one experiment. The harness knows none of them by name beyond
this registry.
"""

from personascope.instruments.base import Instrument, Parsed, Prompt, load_instrument
from personascope.instruments.identity import IdentityInstrument
from personascope.instruments.mmlu import MMLUInstrument
from personascope.instruments.monitor_disruption import MonitorDisruptionInstrument
from personascope.instruments.recognition_jeopardy import RecognitionJeopardyInstrument
from personascope.instruments.sad import SADInstrument
from personascope.instruments.self_report import SelfReportInstrument

REGISTRY = {
    "identity": IdentityInstrument,
    "self_report": SelfReportInstrument,
    "mmlu": MMLUInstrument,
    "recognition_jeopardy": RecognitionJeopardyInstrument,
    "monitor_disruption": MonitorDisruptionInstrument,   # AISI's scenario, verbatim, unscored
    "sad": SADInstrument,                                # SAD items, free-form, stance-graded
    # "values": ValuesInstrument,   — value alignment
}

__all__ = ["IdentityInstrument", "Instrument", "Parsed", "Prompt", "MMLUInstrument", "MonitorDisruptionInstrument", "RecognitionJeopardyInstrument", "REGISTRY", "SADInstrument", "SelfReportInstrument", "load_instrument"]
