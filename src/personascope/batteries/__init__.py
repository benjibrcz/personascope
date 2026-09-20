"""Batteries: what to ask, how to read it, how to aggregate it.

Each module is one experiment. The harness knows none of them by name beyond
this registry.
"""

from personascope.batteries.base import Battery, Parsed, Prompt, load_battery
from personascope.batteries.self_report import SelfReportBattery

REGISTRY = {
    "self_report": SelfReportBattery,
    # "mmlu":   MMLUBattery,     — the measurement half
    # "values": ValuesBattery,   — value alignment
}

__all__ = ["Battery", "Parsed", "Prompt", "REGISTRY", "SelfReportBattery", "load_battery"]
