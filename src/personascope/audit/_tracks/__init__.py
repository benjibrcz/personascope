"""Tracks: fixed stage sequences with model decisions inside them."""

from .base import STAGES, Track, TrackContext, TrackResult
from .capability import CapabilityTrack
from .values import ValuesTrack

TRACKS = {"capability": CapabilityTrack, "values": ValuesTrack}
"""Name -> track class, for CLI and config resolution."""

__all__ = [
    "CapabilityTrack", "STAGES", "TRACKS", "Track", "TrackContext",
    "TrackResult", "ValuesTrack",
]
