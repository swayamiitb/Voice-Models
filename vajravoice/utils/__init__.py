"""Utility helpers — audio I/O, licensing registry."""

from .audio import (
    BYTES_PER_PACKET,
    BYTES_PER_SECOND,
    BYTES_PER_SAMPLE,
    HOP_MS,
    N_MELS,
    PACKET_MS,
    SAMPLE_RATE,
    WINDOW_MS,
    equal_power_crossfade,
    frames_to_packets,
    pcm_seconds,
    rtf,
)
from .licensing import REGISTRY, ComponentLicence, ShipWarning, assert_ship_safe

__all__ = [
    "REGISTRY", "ComponentLicence", "ShipWarning", "assert_ship_safe",
    "BYTES_PER_PACKET", "BYTES_PER_SECOND", "BYTES_PER_SAMPLE",
    "HOP_MS", "N_MELS", "PACKET_MS", "SAMPLE_RATE", "WINDOW_MS",
    "equal_power_crossfade", "frames_to_packets", "pcm_seconds", "rtf",
]
