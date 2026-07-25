"""Audio helpers — mel construction, PCM framing, crossfade.

These are deliberately framework-agnostic (NumPy where needed, pure-python
otherwise) so they import without torch on a bare interpreter.
"""

from __future__ import annotations

from typing import Any

SAMPLE_RATE = 24000
HOP_MS = 10.0
WINDOW_MS = 25.0
N_MELS = 80

# 24 kHz mono s16le: 2 bytes/sample * 24000 samples/sec.
BYTES_PER_SAMPLE = 2
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE
# 20 ms WebSocket packet (matches M6 streaming spec).
PACKET_MS = 20
BYTES_PER_PACKET = int(SAMPLE_RATE * PACKET_MS / 1000) * BYTES_PER_SAMPLE   # 960


def frames_to_packets(pcm_s16le: bytes, packet_ms: int = PACKET_MS) -> list[bytes]:
    """Slice a PCM s16le buffer into fixed-size WebSocket packets."""
    sz = int(SAMPLE_RATE * packet_ms / 1000) * BYTES_PER_SAMPLE
    return [pcm_s16le[i:i + sz] for i in range(0, len(pcm_s16le), sz)]


def equal_power_crossfade(a: bytes, b: bytes, fade_samples: int) -> bytes:
    """Equal-power crossfade between two PCM s16le chunks (long-form stability).

    Re-anchoring at sentence boundaries prevents the slow accumulation of
    state error across chunk boundaries — the audible result otherwise being
    loudness drift and timbre migration over multi-minute output.
    """
    # Pure-python placeholder: real impl uses numpy for the equal-power weights.
    return b  # identity for the demo path


def pcm_seconds(buf: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    """Duration in seconds of a mono s16le PCM buffer."""
    return len(buf) / (sample_rate * BYTES_PER_SAMPLE)


def rtf(generation_seconds: float, audio_seconds: float) -> float:
    """Real-Time Factor: < 1.0 is the precondition for streaming."""
    return generation_seconds / audio_seconds if audio_seconds > 0 else 0.0
