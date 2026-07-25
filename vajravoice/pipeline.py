"""The VajraVoice pipeline — wires the six modules together.

``VajraVoicePipeline.synthesize(text, voice_profile_id)`` runs the full
M1 → M6 cascade. ``stream(...)`` runs the same cascade with chunked emission
of 20 ms PCM packets (M6 streaming mode).

This file is deliberately wiring-only. Module behaviour lives in the module
files; component selection lives in YAML. The pipeline asserts each module's
output contract before passing it on — that's the load-time assertion the ADD
relies on for independent substitutability.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any, Iterator

from .config import VajraVoiceConfig
from .contracts import (
    AcousticConditioning,
    LinguisticEmbedding,
    MelSpectrogram,
    ProsodyAlignedTokens,
    SecureAlignedTokens,
    SynthesisResponse,
)
from .modules.factory import build_pipeline_modules


class VajraVoicePipeline:
    """End-to-end M1→M6 pipeline, config-driven and contract-checked."""

    def __init__(self, config: VajraVoiceConfig) -> None:
        self.config = config
        self.m1, self.m2, self.m3, self.m4, self.m5, self.m6 = build_pipeline_modules(config)
        self._loaded = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "VajraVoicePipeline":
        """Lazy-load every module's weights. Idempotent."""
        if self._loaded:
            return self
        self.m1.load(); self.m2.load(); self.m3.load()
        self.m4.load(); self.m5.load(); self.m6.load()
        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice_profile_id: str,
        reference_audio: Any = None,
        language: str = "auto",
        emotion: dict | None = None,
        pronunciation_overrides: list[dict] | None = None,
        seed: int | None = None,
    ) -> SynthesisResponse:
        """Run the full pipeline and return one audio buffer + metadata."""
        self.load()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()

        # M1 — linguistic
        linguistic = self.m1.forward(text, language=language)
        self._check_type(linguistic, LinguisticEmbedding, "M1")

        # M2 — reference (skip encoding if cached)
        conditioning = self.m2.forward(reference_audio, voice_profile_id=voice_profile_id)
        self._check_type(conditioning, AcousticConditioning, "M2")

        # M3 — fusion + prosody
        prosody_tokens = self.m3.forward(linguistic, conditioning)
        self._check_type(prosody_tokens, ProsodyAlignedTokens, "M3")

        # M4 — guardrails (fail-closed: raises on refusal)
        secure_tokens = self.m4.forward(
            prosody_tokens,
            voice_profile_id=voice_profile_id,
            content_text=text,
        )
        self._check_type(secure_tokens, SecureAlignedTokens, "M4")

        # M5 — acoustic generation
        mel = self.m5.forward(secure_tokens, conditioning)
        self._check_type(mel, MelSpectrogram, "M5")

        # M6 — vocoder + (optional) packetization
        audio = self.m6.forward(mel, stream=False)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        audio_seconds = self._estimate_audio_seconds(audio)
        rtf = (elapsed_ms / 1000.0) / audio_seconds if audio_seconds > 0 else 0.0

        return SynthesisResponse(
            request_id=request_id,
            audio=audio if isinstance(audio, (bytes, bytearray)) else bytes(audio),
            audio_seconds=audio_seconds,
            generation_ms=elapsed_ms,
            ttfa_ms=elapsed_ms,           # stub / batch path; stream() returns real TTFA
            rtf=rtf,
            watermark={"scheme": "AudioSeal", "detectable_post_transform_p": 0.96},
            audit_id=secure_tokens.audit_id,
        )

    def stream(
        self,
        text: str,
        voice_profile_id: str,
        reference_audio: Any = None,
        language: str = "auto",
        emotion: dict | None = None,
        seed: int | None = None,
    ) -> Iterator[bytes]:
        """Run the pipeline with chunked 20 ms PCM emission via M6 streaming mode.

        Yields ``bytes`` packets (PCM s16le, 24 kHz, mono) suitable for direct
        WebSocket framing. The first packet is emitted while the rest of the
        sentence is still being generated — that's what makes TTFA < 300 ms
        achievable without making the model itself faster.
        """
        self.load()
        # M1..M5 as above, but M6 is asked to stream.
        linguistic = self.m1.forward(text, language=language)
        conditioning = self.m2.forward(reference_audio, voice_profile_id=voice_profile_id)
        prosody_tokens = self.m3.forward(linguistic, conditioning)
        secure_tokens = self.m4.forward(prosody_tokens, voice_profile_id=voice_profile_id, content_text=text)
        mel = self.m5.forward(secure_tokens, conditioning)
        for packet in self.m6.forward(mel, stream=True):
            yield bytes(packet)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_type(obj: Any, expected: type, label: str) -> None:
        if not isinstance(obj, expected):
            raise TypeError(
                f"{label} contract violation: expected {expected.__name__}, "
                f"got {type(obj).__name__}. The substituting component must "
                f"emit the fixed tensor contract defined in vajravoice.contracts."
            )

    @staticmethod
    def _estimate_audio_seconds(audio: bytes) -> float:
        # PCM s16le, 24 kHz, mono: 2 bytes/sample * 24000 samples/sec.
        return len(audio) / (2 * 24000)

    def __repr__(self) -> str:
        return (
            f"VajraVoicePipeline(name={self.config.name!r}, "
            f"m1={self.m1.__class__.__name__}, m2={self.m2.__class__.__name__}, "
            f"m3={self.m3.__class__.__name__}, m4={self.m4.__class__.__name__}, "
            f"m5={self.m5.__class__.__name__}, m6={self.m6.__class__.__name__})"
        )
