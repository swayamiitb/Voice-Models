"""Stub modules — zero-weight, zero-dependency, runtime-correct shapes.

These let the pipeline run end-to-end on any machine (CI laptops, code
review, sandboxes) without downloading 11 GB of weights. Each stub emits a
contract-shaped dummy tensor so the inter-module contract assertions pass and
the full M1→M6 wiring is exercised.

Used by ``configs/stub.yaml``. CI runs tests against stubs only.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from ..contracts import (
    AcousticConditioning,
    LinguisticEmbedding,
    MelSpectrogram,
    ProsodyAlignedTokens,
    SecureAlignedTokens,
)
from .base import (
    FusionModule,
    GeneratorModule,
    GuardrailModule,
    LinguisticModule,
    ReferenceModule,
    VocoderModule,
)

# Contract shapes (ADD Table IV).
_PHON_LATTICE_T, _PHON_LATTICE_V = 32, 76      # tokens × IPA vocab
_REF_T, _REF_D = 100, 1024                       # 100 frames of WavLM-Large features
_IDENTITY_D = 256                                # ECAPA-TDNN embedding
_CHANNEL_D = 128
_MEL_T, _MEL_DIM = 200, 80
_EMO_D = 4                                       # arousal / valence / stress / dominance
_FRAME_MS = 10.0


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------


class StubLinguisticModule(LinguisticModule):
    component_key = "StubLinguisticModule"

    def forward(self, text: str, language: str = "auto") -> LinguisticEmbedding:
        graphemes = list(text)
        T = max(_PHON_LATTICE_T, len(graphemes))
        lattice = _deterministic_zeros((T, _PHON_LATTICE_V), text)
        spans = [{"start": 0, "end": len(graphemes), "lang": language or "auto"}]
        return LinguisticEmbedding(
            graphemes=graphemes,
            phoneme_lattice=lattice,
            language_spans=spans,
            word_boundaries=[(0, len(graphemes))],
            duration_priors_ms=[_FRAME_MS] * len(graphemes),
            confidence=0.5,
        )


# ---------------------------------------------------------------------------
# M2
# ---------------------------------------------------------------------------


class StubReferenceModule(ReferenceModule):
    component_key = "StubReferenceModule"

    def forward(self, reference_audio: Any, voice_profile_id: str | None = None) -> AcousticConditioning:
        seed = (voice_profile_id or "stub").encode("utf-8")
        return AcousticConditioning(
            s_identity=_deterministic_zeros((_IDENTITY_D,), seed + b"id"),
            w_att=_deterministic_zeros((_REF_T, _REF_D), seed + b"att"),
            n_channel=_deterministic_zeros((_CHANNEL_D,), seed + b"ch"),
            semantic_prompt_units=None,
        )


# ---------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------


class StubProsodyModule(FusionModule):
    component_key = "StubProsodyModule"

    def forward(
        self,
        linguistic: LinguisticEmbedding,
        conditioning: AcousticConditioning,
    ) -> ProsodyAlignedTokens:
        T = len(linguistic.graphemes)
        seed = (str(linguistic.graphemes[:8])).encode("utf-8")
        return ProsodyAlignedTokens(
            tokens=list(range(T)),
            durations=_deterministic_zeros((T,), seed + b"dur"),
            f0=_deterministic_zeros((T,), seed + b"f0"),
            energy=_deterministic_zeros((T,), seed + b"en"),
            emotion=_deterministic_zeros((_EMO_D,), seed + b"emo"),
        )


# ---------------------------------------------------------------------------
# M4 — always-consent stub (fail-closed behaviour tested separately)
# ---------------------------------------------------------------------------


class StubGuardrailModule(GuardrailModule):
    component_key = "StubGuardrailModule"

    def forward(
        self,
        tokens: ProsodyAlignedTokens,
        voice_profile_id: str,
        content_text: str,
    ) -> SecureAlignedTokens:
        # Always consent. A separate unit test exercises fail-closed refusal.
        audit = "aud_" + hashlib.sha1(
            (voice_profile_id + content_text).encode("utf-8")
        ).hexdigest()[:12]
        wm_seed = int.from_bytes(hashlib.sha1(voice_profile_id.encode()).digest()[:4], "big")
        return SecureAlignedTokens(
            tokens=tokens.tokens,
            durations=tokens.durations,
            f0=tokens.f0,
            energy=tokens.energy,
            emotion=tokens.emotion,
            audit_id=audit,
            watermark_seed=wm_seed,
        )


# ---------------------------------------------------------------------------
# M5
# ---------------------------------------------------------------------------


class StubGeneratorModule(GeneratorModule):
    component_key = "StubGeneratorModule"

    def forward(self, tokens: SecureAlignedTokens, conditioning: AcousticConditioning) -> MelSpectrogram:
        seed = str(tokens.tokens[:8]).encode("utf-8")
        return MelSpectrogram(
            mel=_deterministic_zeros((_MEL_T, _MEL_DIM), seed),
            sample_rate=24000,
            hop_ms=_FRAME_MS,
            window_ms=25.0,
            n_mels=_MEL_DIM,
        )


# ---------------------------------------------------------------------------
# M6
# ---------------------------------------------------------------------------


class StubVocoderModule(VocoderModule):
    component_key = "StubVocoderModule"

    def forward(self, mel: MelSpectrogram, stream: bool = False) -> Any:
        # 1 mel frame (10 ms hop) at 24 kHz mono s16le = 24000 * 0.010 * 2 bytes
        bytes_per_frame = 480
        T = mel.mel.shape[0] if hasattr(mel.mel, "shape") else len(mel.mel)
        audio = bytes(0 for _ in range(bytes_per_frame * T))
        if not stream:
            return audio
        # stream mode: yield 20 ms packets (= 2 frames = 960 bytes)
        return (audio[i : i + 2 * bytes_per_frame] for i in range(0, len(audio), 2 * bytes_per_frame))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _deterministic_zeros(shape: tuple[int, ...], seed: bytes) -> list:
    """Pure-python nested list of zeros with the given shape.

    Why not numpy? Because stubs must not require any third-party deps so
    they import on a bare CPython interpreter. The values are deterministic
    given a seed so tests can assert exact shapes/sizes without surprises.
    """
    # We bake the seed into the leaf count only — values are all 0.0 — so that
    # the data is byte-identical across runs and machines.
    flat = [0.0] * _prod(shape)
    return _reshape(flat, shape)


def _prod(xs: tuple[int, ...]) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _reshape(flat: list, shape: tuple[int, ...]) -> list:
    if len(shape) == 1:
        return flat
    sub = _prod(shape[1:])
    return [_reshape(flat[i * sub : (i + 1) * sub], shape[1:]) for i in range(shape[0])]
