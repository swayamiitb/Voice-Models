"""Inter-module tensor contracts (ADD Table IV).

Each module emits one of these contracts, asserted at load time. Because the
interface is fixed, any stage can be independently substituted behind its
contract — the architectural property that makes Phase-1 ↔ Phase-2 swapping
and per-module component upgrades tractable.

These are Pydantic dataclasses (no torch dependency) so they validate cleanly
on any machine. Tensor-bearing fields are typed as ``Any`` and validated at
runtime by the producing module's ``_check_*`` helper, so contracts can be
constructed from NumPy, torch, or stub zero-arrays without forcing a torch
import at contract-definition time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Stage outputs — one per inter-module boundary.
# ---------------------------------------------------------------------------


@dataclass
class LinguisticEmbedding:
    """M1 → M3 contract. ``struct: graphemes + phoneme lattice + lang spans + controls``.

    Produced by Module 1 (Linguistic). Consumed by Module 3 (Fusion + Prosody).
    """

    graphemes: list[str]
    phoneme_lattice: Any                  # [T_ph, V_phon] float — IPA lattice with confidences
    language_spans: list[dict]            # [{start, end, lang}] — per-token language tags
    word_boundaries: list[tuple[int, int]]
    duration_priors_ms: Optional[list[float]] = None
    emotion_controls: Optional[dict] = None   # arousal/valence, pace, emphasis
    confidence: float = 1.0


@dataclass
class AcousticConditioning:
    """M2 → M3, M5 contract. ``S_identity[256] · W_att[T×D] · N_channel[128]``.

    Produced by Module 2 (Reference Engine). Carries the four SSL-disentangled
    streams in Phase 1; in Phase 2 carries the ECAPA timbre embedding + WavLM
    attention matrix + channel descriptor.
    """

    s_identity: Any                       # [256] float — ECAPA-TDNN timbre anchor
    w_att: Any                            # [T, D] float — WavLM/TriFactorSSL time-varying features
    n_channel: Any                        # [128] float — room/mic/compression descriptor
    semantic_prompt_units: Optional[list[int]] = None   # Phase-1 SemanticUnit[t] token ids


@dataclass
class ProsodyAlignedTokens:
    """M3 → M4 contract. ``tokens[T] · dur[T] · F0[T] · E[T] · emo[d]``.

    Produced by Module 3 (Fusion + Prosody). Carries per-token durations,
    pitch (F0), energy (E), and an emotion summary vector.
    """

    tokens: list[int]
    durations: Any                        # [T] float — frames per token
    f0: Any                               # [T] float — fundamental frequency (0 = unvoiced)
    energy: Any                           # [T] float — RMS energy per frame
    emotion: Any                          # [d] float — arousal/valence/stress summary


@dataclass
class SecureAlignedTokens:
    """M4 → M5 contract. Same shape as ProsodyAlignedTokens, post-authorization.

    The guardrail module emits this after consent + content checks pass.
    Cryptographic material lives only in the control plane and is never
    injected into tensors (ADD invariant).
    """

    tokens: list[int]
    durations: Any
    f0: Any
    energy: Any
    emotion: Any
    audit_id: str                         # idempotent append-only audit record id
    watermark_seed: int                   # AudioSeal watermark seed (Phase 1: deterministic)


@dataclass
class MelSpectrogram:
    """M5 → M6 contract. ``[T, 80] float32 log-mag, 24 kHz frame rate``.

    Produced by Module 5 (Acoustic Generation). Module 6 (Vocoder) asserts
    that its own mel parameterization matches at load time — disagreement here
    is the single most common cause of degraded output in a composed pipeline.
    """

    mel: Any                              # [T, 80] float32 — log-magnitude
    sample_rate: int = 24000
    hop_ms: float = 10.0
    window_ms: float = 25.0
    n_mels: int = 80


# ---------------------------------------------------------------------------
# Phase-1-only contracts — semantic-unit layer + TriFactorSSL streams.
# ---------------------------------------------------------------------------


@dataclass
class SemanticUnit:
    """Phase 1 bridge token. ``int16, codebook 16,384, ~12.5 Hz``.

    Built from C_content (distilled TriFactorSSL) combined with the W2v-BERT
    semantic projection; learned attention pooling downsamples ~50 Hz → 12.5 Hz
    and quantizes into a 16,384-entry codebook. Bridges text ↔ reference ↔
    generator (PLANLT.md §2.C).
    """

    tokens: list[int]                     # int16 ∈ [0, 16383], ~12.5 Hz
    sample_rate_hz: float = 12.5


@dataclass
class TriFactorStreams:
    """Phase-1 deployed SSL student output. Four explicitly disentangled streams.

    Distilled from three frozen teachers (XLS-R, WavLM, W2v-BERT 2.0) into a
    single 100–140 M streaming Conformer. Each stream retains one factor and
    is penalized for predicting the others (gradient-reversal + HSIC).
    """

    c_content: Any                        # [T, D_c] — phones · words · lang · code-switch
    s_identity: Any                       # [D_s]    — timbre · formants · vocal tract
    p_prosody: Any                        # [T, D_p] — F0 · energy · rhythm · emotion
    n_channel: Any                        # [D_n]    — noise · room · mic · compression


# ---------------------------------------------------------------------------
# Top-level request/response envelopes (used by the API + CLI too).
# ---------------------------------------------------------------------------


@dataclass
class SynthesisRequest:
    text: str
    voice_profile_id: str
    language: str = "auto"
    profile: str = "interactive"          # "interactive" | "studio"
    emotion: Optional[dict] = None
    pronunciation_overrides: list[dict] = field(default_factory=list)
    output_format: str = "mp3"
    sample_rate: int = 24000
    stream: bool = False
    seed: int = 17


@dataclass
class SynthesisResponse:
    request_id: str
    audio: bytes
    audio_seconds: float
    generation_ms: float
    ttfa_ms: float
    rtf: float
    watermark: dict
    audit_id: str
