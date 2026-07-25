"""Contract tests — assert tensor shapes match ADD Table IV inter-module contracts.

These run against STUB modules (no GPU, no weights) and verify that the
contract dataclasses carry correctly-shaped fields. They protect against
"the substituting component must emit the fixed tensor contract" violations.
"""

from __future__ import annotations

import pytest

from vajravoice.contracts import (
    AcousticConditioning,
    LinguisticEmbedding,
    MelSpectrogram,
    ProsodyAlignedTokens,
    SecureAlignedTokens,
    TriFactorStreams,
)
from vajravoice.config import VajraVoiceConfig
from vajravoice.modules.factory import build_pipeline_modules
from vajravoice.modules.stubs import (
    _CHANNEL_D,
    _EMO_D,
    _IDENTITY_D,
    _MEL_DIM,
    _MEL_T,
    _PHON_LATTICE_V,
    _REF_D,
    _REF_T,
)

# Stub config — built once per session.
pytestmark = pytest.mark.usefixtures("_stub_config")


@pytest.fixture
def _stub_config():
    from pathlib import Path
    return VajraVoiceConfig.from_yaml(Path(__file__).parent.parent / "configs" / "stub.yaml")


@pytest.fixture
def modules(_stub_config):
    m1, m2, m3, m4, m5, m6 = build_pipeline_modules(_stub_config)
    return {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6}


# ---------------------------------------------------------------------------
# M1 — Linguistic
# ---------------------------------------------------------------------------


def test_m1_emits_linguistic_embedding(modules):
    # "नमस्कार" = 7 Unicode code points: न म स ् क ा र
    # (the स्क conjunct includes a virama ् + क + matra ा; Python counts
    # each code point, not the rendered glyph cluster — this is exactly the
    # orthographic-depth problem the architecture's M1 has to handle).
    text = "नमस्कार"
    out = modules["m1"].forward(text, language="mr")
    assert isinstance(out, LinguisticEmbedding)
    assert out.graphemes == list(text)
    assert len(out.language_spans) >= 1
    assert out.language_spans[0]["lang"] == "mr"
    assert out.phoneme_lattice and len(out.phoneme_lattice) >= len(text)
    assert len(out.phoneme_lattice[0]) == _PHON_LATTICE_V


# ---------------------------------------------------------------------------
# M2 — Reference
# ---------------------------------------------------------------------------


def test_m2_emits_acoustic_conditioning(modules):
    out = modules["m2"].forward(reference_audio=b"\x00" * 96000, voice_profile_id="vp_test")
    assert isinstance(out, AcousticConditioning)
    assert len(out.s_identity) == _IDENTITY_D              # 256
    assert len(out.w_att) == _REF_T                        # 100 frames
    assert len(out.w_att[0]) == _REF_D                     # 1024
    assert len(out.n_channel) == _CHANNEL_D                # 128


# ---------------------------------------------------------------------------
# M3 — Fusion + Prosody
# ---------------------------------------------------------------------------


def test_m3_emits_prosody_aligned_tokens(modules):
    ling = modules["m1"].forward("hello world", language="en")
    cond = modules["m2"].forward(b"\x00" * 96000, voice_profile_id="vp")
    out = modules["m3"].forward(ling, cond)
    assert isinstance(out, ProsodyAlignedTokens)
    T = len(ling.graphemes)
    assert len(out.tokens) == T
    assert len(out.durations) == T
    assert len(out.f0) == T
    assert len(out.energy) == T
    assert len(out.emotion) == _EMO_D                      # 4


# ---------------------------------------------------------------------------
# M4 — Guardrails
# ---------------------------------------------------------------------------


def test_m4_emits_secure_tokens_with_audit(modules):
    ling = modules["m1"].forward("test", language="en")
    cond = modules["m2"].forward(b"\x00" * 96000, voice_profile_id="vp")
    prosody = modules["m3"].forward(ling, cond)
    out = modules["m4"].forward(prosody, voice_profile_id="vp", content_text="test")
    assert isinstance(out, SecureAlignedTokens)
    assert out.audit_id.startswith("aud_")
    assert isinstance(out.watermark_seed, int)
    # ADD invariant: cryptographic tokens never in tensors.
    assert not hasattr(out, "consent_token")
    assert not hasattr(out, "signed_token")


# ---------------------------------------------------------------------------
# M5 — Generator
# ---------------------------------------------------------------------------


def test_m5_emits_80_channel_mel(modules):
    ling = modules["m1"].forward("test", language="en")
    cond = modules["m2"].forward(b"\x00" * 96000, voice_profile_id="vp")
    prosody = modules["m3"].forward(ling, cond)
    secure = modules["m4"].forward(prosody, voice_profile_id="vp", content_text="test")
    out = modules["m5"].forward(secure, cond)
    assert isinstance(out, MelSpectrogram)
    assert out.n_mels == _MEL_DIM                          # 80
    assert out.sample_rate == 24000
    assert out.hop_ms == 10.0
    # mel has shape [T, 80]
    assert len(out.mel) == _MEL_T
    assert len(out.mel[0]) == _MEL_DIM


# ---------------------------------------------------------------------------
# M6 — Vocoder
# ---------------------------------------------------------------------------


def test_m6_batch_path_returns_pcm_bytes(modules):
    ling = modules["m1"].forward("test", language="en")
    cond = modules["m2"].forward(b"\x00" * 96000, voice_profile_id="vp")
    prosody = modules["m3"].forward(ling, cond)
    secure = modules["m4"].forward(prosody, voice_profile_id="vp", content_text="test")
    mel = modules["m5"].forward(secure, cond)
    audio = modules["m6"].forward(mel, stream=False)
    assert isinstance(audio, (bytes, bytearray))
    # 200 frames × 10ms × 24000 Hz × 2 bytes/sample = 96000 bytes
    assert len(audio) == _MEL_T * 480


def test_m6_stream_path_yields_20ms_packets(modules):
    ling = modules["m1"].forward("test", language="en")
    cond = modules["m2"].forward(b"\x00" * 96000, voice_profile_id="vp")
    prosody = modules["m3"].forward(ling, cond)
    secure = modules["m4"].forward(prosody, voice_profile_id="vp", content_text="test")
    mel = modules["m5"].forward(secure, cond)
    packets = list(modules["m6"].forward(mel, stream=True))
    assert len(packets) == _MEL_T // 2                     # 2 frames per 20ms packet
    assert all(len(p) == 960 for p in packets)             # 20ms × 24kHz × 2 bytes


# ---------------------------------------------------------------------------
# Phase-1 contracts
# ---------------------------------------------------------------------------


def test_trifactor_streams_contract():
    """TriFactorSSL emits 4 explicitly disentangled streams (Phase 1 only)."""
    streams = TriFactorStreams(
        c_content=[[0.0] * 512 for _ in range(100)],
        s_identity=[0.0] * 256,
        p_prosody=[[0.0] * 64 for _ in range(100)],
        n_channel=[0.0] * 128,
    )
    # Each stream is a different shape — that's the disentanglement contract.
    assert len(streams.c_content) == 100
    assert len(streams.c_content[0]) == 512
    assert len(streams.s_identity) == 256
    assert len(streams.p_prosody) == 100
    assert len(streams.n_channel) == 128
