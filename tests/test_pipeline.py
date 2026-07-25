"""Pipeline integration tests — full M1→M6 cascade in stub mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from vajravoice.config import VajraVoiceConfig
from vajravoice.pipeline import VajraVoicePipeline

STUB_CONFIG = Path(__file__).parent.parent / "configs" / "stub.yaml"


@pytest.fixture
def pipeline():
    return VajraVoicePipeline(VajraVoiceConfig.from_yaml(STUB_CONFIG)).load()


def test_pipeline_loads_without_heavy_deps(pipeline):
    """The pipeline must build on a bare interpreter (no torch/transformers)."""
    assert pipeline._loaded
    assert pipeline.m1.__class__.__name__ == "StubLinguisticModule"


def test_synthesize_round_trip(pipeline):
    result = pipeline.synthesize(
        text="Namaskar, aaj aapan ek navin prakalp baddal bolu.",
        voice_profile_id="vp_marathi_mf_asha",
    )
    assert result.request_id.startswith("req_")
    assert result.audio_seconds > 0
    assert result.audio_seconds < 60                      # sanity bound
    assert result.generation_ms > 0
    assert result.rtf >= 0
    assert result.watermark["scheme"] == "AudioSeal"
    assert result.audit_id.startswith("aud_")


def test_stream_yields_pcm_packets(pipeline):
    packets = list(pipeline.stream(
        text="test streaming",
        voice_profile_id="vp_marathi_mm_vivek",
    ))
    assert len(packets) > 0
    assert all(isinstance(p, (bytes, bytearray)) for p in packets)
    assert all(len(p) == 960 for p in packets)             # 20ms × 24kHz × 2 bytes


def test_pipeline_repr_lists_all_modules(pipeline):
    s = repr(pipeline)
    for slot in ("m1=", "m2=", "m3=", "m4=", "m5=", "m6="):
        assert slot in s
