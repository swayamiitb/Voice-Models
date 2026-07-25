"""Licensing audit tests — verify the commercial-safety guards fire correctly."""

from __future__ import annotations

import pytest

from vajravoice.utils.licensing import REGISTRY, ShipWarning, assert_ship_safe


def test_clean_components_pass_commercial():
    """MIT/Apache/BSD components must pass without warning under commercial build."""
    for clean in ["Vocos", "Matcha-TTS", "ECAPA-TDNN", "Wav2Vec2-XLS-R", "AudioSeal", "WavLM"]:
        assert_ship_safe(component=clean, commercial=True)   # no exception


def test_f5_tts_weights_blocked_commercial():
    """F5-TTS released weights are CC-BY-NC — must raise under commercial."""
    with pytest.raises(ShipWarning, match="CHECKPOINT_REPLACEMENT"):
        assert_ship_safe(component="F5-TTS", commercial=True)


def test_f5_tts_with_openf5_checkpoint_is_clean():
    """F5-TTS code with OpenF5-TTS (Apache) weights is shippable."""
    # weights_id overrides the component bucket when explicitly provided.
    assert_ship_safe(
        component="F5-TTS", weights_id="OpenF5-TTS", commercial=True,
    )


def test_espeak_ng_blocked_commercial():
    """eSpeak-ng is GPL-3.0 — replaced by design, must raise under commercial."""
    with pytest.raises(ShipWarning, match="REPLACED_BY_DESIGN"):
        assert_ship_safe(component="eSpeak-ng", commercial=True)


def test_non_commercial_skips_guard():
    """Research/non-commercial builds load anything without complaint."""
    assert_ship_safe(component="F5-TTS", commercial=False)
    assert_ship_safe(component="eSpeak-ng", commercial=False)


def test_unknown_component_warns():
    with pytest.warns(ShipWarning, match="not in licensing registry"):
        assert_ship_safe(component="SomeUnknownThing", commercial=True)


def test_registry_covers_all_six_modules():
    """Every Part2.md component pick must be in the registry."""
    required = [
        "WeTextProcessing", "Misaki", "Epitran", "XLM-RoBERTa",      # M1
        "ECAPA-TDNN", "WavLM", "Silero-VAD",                          # M2
        "StyleTTS2", "Llama-3-8B", "Emotion2Vec",                     # M3
        "ShieldGemma-2B", "LlamaGuard-3-1B", "AudioSeal",             # M4
        "Matcha-TTS", "F5-TTS", "Mamba-2",                            # M5
        "Vocos", "BigVGAN",                                           # M6
    ]
    for c in required:
        assert c in REGISTRY, f"missing registry entry: {c}"
        assert REGISTRY[c].bucket in ("CLEAN", "CHECKPOINT_REPLACEMENT", "REPLACED_BY_DESIGN")


def test_licensing_audit_buckets_are_balanced():
    """Sanity: the registry has a real mix of buckets, not all-CLEAN."""
    buckets = [r.bucket for r in REGISTRY.values()]
    assert buckets.count("CLEAN") >= 10
    assert buckets.count("CHECKPOINT_REPLACEMENT") >= 3
    assert buckets.count("REPLACED_BY_DESIGN") >= 2     # eSpeak-ng, phonemizer
