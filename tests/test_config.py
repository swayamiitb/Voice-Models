"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from vajravoice.config import VajraVoiceConfig

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_stub_config_loads():
    cfg = VajraVoiceConfig.from_yaml(CONFIGS_DIR / "stub.yaml")
    assert cfg.name == "vajravoice-stub"
    assert cfg.commercial is False
    assert cfg.sample_rate == 24000
    assert cfg.m1_linguistic.component == "StubLinguisticModule"
    assert cfg.m6_vocoder.component == "StubVocoderModule"


def test_default_config_loads():
    cfg = VajraVoiceConfig.from_yaml(CONFIGS_DIR / "default.yaml")
    assert cfg.name == "vajravoice-part2"
    assert cfg.m1_linguistic.component == "Part2LinguisticModule"
    assert cfg.m2_reference.component == "Part2ReferenceModule"
    assert cfg.m3_prosody.component == "Part2ProsodyModule"
    assert cfg.m4_guardrails.component == "Part2GuardrailModule"
    assert cfg.m5_generator.component == "Part2GeneratorModule"
    assert cfg.m6_vocoder.component == "Part2VocoderModule"


def test_default_config_has_part2_component_kwargs():
    cfg = VajraVoiceConfig.from_yaml(CONFIGS_DIR / "default.yaml")
    # Verify the Part2.md component selections are present.
    assert cfg.m5_generator.kwargs["acoustic_transformer"] == "F5-TTS"
    assert cfg.m5_generator.kwargs["f5_checkpoint"] == "SWivid/F5-TTS"
    assert cfg.m6_vocoder.kwargs["vocoder"] == "Vocos"
    assert cfg.m2_reference.kwargs["speaker_encoder"] == "ECAPA-TDNN"


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        VajraVoiceConfig.from_yaml(CONFIGS_DIR / "nonexistent.yaml")


def test_from_env_falls_back_to_stub(monkeypatch):
    monkeypatch.delenv("VAJRAVOICE_CONFIG", raising=False)
    cfg = VajraVoiceConfig.from_env()
    assert cfg.name == "vajravoice-stub"
