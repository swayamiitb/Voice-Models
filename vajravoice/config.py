"""YAML-driven config for the VajraVoice pipeline.

The pipeline is fully config-driven: which component fills each module slot is
chosen by a YAML file, not by editing code. This is the architectural property
the ADD relies on for independent substitutability ("any stage swappable
behind fixed tensor contracts").

Two reference configs ship:
  * configs/stub.yaml     — stub components; zero weights; runs anywhere.
  * configs/default.yaml  — the real Part2.md component picks; needs the heavy ML stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "stub.yaml"


@dataclass
class ModuleConfig:
    """A single module's component selection + kwargs."""

    component: str                        # factory key, e.g. "Part2LinguisticModule" or "StubLinguisticModule"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VajraVoiceConfig:
    """Top-level pipeline config."""

    name: str
    description: str = ""
    commercial: bool = False              # if True, non-ship components raise ShipWarning
    sample_rate: int = 24000
    seed: int = 17

    # Module slots
    m1_linguistic: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(component="StubLinguisticModule")
    )
    m2_reference: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(component="StubReferenceModule")
    )
    m3_prosody: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(component="StubProsodyModule")
    )
    m4_guardrails: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(component="StubGuardrailModule")
    )
    m5_generator: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(component="StubGeneratorModule")
    )
    m6_vocoder: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(component="StubVocoderModule")
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VajraVoiceConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        modules = raw.pop("modules", {}) or {}
        kw = {
            "name": raw.get("name", path.stem),
            "description": raw.get("description", ""),
            "commercial": raw.get("commercial", False),
            "sample_rate": raw.get("sample_rate", 24000),
            "seed": raw.get("seed", 17),
        }
        for slot in (
            "m1_linguistic", "m2_reference", "m3_prosody",
            "m4_guardrails", "m5_generator", "m6_vocoder",
        ):
            if slot in modules:
                mc = modules[slot]
                kw[slot] = ModuleConfig(
                    component=mc["component"],
                    kwargs=mc.get("kwargs", {}) or {},
                )
        return cls(**kw)

    @classmethod
    def from_env(cls) -> "VajraVoiceConfig":
        """Load config path from VAJRAVOICE_CONFIG env var, else fall back to stub."""
        p = os.environ.get("VAJRAVOICE_CONFIG")
        if p and Path(p).exists():
            return cls.from_yaml(p)
        return cls.from_yaml(DEFAULT_CONFIG_PATH)
