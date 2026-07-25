"""Abstract base classes for the six neural modules.

Each module consumes the previous module's contract and emits its own. The
contract types are defined in ``vajravoice.contracts``. Concrete
implementations live in ``m1_linguistic.py`` … ``m6_vocoder.py``; stub
implementations live in ``modules/stubs.py`` and are used by the test suite
and CI (they never import torch / transformers / speechbrain).

The factory ``build_module()`` instantiates a module by its config key, which
keeps component selection in YAML, not in code.
"""

from __future__ import annotations

import abc
from typing import Any

from ..contracts import (
    AcousticConditioning,
    LinguisticEmbedding,
    MelSpectrogram,
    ProsodyAlignedTokens,
    SecureAlignedTokens,
)


class Module(abc.ABC):
    """Common base. Concrete modules override the relevant ``forward_*``."""

    #: Config key — overridden by subclasses so the factory can find them.
    component_key: str = "base"

    #: Phase this component belongs to: "shared", "phase1", "phase2".
    phase: str = "shared"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._loaded = False

    def load(self) -> "Module":
        """Lazy-load weights. Called once before the first forward pass.

        Subclasses should override to do real loading (HF downloads, ONNX
        sessions, etc.). Default is a no-op for stubs.
        """
        self._loaded = True
        return self

    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# M1 — Linguistic processing
# ---------------------------------------------------------------------------


class LinguisticModule(Module):
    """M1: raw text → LinguisticEmbedding."""

    component_key = "linguistic"

    @abc.abstractmethod
    def forward(self, text: str, language: str = "auto") -> LinguisticEmbedding: ...


# ---------------------------------------------------------------------------
# M2 — Reference engine (zero-shot cloning)
# ---------------------------------------------------------------------------


class ReferenceModule(Module):
    """M2: 5–60 s reference audio → AcousticConditioning."""

    component_key = "reference"

    @abc.abstractmethod
    def forward(self, reference_audio: Any, voice_profile_id: str | None = None) -> AcousticConditioning: ...


# ---------------------------------------------------------------------------
# M3 — Fusion + prosody
# ---------------------------------------------------------------------------


class FusionModule(Module):
    """M3: LinguisticEmbedding + AcousticConditioning → ProsodyAlignedTokens."""

    component_key = "fusion"

    @abc.abstractmethod
    def forward(
        self,
        linguistic: LinguisticEmbedding,
        conditioning: AcousticConditioning,
    ) -> ProsodyAlignedTokens: ...


# ---------------------------------------------------------------------------
# M4 — Guardrails (consent + content + provenance)
# ---------------------------------------------------------------------------


class GuardrailModule(Module):
    """M4: ProsodyAlignedTokens + consent claim → SecureAlignedTokens (fail-closed)."""

    component_key = "guardrail"

    @abc.abstractmethod
    def forward(
        self,
        tokens: ProsodyAlignedTokens,
        voice_profile_id: str,
        content_text: str,
    ) -> SecureAlignedTokens: ...


# ---------------------------------------------------------------------------
# M5 — Acoustic generation
# ---------------------------------------------------------------------------


class GeneratorModule(Module):
    """M5: SecureAlignedTokens → MelSpectrogram ([T, 80] log-mel)."""

    component_key = "generator"

    @abc.abstractmethod
    def forward(self, tokens: SecureAlignedTokens, conditioning: AcousticConditioning) -> MelSpectrogram: ...


# ---------------------------------------------------------------------------
# M6 — Vocoder + streaming
# ---------------------------------------------------------------------------


class VocoderModule(Module):
    """M6: MelSpectrogram → 24 kHz PCM (optionally streamed as 20 ms packets)."""

    component_key = "vocoder"

    @abc.abstractmethod
    def forward(self, mel: MelSpectrogram, stream: bool = False) -> Any: ...
