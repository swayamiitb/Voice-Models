"""Licensing registry + commercial-safety guards.

The deployment licence is governed by the licence attached to each CHECKPOINT,
which frequently differs from the licence on the repository — the weights are
the binding term. This module encodes the licensing audit from
``docs/licensing_audit.md`` so that selecting a non-ship component in a
``commercial: true`` config raises immediately at load time rather than
discovered at the term sheet.

The registry is the source of truth for the three-bucket classification used
throughout the workspace's TDD/ADD docs:

  * CLEAN                  — ship freely (MIT/Apache/BSD on both code + weights)
  * CHECKPOINT_REPLACEMENT — code permissive, released weights non-commercial
  * REPLACED_BY_DESIGN     — copyleft (GPL/AGPL) — replaced in the build
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


class ShipWarning(UserWarning):
    """Raised when a non-ship component is selected in a commercial config."""


@dataclass(frozen=True)
class ComponentLicence:
    component: str
    code_licence: str
    weights_licence: str
    bucket: str            # "CLEAN" | "CHECKPOINT_REPLACEMENT" | "REPLACED_BY_DESIGN"
    note: str = ""


# --- The registry (mirrors docs/licensing_audit.md) --------------------------
REGISTRY: dict[str, ComponentLicence] = {
    # M1 — Linguistic
    "WeTextProcessing": ComponentLicence("WeTextProcessing", "Apache-2.0", "—", "CLEAN"),
    "Misaki":           ComponentLicence("Misaki", "MIT", "—", "CLEAN"),
    "Epitran":          ComponentLicence("Epitran", "MIT", "—", "CLEAN"),
    "XLM-RoBERTa":      ComponentLicence("XLM-RoBERTa", "MIT", "MIT", "CLEAN"),
    "eSpeak-ng":        ComponentLicence(
        "eSpeak-ng", "GPL-3.0", "—", "REPLACED_BY_DESIGN",
        "Replaced by a permissively-licensed Indic articulatory ruleset so "
        "the front end ships without copyleft exposure.",
    ),
    "phonemizer":       ComponentLicence(
        "phonemizer", "GPL-3.0", "—", "REPLACED_BY_DESIGN",
        "GPL wrapper around eSpeak-ng — same copyleft trap.",
    ),

    # M2 — Reference
    "ECAPA-TDNN":       ComponentLicence(
        "ECAPA-TDNN", "Apache-2.0", "Apache-2.0", "CLEAN",
        "VoxCeleb provenance: research-only dataset. Commercial deployment "
        "may require retraining on consented data — the model licence does "
        "not launder the dataset terms.",
    ),
    "WavLM":            ComponentLicence("WavLM", "MIT", "MIT", "CLEAN",
                                         "Verify Microsoft artifact + data terms."),
    "WavLM-SV":         ComponentLicence(
        "WavLM-SV", "MIT", "CC-BY-SA", "CHECKPOINT_REPLACEMENT",
        "Speaker-verification weights are copyleft — not MIT.",
    ),
    "Silero-VAD":       ComponentLicence("Silero-VAD", "MIT", "MIT", "CLEAN"),

    # M3 — Prosody
    "StyleTTS2":        ComponentLicence(
        "StyleTTS2", "MIT", "MIT", "CLEAN",
        "Pretrained use carries a disclose-synthesis + voice-consent term.",
    ),
    "Llama-3-8B":       ComponentLicence(
        "Llama-3-8B", "Llama-3 Community", "Llama-3 Community", "CHECKPOINT_REPLACEMENT",
        "Provider community licence permitting commercial use subject to "
        "conditions — read against deployment.",
    ),
    "Emotion2Vec":      ComponentLicence("Emotion2Vec", "Apache-2.0", "Apache-2.0", "CLEAN"),

    # M4 — Guardrails
    "ShieldGemma-2B":   ComponentLicence(
        "ShieldGemma-2B", "Gemma", "Gemma", "CHECKPOINT_REPLACEMENT",
        "Gemma community licence — read against deployment.",
    ),
    "LlamaGuard-3-1B":  ComponentLicence(
        "LlamaGuard-3-1B", "Llama-3 Community", "Llama-3 Community", "CHECKPOINT_REPLACEMENT",
    ),
    "AudioSeal":        ComponentLicence("AudioSeal", "MIT", "MIT", "CLEAN",
                                         "Permissive on both code and weights — uncommon."),

    # M5 — Generator
    "Matcha-TTS":       ComponentLicence("Matcha-TTS", "MIT", "MIT", "CLEAN"),
    "F5-TTS":           ComponentLicence(
        "F5-TTS", "MIT", "CC-BY-NC", "CHECKPOINT_REPLACEMENT",
        "Code permissive; released weights non-commercial (training-corpus "
        "provenance). Commercial use needs a community/permissive checkpoint "
        "(e.g. OpenF5-TTS Apache) or a retrain on owned data. Architecture "
        "unaffected; only the checkpoint is.",
    ),
    "OpenF5-TTS":       ComponentLicence("OpenF5-TTS", "Apache-2.0", "Apache-2.0", "CLEAN",
                                         "Apache replacement for F5-TTS weights."),
    "Mamba-2":          ComponentLicence("Mamba-2", "Apache-2.0", "Apache-2.0", "CLEAN"),

    # M6 — Vocoder
    "Vocos":            ComponentLicence("Vocos", "MIT", "MIT", "CLEAN"),
    "BigVGAN":          ComponentLicence("BigVGAN", "MIT", "MIT", "CLEAN"),

    # Phase-1 SSL teachers
    "Wav2Vec2-XLS-R":   ComponentLicence("Wav2Vec2-XLS-R", "Apache-2.0", "Apache-2.0", "CLEAN"),
    "W2v-BERT-2.0":     ComponentLicence("W2v-BERT-2.0", "MIT", "MIT", "CLEAN"),

    # Phase-1 codec
    "DualCodec":        ComponentLicence(
        "DualCodec", "MIT", "research", "CHECKPOINT_REPLACEMENT",
        "MIT code/arch usable as reference. Do not ship current pretrained "
        "checkpoint unless weight + Emilia-data audit passes.",
    ),
    "Qwen3-TTS":        ComponentLicence(
        "Qwen3-TTS", "Apache-2.0", "Qwen Community", "CHECKPOINT_REPLACEMENT",
        "Community licence; read against deployment.",
    ),

    # End-to-end alternatives (from the open-source table)
    "CosyVoice2":       ComponentLicence("CosyVoice2", "Apache-2.0", "Apache-2.0", "CLEAN"),
    "Chatterbox":       ComponentLicence("Chatterbox", "MIT", "MIT", "CLEAN"),
    "Orpheus":          ComponentLicence("Orpheus", "Apache-2.0", "Apache-2.0", "CLEAN"),
    "OpenVoice-v2":     ComponentLicence("OpenVoice-v2", "MIT", "MIT", "CLEAN"),
    "IndexTTS-2":       ComponentLicence("IndexTTS-2", "Apache-2.0", "Apache-2.0", "CLEAN"),
    "VoxCPM2":          ComponentLicence("VoxCPM2", "Apache-2.0", "Apache-2.0", "CLEAN"),
}


def assert_ship_safe(*, component: str, weights_id: str | None = None, commercial: bool) -> None:
    """Raise ShipWarning if `component` (or its `weights_id`) is not shippable in a
    commercial build. No-op for non-commercial configs.

    Selected component is matched by `component` first, then by `weights_id`
    (so e.g. F5-TTS code with an OpenF5-TTS checkpoint is CLEAN).
    """
    if not commercial:
        return

    # If a specific weights id is supplied and it overrides the component bucket,
    # honour that (e.g. F5-TTS + OpenF5-TTS weights ⇒ CLEAN).
    if weights_id and weights_id in REGISTRY:
        rec = REGISTRY[weights_id]
    else:
        rec = REGISTRY.get(component)

    if rec is None:
        warnings.warn(
            f"Component '{component}' not in licensing registry — verify before commercial use.",
            ShipWarning, stacklevel=2,
        )
        return

    if rec.bucket == "CLEAN":
        return
    if rec.bucket in ("CHECKPOINT_REPLACEMENT", "REPLACED_BY_DESIGN"):
        raise ShipWarning(
            f"Component '{component}' is {rec.bucket} (code={rec.code_licence}, "
            f"weights={rec.weights_licence}). {rec.note} "
            f"Select a CLEAN alternative (see docs/licensing_audit.md) before "
            f"shipping commercially."
        )
