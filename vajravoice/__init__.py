"""VajraVoice — a six-module neural text-to-speech engine.

Assembled from permissively-licensed open-source components, reproducing the
MAI-Voice-2 architectural class for Indian languages. Six modules —
linguistic, reference, fusion + prosody, guardrails, generator, vocoder —
compose behind fixed tensor contracts so any stage is independently
substitutable.

This package is deliberately import-light: heavy ML dependencies (torch,
transformers, speechbrain, ...) are lazy-imported inside each module so the
contract / config / pipeline layer loads on any machine. Real component
modules require `pip install -e ".[models]"` on a CUDA box; stub mode
(`configs/stub.yaml`) runs end-to-end with zero-weight dummy tensors.

Architecture reference: ARCHITECTURE.md (mirrors the workspace's IEEE-format
Two-Phase Breakthrough Technical Design Document).
"""

__version__ = "0.1.0"
__author__ = "swayamiitb"

__all__ = ["__version__"]
