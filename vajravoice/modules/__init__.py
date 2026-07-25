"""Module registry convenience exports."""

from .base import (
    FusionModule,
    GeneratorModule,
    GuardrailModule,
    LinguisticModule,
    Module,
    ReferenceModule,
    VocoderModule,
)
from .factory import build_module, build_pipeline_modules, get_class

__all__ = [
    "Module",
    "LinguisticModule",
    "ReferenceModule",
    "FusionModule",
    "GuardrailModule",
    "GeneratorModule",
    "VocoderModule",
    "build_module",
    "build_pipeline_modules",
    "get_class",
]
