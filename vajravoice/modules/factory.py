"""Module factory — instantiate a module by its config key.

The pipeline is config-driven: YAML picks the component for each slot. This
file is the single registry that maps a config key (e.g.
``"Part2LinguisticModule"``) to the implementing class. Adding a new
component is a one-line change here + one config edit.
"""

from __future__ import annotations

from ..config import VajraVoiceConfig
from ..utils.licensing import assert_ship_safe
from .base import (
    FusionModule,
    GeneratorModule,
    GuardrailModule,
    LinguisticModule,
    Module,
    ReferenceModule,
    VocoderModule,
)

# Lazy imports of concrete modules — done at registration time only, so that
# importing this file alone never pulls torch/transformers/etc.
_REGISTRY: dict[str, type[Module]] = {}


def _register() -> None:
    # Stubs first — always available.
    from .stubs import (
        StubGeneratorModule,
        StubGuardrailModule,
        StubLinguisticModule,
        StubProsodyModule,
        StubReferenceModule,
        StubVocoderModule,
    )
    _REGISTRY.update({
        StubLinguisticModule.component_key: StubLinguisticModule,
        StubReferenceModule.component_key: StubReferenceModule,
        StubProsodyModule.component_key: StubProsodyModule,
        StubGuardrailModule.component_key: StubGuardrailModule,
        StubGeneratorModule.component_key: StubGeneratorModule,
        StubVocoderModule.component_key: StubVocoderModule,
    })

    # Real components — imported only if their heavy deps are installed.
    try:
        from .m1_linguistic import Part2LinguisticModule
        from .m2_reference import Part2ReferenceModule
        from .m3_prosody import Part2ProsodyModule
        from .m4_guardrails import Part2GuardrailModule
        from .m5_generator import Part2GeneratorModule
        from .m6_vocoder import Part2VocoderModule
        _REGISTRY.update({
            Part2LinguisticModule.component_key: Part2LinguisticModule,
            Part2ReferenceModule.component_key: Part2ReferenceModule,
            Part2ProsodyModule.component_key: Part2ProsodyModule,
            Part2GuardrailModule.component_key: Part2GuardrailModule,
            Part2GeneratorModule.component_key: Part2GeneratorModule,
            Part2VocoderModule.component_key: Part2VocoderModule,
        })
    except ImportError:
        # Heavy stack not installed — fine. Stub configs still work.
        pass


def get_class(component_key: str) -> type[Module]:
    if not _REGISTRY:
        _register()
    if component_key not in _REGISTRY:
        raise KeyError(
            f"Unknown component '{component_key}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[component_key]


def build_module(slot: str, component: str, kwargs: dict, *, commercial: bool) -> Module:
    """Instantiate one module. Injects ``commercial`` so licensing guards fire."""
    cls = get_class(component)
    kwargs = {**kwargs, "commercial": commercial}
    return cls(**kwargs)


def build_pipeline_modules(config: VajraVoiceConfig) -> tuple[
    LinguisticModule, ReferenceModule, FusionModule,
    GuardrailModule, GeneratorModule, VocoderModule,
]:
    """Build all six modules for a config in pipeline order (M1..M6)."""
    commercial = config.commercial
    m1 = build_module("m1", config.m1_linguistic.component, config.m1_linguistic.kwargs, commercial=commercial)
    m2 = build_module("m2", config.m2_reference.component, config.m2_reference.kwargs, commercial=commercial)
    m3 = build_module("m3", config.m3_prosody.component, config.m3_prosody.kwargs, commercial=commercial)
    m4 = build_module("m4", config.m4_guardrails.component, config.m4_guardrails.kwargs, commercial=commercial)
    m5 = build_module("m5", config.m5_generator.component, config.m5_generator.kwargs, commercial=commercial)
    m6 = build_module("m6", config.m6_vocoder.component, config.m6_vocoder.kwargs, commercial=commercial)
    return m1, m2, m3, m4, m5, m6
