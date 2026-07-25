# VajraVoice

> A six-module neural text-to-speech engine, assembled entirely from permissively-licensed open-source components, reproducing the **MAI-Voice-2** architectural class for Indian languages.

[![tests](https://github.com/swayamiitb/Voice-Models/actions/workflows/test.yml/badge.svg)](.github/workflows/test.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

---

## What this is

VajraVoice takes the **public** MAI-Voice-2 architecture — Microsoft's current
frontier text-to-speech system — and rebuilds the same capability class from
open-source models that you own end-to-end. The contribution is **the
decomposition**: MAI-Voice-2's pipeline is split into **6 modules and 18
blocks**, and for every block this repo knows *what it does, which
open-source component implements it, what that component's licence permits,
and whether it works for Indian languages*.

> **The architecture is public. The decomposition isn't.**
> Anyone can draw the block diagram. Almost nobody can name the five
> language-specific blocks. That specificity is the credential.

This is **not** a claim to have cracked MAI-Voice-2's code or weights. There
is no public evidence MAI-Voice-2 uses self-supervised learning internally;
its use in VajraVoice is a performance-driven design choice, not a claim
about Microsoft's undisclosed internals.

---

## The pipeline

Six modules behind **fixed tensor contracts** — any stage is independently
substitutable. Two phases share one backbone: Phase 1 builds a disentangled
self-supervised representation core; Phase 2 ports it onto a faster
mel-cascade that closes the latency budget on a single GPU.

```
text ─▶ M1 Linguistic ─▶ M3 Fusion + Prosody ─▶ M4 Guardrails ─▶ M5 Generator ─▶ M6 Vocoder ─▶ 24 kHz PCM
                                 ▲                                                      ▲
reference (5–60s) ──────▶ M2 Reference Engine ──────────────────────────────────────┘
```

### Module → open-source component map (Part2.md)

| Module | Role | Open-source component | License |
|---|---|---|---|
| **M1** Linguistic | text normalization, G2P, UMIM | WeTextProcessing + Misaki + Epitran + XLM-RoBERTa | Apache / MIT |
| **M2** Reference | zero-shot voice cloning | ECAPA-TDNN (SpeechBrain) + WavLM-Large FP16 | Apache / MIT |
| **M3** Fusion + Prosody | cross-modal fusion, global context, emotion | StyleTTS2 + Llama-3-8B (4-bit AWQ) + Emotion2Vec | MIT / Llama Community / Apache |
| **M4** Guardrails | consent, content moderation, provenance | ShieldGemma-2B (INT8) + ECAPA match + AudioSeal | Gemma / Apache / MIT |
| **M5** Generator | acoustic generation | Matcha-TTS (OT-CFM) + F5-TTS (DiT + flow matching) | MIT / MIT-code ⚠ CC-BY-NC weights |
| **M6** Vocoder | waveform synthesis + streaming | Vocos (ConvNeXt + inverse STFT) | MIT |

---

## Why this matters for Indian languages

Microsoft's MAI-Voice-2 ships **15 languages; one is Indian (Hindi)**. The
languages it does not serve collectively cover roughly **half a billion
speakers**. This gap is not primarily a data problem — it is architectural,
with three distinct components:

1. **Orthographic depth.** Devanagari looks phonetic and is not. `कमल` reads
   `ka-ma-la` symbol by symbol but is spoken `kamal`. Schwa deletion is
   systematic, phonologically conditioned, and written nowhere in the script.
2. **Phonological inventory.** Indian languages contrast retroflex vs dental
   (ट /ʈ/ vs त /t̪/) and have breathy-voiced aspirates (घ /ɡʱ/) that English
   lacks. Inventories designed around English cannot represent these and
   silently merge them.
3. **Prosodic rhythm.** Indian languages are syllable-timed; English is
   stress-timed. A duration model trained on English rhythm produces correct
   phonemes with **incorrect timing** — audibly non-native speech in which no
   individual sound is wrong.

These are properties of the model, not of the corpus. **More data does not
fix them** — it makes the wrong prior more confident. The fix is knowing
which of the 18 blocks to change. Only **5** are language-specific.

---

## The two-phase program

| Phase | Goal | Components | VRAM |
|---|---|---|---|
| **Phase 1** | Disentangled representation core | 3 frozen SSL teachers → 100–140M TriFactorSSL student (4 streams) + semantic-unit layer | ~1.1–1.3 B params, ≤ 28 GiB |
| **Phase 2** | Speed + footprint | Flow-matching DiT (4–8 steps), Vocos iSTFT, FlashAttention-2 | ~11.5 GB inference base, ≤ 32 GB |

The ordering is the contribution: build the disentangled SSL core **first**,
then pursue raw speed. The mandatory SSL ablation in `docs/design_decisions.md`
is what makes the two-phase thesis falsifiable rather than aspirational.

---

## Quick start

### Clone & install (stub mode — no GPU needed)

```bash
git clone https://github.com/swayamiitb/Voice-Models.git
cd Voice-Models
python -m pip install -e ".[api,audio,dev]"
```

### Run the test suite

```bash
make test        # → 25 tests, all green, no torch/transformers required
```

### Synthesize audio (stub mode)

```bash
python -m scripts.synthesize \
  --text "Namaskar, aaj aapan ek navin prakalp baddal bolu." \
  --config configs/stub.yaml \
  --out out.wav
```

### Run the benchmark harness

```bash
python -m scripts.benchmark --config configs/stub.yaml --n 20
```

### Start the API server

```bash
make serve
# → http://localhost:8000  (Swagger UI at /docs)
# POST /v1/voices/enroll
# POST /v1/audio/speech
# WS   /v1/audio/stream
```

### Full ML stack (CUDA box only)

```bash
python -m pip install -e ".[models]"    # ~5 GB of wheels (torch, transformers, speechbrain, ...)
python -m scripts.synthesize --text "नमस्कार" \
  --voice vp_marathi_mf_asha --config configs/default.yaml \
  --reference ref.wav --out marathi.mp3
```

---

## Repository layout

```
Voice-Models/
├── vajravoice/             ← The Python package
│   ├── contracts.py        ← Inter-module tensor contracts (ADD Table IV)
│   ├── config.py           ← YAML-driven pipeline config
│   ├── pipeline.py         ← VajraVoicePipeline (M1→M6 wiring + assertions)
│   ├── modules/            ← Six modules (M1–M6) + stubs + factory
│   ├── api/                ← FastAPI server + Pydantic schemas
│   └── utils/              ← Audio helpers + licensing registry
├── tests/                  ← Contract, pipeline, config, licensing tests (25 tests)
├── configs/                ← stub.yaml (CI) + default.yaml (Part2.md real components)
├── scripts/                ← CLI: synthesize, enroll, benchmark
├── benchmarks/             ← Decision matrix, memory + latency budgets
├── docs/                   ← ARCHITECTURE, licensing_audit, design_decisions, roadmap
└── .github/workflows/      ← CI: pytest in stub mode on every push
```

---

## Key design decisions

The six ADRs (full text in `docs/design_decisions.md`):

1. **Representation-first ordering** — build the disentangled SSL core (Phase 1) before any speed optimization (Phase 2).
2. **Flow matching over iterative diffusion** — 4–8 evaluations vs 50–1000.
3. **Vocos iSTFT over HiFi-GAN** — ~90% vocoder latency cut, superior HF phase.
4. **ECAPA-TDNN + WavLM over a pooled embedding** — timbre + time-varying cadence.
5. **Fail-closed pre-generation guardrail** — refusal precedes generation; unmade audio cannot leak.
6. **Permissively-licensed stack** — MIT/Apache/BSD on shipped weights; F5-TTS checkpoint + eSpeak-ng replaced by design.

---

## Licensing

This repository's own code is **MIT-licensed**. Individual neural components
carry their own licences — see `docs/licensing_audit.md` for the per-component
commercial-safety posture. In particular:

- **F5-TTS released weights are CC-BY-NC** (non-commercial). Commercial use
  requires a permissively-licensed replacement checkpoint (e.g. OpenF5-TTS,
  Apache) or a retrain on owned data. The architecture is unaffected; only
  the checkpoint is.
- **eSpeak-ng / phonemizer are GPL-3.0**. The pipeline replaces these by
  design with a permissively-licensed Indic articulatory ruleset so the front
  end ships without copyleft exposure.
- **VoxCeleb-trained speaker encoders** (ECAPA-TDNN) inherit the dataset's
  research-only terms. Commercial deployment may require retraining on
  consented data — the model licence does not launder the dataset terms.

The licensing guard at `vajravoice/utils/licensing.py` enforces this at load
time: selecting a non-ship component in a `commercial: true` config raises
`ShipWarning` immediately, not at the term sheet.

---

## Honest scope

This is an **architecture & integration portfolio**, not a deployed product:

- The pipeline wiring, contracts, API, configs, and tests are **real and
  runnable** (25/25 tests pass on a bare interpreter).
- The heavy ML component hooks (`Part2LinguisticModule`, `Part2ReferenceModule`,
  ...) are **scaffolded against the real OSS libraries** but require
  `pip install -e ".[models]"` on a CUDA box with ≥ 11.5 GB VRAM to actually
  load weights.
- No model was trained from scratch. Part2.md is a *component-selection /
  architecture-assembly* doc, not a *training* doc. The honest framing is
  *"assembled and integrated SOTA pretrained components behind fixed
  contracts"*, which is itself the architectural credential.

---

## References

The MAI-Voice-2 architecture, the two-phase program, and the licensing audit
are documented in depth in:

- **`ARCHITECTURE.md`** — the full technical design (mirrors the IEEE-format
  Two-Phase Breakthrough Technical Design Document).
- **`docs/licensing_audit.md`** — the per-component CLEAN / CHECKPOINT-REPLACEMENT /
  REPLACED-BY-DESIGN table.
- **`docs/design_decisions.md`** — the six ADRs.
- **`docs/roadmap.md`** — Phase 1 → Phase 2 → Indian-language expansion.
- **`benchmarks/`** — the 11-model commercial TTS decision matrix, the 11.5 GB
  / 32 GB memory budget, and the 240 ms TTFA latency decomposition.

## Citation

If you reference this work, please cite:

```bibtex
@misc{vajravoice2026,
  title  = {VajraVoice: A Six-Module Open-Source Neural TTS Engine for Indian Languages},
  author = {swayamiitb},
  year   = {2026},
  url    = {https://github.com/swayamiitb/Voice-Models}
}
```

## License

MIT — see [LICENSE](LICENSE).
