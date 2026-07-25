# Memory Budget — Phase 2 inference stack on a single 32 GB GPU

Per-component VRAM accounting for the deployed mel-cascade. The ~11.5 GB
inference base leaves ~20.5 GB of deliberate headroom for KV cache, ODE
solver working allocations, and concurrent stream buffers — weights are
shared across streams while cache and buffer requirements scale linearly.

## Inference base

| Module | Component | Precision | VRAM |
|---|---|---|---:|
| M1 | Text pipeline (WeTextProcessing WFST + BERT-mini + ONNX XLM-R) | INT8 / FST | < 0.5 GB |
| M2 | Speaker encoder + WavLM-Large | FP16 | ~1.2 GB |
| M3 | Llama-3-8B (4-bit AWQ) + StyleTTS2 fusion | 4-bit | ~6.0 GB |
| M4 | Safety classifier (ShieldGemma-2B / LlamaGuard-3-1B) | INT8 | ~1.2 GB |
| M5 | Acoustic transformer (F5-TTS DiT + flow matching) | BF16 | ~2.5 GB |
| M6 | Vocoder (Vocos ConvNeXt + iSTFT) | FP32 | ~0.05 GB |
| **Inference base** | | | **~11.5 GB** |

## Headroom

| Use | VRAM |
|---|---:|
| KV cache (Llama-3-8B context window) | scales with concurrency |
| ODE solver working allocations (F5-TTS flow matching) | ~1–2 GB / stream |
| Audio stream buffers (20 ms packets × concurrency) | small |
| **Total headroom** | **~20.5 GB** |

## What dominates

Two components dominate and are therefore the first candidates for reduction
if the budget tightens:

1. **4-bit Llama-3-8B aligner (~6 GB)** — the single largest component. Its
   contribution should be established by ablation first; a smaller aligner
   is a legitimate substitution if the ablation does not justify the
   footprint.
2. **F5-TTS acoustic transformer (~2.5 GB)** — research substitution:
   Mamba-2 SSM block (linear-time O(N)) in place of quadratic attention.
   Requires retraining; benefit contingent on sequence lengths.

Every remaining module is under 1.5 GB, and the vocoder — the stage whose
cost the architecture most reduces — is under 100 MB.
