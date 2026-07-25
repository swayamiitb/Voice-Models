# Latency Budget — 240 ms TTFA on a single GPU

The end-to-end time-to-first-audio budget, decomposed by pipeline stage. The
natural gap between conversational turns is ~200 ms; beyond ~300 ms a
listener perceives a pause rather than a reply. **TTFA is the binding
metric**, and it is a function of pipeline structure (chunked emission) as
much as of model speed.

## The 240 ms budget

| Stage | Latency | Notes |
|---|---:|---|
| G2P / front-end | 5 ms | M1: WFST normalization + Misaki/Epitran G2P |
| Speaker encoder | 20 ms | M2: ECAPA-TDNN — **0 ms if cached** |
| Safety classifier | 15 ms | M4: ShieldGemma-2B INT8 |
| Flow-matching (8 steps) | 80 ms | M5: F5-TTS — ⚠ 100 steps would be 1000 ms; budget blown |
| Vocoder | 30 ms | M6: Vocos iSTFT |
| Network | 40 ms | client ↔ server round-trip |
| Jitter buffer | 50 ms | absorbs scheduling variance |
| **TOTAL** | **240 ms** | ✅ under the 300 ms perceptual threshold |

## Why flow matching is non-negotiable

| Generator | Solver steps | Estimated TTFA |
|---|---:|---:|
| Iterative diffusion | 50–1000 | 600–10,000 ms ❌ |
| Flow matching (F5-TTS) | 4–8 | 40–80 ms ✅ |
| Matcha-TTS OT-CFM | 4–8 | 40–80 ms ✅ |

That single change — flow matching instead of diffusion — is what makes
real-time streaming possible on one GPU. It is not a nice-to-have; **it is
the only reason the budget closes.**

## Why Vocos is non-negotiable

| Vocoder | Approach | Relative cost |
|---|---|---:|
| HiFi-GAN | progressive time-domain upsampling (24000 samples/sec, sample-by-sample) | 1× baseline |
| BigVGAN | same lineage, larger | ~1.2× baseline |
| **Vocos** | predict STFT magnitude + phase, single inverse STFT | **~0.01× baseline** |

Vocos removes the dominant cost of the vocoder stage while improving
high-frequency phase reconstruction. Footprint ~50 MB.

## Streaming is structural, not a flag

Module 6 begins emitting 20 ms packets as soon as ~300 ms of mel-spectrogram
is available. The listener hears the opening of the utterance while the rest
of the sentence is still being generated. This is what makes the perceptual
latency target achievable **without making the model itself faster**.
