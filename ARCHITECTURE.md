# VajraVoice — Architecture

> A six-module neural text-to-speech engine assembled from permissively-licensed
> open-source components, reproducing the **MAI-Voice-2** architectural class.

This document specifies the system architecture: the shared six-module
backbone, the two-phase program (representation first, speed second), the
fixed tensor contracts between modules, the deployment topology, and the
benchmark that justifies MAI-Voice-2 as the reference architecture. It
mirrors the IEEE-format *Two-Phase Breakthrough* Technical Design Document.

---

## 1. The thesis, in one breath

> The frontier gap on Indian languages is a **representation failure**, not a
> data failure. So VajraVoice builds the disentangled self-supervised speech
> core **first** (Phase 1), and only then pursues raw speed (Phase 2). Every
> other open pipeline reaches for the cheapest generator first and inherits
> an entangled representation it can never fully correct.

The contribution is the **ordering**. The mandatory SSL ablation in
`docs/design_decisions.md` makes the two-phase thesis falsifiable rather
than aspirational.

---

## 2. The shared six-module backbone

Both phases are instantiations of one modular cascade. The module boundaries
and their **fixed tensor contracts** are constant across phases; what changes
is the internal realization of M2 (reference), M5 (generator), and M6
(vocoder). M1 (linguistic), M3 (fusion+prosody), and M4 (guardrails) are
shared.

```
                                                    ┌─ Phase 1: sem → acoustic (masked diffusion)
                            M5 Generator ───────────┤
                            (P1: TriFactorSSL        └─ Phase 2: flow-matching DiT (4–8 steps)
M1 Linguistic   ─▶  M3 Fusion + Prosody  ─▶  M4 Guardrails  ─▶                  │
(TN · G2P · UMIM)    (cross-modal align,           (authorize ·         M6 Vocoder   ─▶ 24 kHz PCM
                     global context,                fail-closed)        (P1: causal codec
                     emotion)                                            P2: Vocos iSTFT)
                                                                                        ▲
                                                    M2 Reference Engine ───────────────────┘
                                                    (P1: TriFactorSSL dual-path  |  P2: ECAPA-TDNN + WavLM)
                                                        ▲
                                                        │
                                                  reference (5–60 s)
```

The architectural property that makes this work: because the interfaces are
fixed tensor contracts asserted at load time, any stage is independently
substitutable. Phase-1 ↔ Phase-2 swapping, per-module component upgrades,
and the per-component licence substitutions described in
`docs/licensing_audit.md` all happen **behind** these contracts.

---

## 3. The six modules

### M1 — Linguistic Processing

> raw text → **LinguisticEmbedding**

| Sub-block | Component | License | Reason |
|---|---|---|---|
| 1.1 Text normalization | WeTextProcessing (PyNini/OpenFst WFST) + BERT-mini INT8 | Apache-2.0 | Expansion must be deterministic and auditable; only homograph ambiguity delegated to a model. An FST mathematically cannot invent a numeric expansion. |
| 1.2 G2P | Misaki (English) + Epitran (multilingual) | MIT | Faster, bounded; avoids eSpeak-ng / phonemizer GPL copyleft. |
| 1.3 UMIM | XLM-RoBERTa-base (ONNX Runtime) | MIT | Shared multilingual space keeps speaker identity orthogonal to language across code-switch boundaries. |

**The Indian-language failure this module addresses:** Devanagari looks
phonetic and is not. `कमल` reads `ka-ma-la` symbol by symbol but is spoken
`kamal`. Schwa deletion is systematic, phonologically conditioned, and
written nowhere in the script. Marathi is **not** Hindi here — Marathi
retains schwas in environments where Hindi deletes them, so a Hindi rule set
ported to Marathi is systematically wrong.

### M2 — Reference Engine (zero-shot cloning)

> 5–60 s reference audio → **AcousticConditioning** (S_identity[256] · W_att[T×D] · N_channel[128])

| Sub-block | Component | License | Reason |
|---|---|---|---|
| 2.1 Feature extraction | torchaudio MelSpectrogram + high-pass biquad + Silero VAD | MIT | Native PyTorch; VAD strips silence/noise. |
| 2.2 Speaker identity encoder | ECAPA-TDNN (SpeechBrain) + WavLM-Large FP16 | Apache / MIT | ECAPA answers *who*; WavLM restores *how* — cadence, breath, micro-formant detail a pooled embedding discards. |
| 2.3 Cross-attention conditioning | xFormers memory-efficient attention | BSD/Apache | Permits 60 s references without quadratic memory growth. |

**Phase-1 realization (TriFactorSSL dual-path):** three frozen SSL teachers
(XLS-R, WavLM, W2v-BERT 2.0) distil into a single 100–140 M streaming student
whose four output streams are explicitly disentangled — content, identity,
prosody, channel — by gradient-reversal + HSIC penalties. Used for uncached
references, enrollment, and asynchronous drift monitoring.

### M3 — Cross-Modal Fusion & Prosody

> LinguisticEmbedding + AcousticConditioning → **ProsodyAlignedTokens**

| Sub-block | Component | License | Reason |
|---|---|---|---|
| 3.1 Fusion mesh | StyleTTS2 mutual cross-attention | MIT | Paragraph-level fusion (vs FastSpeech2's sentence-local averaging). |
| 3.2 Global context | Llama-3-8B (4-bit AWQ) | Llama Community | Prosody is non-local — a question mark at the end of a sentence shapes the pitch contour of its first word. ~6 GB in 4-bit. |
| 3.3 Emotion injector | Emotion2Vec (FunASR) + gradient reversal | Apache-2.0 | Arousal/valence injected adversarially so reference cadence transfers without lexical leakage. |

### M4 — Security & Moderation Guardrails

> ProsodyAlignedTokens + consent claim → **SecureAlignedTokens** (fail-closed)

| Sub-block | Component | License | Reason |
|---|---|---|---|
| 4.1 Moderation | ShieldGemma-2B (INT8) / LlamaGuard-3-1B | Gemma / Llama | Screening precedes generation: refusal is free, unmade audio cannot leak. |
| 4.2 Consent gate | ECAPA-TDNN speaker-match + signed token | Apache-2.0 | Only authorized voices synthesize. Fail-closed. |
| 4.3 Provenance | AudioSeal localized watermark | MIT (code + weights) | Inaudible cryptographic signature; embedded after decode, before compression. |

**The load-bearing invariant:** cryptographic tokens are never injected into
semantic or acoustic tensors — they live only in the control plane.

### M5 — Acoustic Generation

> SecureAlignedTokens + AcousticConditioning → **MelSpectrogram** ([T, 80] log-mag)

| Sub-block | Component | License | Reason |
|---|---|---|---|
| 5.1 Latent predictor | Matcha-TTS OT-CFM | MIT | Optimal-transport CFM: constant-velocity target, stable objective, large solver steps. |
| 5.2 Acoustic transformer | F5-TTS (DiT + flow matching) | MIT code / ⚠ CC-BY-NC weights | 4–8 evaluations vs 50–1000. Zero-shot cloning falls out of in-context infilling. |
| 5.3 Mel construction | native torch rescaling + 1D conv phase-correction head | — | Mel params are a versioned contract with M6. |

**Robotic-TTS insight:** speech is one-to-many. Plain regression converges to
the conditional mean of valid renderings, and the mean of all human speech is
a flat monotone. **More data does not fix this** — it makes the average
smoother, and therefore more robotic. The fix is explicit variance modelling.

### M6 — Neural Vocoder & Streaming

> MelSpectrogram → 24 kHz PCM (optionally 20 ms streamed packets)

| Sub-block | Component | License | Reason |
|---|---|---|---|
| 6.1 Vocoder core | Vocos (ConvNeXt + inverse STFT) | MIT | Predicts STFT magnitude + phase, single inverse STFT. ~100× faster than HiFi-GAN; superior HF phase; ~50 MB. |
| 6.2 Long-form stability | Chunked inference + 50 ms crossfade | — | Re-anchoring suppresses volume drift / fade-out over multi-minute output. |
| 6.3 Streaming | FastAPI WS + torch.cuda.Stream + PyAV | MIT | First audio emitted once ~300 ms mel is available; 20 ms packets; barge-in. |

---

## 4. The two-phase program

| | Phase 1 — Representation Core | Phase 2 — Speed & Footprint |
|---|---|---|
| **Goal** | Build the speech-understanding plane. Disentangle the factors of speech so the generator can treat them as independent levers. | Port the Phase-1 representation onto a lighter, faster mel-cascade that closes the latency budget on one 32 GB accelerator. |
| **M2** | TriFactorSSL dual-path (SSL ∥ Qwen3-TTS codec) | ECAPA-TDNN + WavLM-Large (FP16) |
| **M5** | Hierarchical semantic → acoustic (masked diffusion) | Flow-matching DiT (F5-TTS), 4–8 evaluations |
| **M6** | Causal codec decode | Vocos iSTFT spectral vocoding |
| **VRAM** | ~1.1–1.3 B params, ≤ 28 GiB | ~11.5 GB inference base, ≤ 32 GB |
| **Profiles** | interactive / studio | same checkpoint, lighter schedule |

**The three Phase-2 latency decisions** — (1) flow matching in place of
iterative diffusion, (2) Vocos spectral vocoding in place of time-domain
synthesis, (3) FlashAttention-2 + torch.compile — do not trade quality for
speed. Each removes computation the task does not require. That is why they
compose.

---

## 5. Memory budget (Phase 2, 32 GB target)

| Module | Component | VRAM |
|---|---|---|
| M1 | Text pipeline (FST + ONNX) | < 0.5 GB |
| M2 | Speaker encoder + WavLM-Large (FP16) | ~1.2 GB |
| M3 | Aligner (8B 4-bit AWQ) + fusion | ~6.0 GB |
| M4 | Safety classifier (INT8) | ~1.2 GB |
| M5 | Acoustic transformer (F5-TTS, BF16) | ~2.5 GB |
| M6 | Vocoder (iSTFT, FP32) | ~0.05 GB |
| **Inference base** | | **~11.5 GB** |
| Remaining | KV cache + ODE solver + stream buffers | ~20.5 GB |

The 20.5 GB headroom is deliberate: concurrent streams multiply cache and
buffer requirements while weights stay shared.

---

## 6. Inter-module tensor contracts

Each module emits one of these. Producer → Consumer.

| Producer → Consumer | Contract | Shape / type |
|---|---|---|
| M1 → M3 | `LinguisticEmbedding` | graphemes + phoneme lattice + lang spans + controls |
| M2 → M3, M5 | `AcousticConditioning` | `S_identity[256] · W_att[T×D] · N_channel[128]` |
| M3 → M4 | `ProsodyAlignedTokens` | `tokens[T] · dur[T] · F0[T] · E[T] · emo[d]` |
| M4 → M5 | `SecureAlignedTokens` | same as above, post-authorization (no crypto in tensors) |
| M5 → M6 | `MelSpectrogram` | `[T, 80]` float32 log-mag, 24 kHz frame rate |
| Phase 1 | `SemanticUnit[t]` | int16, codebook 16,384, ~12.5 Hz |
| Phase 1 | `TriFactorStreams` | `C_content[T×D_c] · S_identity[D_s] · P_prosody[T×D_p] · N_channel[D_n]` |

These contracts are encoded as Pydantic dataclasses in
`vajravoice/contracts.py` and asserted by the pipeline at every stage
boundary.

---

## 7. Deployment topology

A single self-hosted 32 GB GPU host. BF16/FP8 mixed runtime; FP8 qualified
only after BF16 quality locks. Throughput uses vLLM-Omni-style continuous
batching, paged KV, CUDA graphs, FlashAttention/FlashInfer, and asynchronous
codec decoding. Cached voice profiles bypass reference encoding on the
synthesis hot path. **No third-party data processor** — data residency is
the operator's own.

Concurrency: 1–4 interactive streams per GPU; weights are shared across
streams while KV cache scales.

---

## 8. Why MAI-Voice-2 is the reference

Every cell below is from each vendor's official pricing/specification page
(July 2026). MAI-Voice-2's quality claims are vendor-reported and flagged.

| Provider / Model | $ / 1M chars | Languages | Cloning | TTFA |
|---|---|---|---|---|
| **MAI-Voice-2 🏆** | **$22** | **15 + code-switch** | **Zero-shot, all langs** | **Real-time** |
| OpenAI tts-1 | $15 | ~6 | No | ~2000 ms HD |
| xAI Grok TTS | $15 | undisclosed | No | undisclosed |
| Azure Neural | $16 | 140+ | No (paid tier) | ~150–250 ms |
| Amazon Polly Neural | $16 | 60+ | No | ~150–250 ms |
| OpenAI tts-1-hd | $30 | ~6 | No | slow |
| Google Chirp 3 HD | $30 | 95+ | No | ~150–250 ms |
| Deepgram Aura-2 | $30 | 7 | No | sub-200 ms |
| Gradium | ~$36 | undisclosed | Yes (~10 s ref) | 155 ms P50 |
| Cartesia Sonic-3 | ~$50 | 40+ | Yes (3 s ref) | ~90 ms (fastest) |
| ElevenLabs v2/v3 | $100 | 32+ | Yes (leader) | ~375 ms |

**The verdict.** MAI-Voice-2 is the **Pareto-optimal point** of the
commercial TTS landscape: the only system in the sub-$25 / 1M-character band
that simultaneously delivers zero-shot voice cloning, 15 languages including
Hindi↔English code-switching, built-in consent guardrails (only licensed
voices synthesize), and Microsoft's reported human-parity naturalness (45.5%
preferred MAI-Voice-2 over a real human recording vs 44% for human; 72.1%
win-rate vs MAI-Voice-1).

**Honesty caveats.** (1) The 45.5% / 72.1% figures are Microsoft-reported;
there is no independent TTS Arena benchmark for MAI-Voice-2 yet (only
MAI-Voice-1 is on those leaderboards). (2) MAI-Voice-2 launched June 2026 —
it is a young model, and its latency has not been independently measured.
Neither caveat changes the architectural conclusion.

---

## 9. Honesty: what is and isn't claimed

- ✅ **Claimed:** the MAI-Voice-2 architecture is public, and VajraVoice
  rebuilds the same capability class from open-source components the operator
  owns end-to-end.
- ✅ **Claimed:** the decomposition into 6 modules and 18 blocks is original
  work, with verified per-component licences and a falsifiable SSL ablation.
- ❌ **NOT claimed:** that MAI-Voice-2 uses self-supervised learning
  internally. There is no public evidence either way; its use here is a
  performance-driven design choice.
- ❌ **NOT claimed:** that this is a deployed product. It is an architecture
  and integration portfolio — wiring, contracts, tests, and scaffolded
  component hooks; the heavy ML stack loads on a CUDA box.
- ❌ **NOT claimed:** that any model was trained from scratch. The
  contribution is assembly + integration behind fixed contracts, not
  pretraining.
