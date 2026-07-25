# Licensing Audit — Commercial Safety by Component

> Deployment is governed by the licence attached to each **checkpoint**, which
> in several cases differs from the licence on the repository. **The weights
> are the binding term.**

This audit is encoded in `vajravoice/utils/licensing.py` and enforced at load
time: selecting a non-ship component in a `commercial: true` config raises
`ShipWarning` immediately.

---

## ✅ CLEAN — ship freely (MIT / Apache / BSD on both code AND weights)

| Component | Licence | Module |
|---|---|---|
| Wav2Vec2-XLS-R | Apache-2.0 | M2 (Phase-1 teacher) |
| WavLM (implementation) | MIT | M2 (Phase-1 teacher + Phase-2 features) |
| W2v-BERT 2.0 | MIT | M2 (Phase-1 teacher) |
| AudioSeal | MIT (code + weights) | M4 (watermark) |
| Vocos | MIT | M6 (vocoder) |
| Matcha-TTS | MIT | M5 (latent predictor) |
| ECAPA-TDNN via SpeechBrain | Apache-2.0 | M2 / M4 (speaker encoder, consent gate) |
| WeTextProcessing | Apache-2.0 | M1 (normalization) |
| Epitran | MIT | M1 (G2P) |
| Misaki | MIT | M1 (G2P) |
| XLM-RoBERTa | MIT | M1 (UMIM) |
| Silero VAD | MIT | M2 (reference cleanup) |
| Emotion2Vec (FunASR) | Apache-2.0 | M3 (emotion injection) |
| StyleTTS 2 | MIT | M3 (fusion) — pretrained use carries a disclose-synthesis + voice-consent term |
| OpenF5-TTS | Apache-2.0 | M5 (F5-TTS weights replacement) |
| Mamba-2 | Apache-2.0 | M5 (research SSM option) |
| BigVGAN | MIT | M6 (alternative vocoder) |

**Note:** The load-bearing Phase-2 latency decisions — flow matching
(Matcha-TTS) and spectral vocoding (Vocos) — both rest on components in this
group.

---

## ⚠️ CHECKPOINT REPLACEMENT — code permissive, released weights non-commercial / community

These selections require a permissively-licensed replacement checkpoint or
retraining on owned data before commercial deployment. **The architecture is
unaffected; only the checkpoint is.**

| Component | Code | Weights | Action |
|---|---|---|---|
| **F5-TTS** | MIT | CC-BY-NC (training-corpus provenance) | Replace with OpenF5-TTS (Apache) or retrain on owned data. The flow-matching architecture is identical — only the weights change. |
| **Llama-3-8B** (M3 prosody aligner) | Llama-3 Community | Llama-3 Community | Read the community licence against your deployment. Permits commercial use subject to conditions (e.g. monthly active user thresholds). |
| **ShieldGemma-2B** / **LlamaGuard-3-1B** (M4 moderation) | Gemma / Llama Community | Gemma / Llama Community | Same as above — provider community licences permitting commercial use subject to conditions. |
| **Qwen3-TTS** (Phase-1 codec donor) | Apache-2.0 | Qwen Community | Read community licence against deployment. |
| **DualCodec** (Phase-1 semantic codec) | MIT | research (Emilia-data provenance) | Do not ship current pretrained checkpoint unless weight + Emilia-data audit passes. MIT code/arch usable as implementation reference. |
| **WavLM-SV** (speaker-verification variant) | MIT | CC-BY-SA | Use the base WavLM-Large instead, which is fully MIT. |

---

## 🚫 REPLACED BY DESIGN — copyleft, replaced in the build

These selections never appear in a shipped binary because they are replaced
by permissively-licensed alternatives at the architecture level.

| Component | Licence | Replacement | Why |
|---|---|---|---|
| **eSpeak-ng** | GPL-3.0 | Permissively-licensed Indic articulatory ruleset | A GPL runtime in a closed product opens the entire product to copyleft. The front end ships without it. |
| **phonemizer** | GPL-3.0 | Misaki + Epitran (MIT) | `phonemizer` is a GPL wrapper around eSpeak-ng — same copyleft trap. Replaced by Misaki (English) + Epitran (multilingual) + an internal permissive Indic ruleset. |

---

## VoxCeleb provenance — a separate risk

Every production speaker encoder (ECAPA-TDNN, Resemblyzer, TitaNet) is
trained on **VoxCeleb**, which is scraped from YouTube and is **research-only
by its dataset licence**. A permissive *model* licence does not launder the
*dataset* terms.

- The encoder learned axes that separate Western English speakers.
- Indian voices sit in a region of that space it never learned to resolve —
  they cluster, they are less discriminable.
- The encoder has high resolution for a Californian and low resolution for a
  Kannadiga. That's an EER gap, and it's a number we can measure.
- **Mitigation:** retrain the encoder on consented Indic speaker data. This
  is a cost line, not a surprise.

---

## Pattern: the code licence is a decoy

Across the open-source TTS landscape, the code licence is frequently a decoy:

| Project | Code licence | Weights licence | Shippable? |
|---|---|---|---|
| F5-TTS | MIT | CC-BY-NC | ❌ |
| XTTS-v2 (Coqui) | MPL-2.0 | CPML (non-commercial) | ❌ (Coqui is shut down; nobody left to sell a commercial licence) |
| Fish Speech | Apache-2.0 | research (S2) | ❌ |
| MaskGCT | MIT | CC-BY-NC | ❌ |
| Spark-TTS | Apache-2.0 | CC-BY-NC | ❌ |
| Echo-TTS | Apache | CC-BY-NC-SA | ❌ |
| **IndexTTS-2** | Apache-2.0 | Apache-2.0 | ✅ |
| **CosyVoice 2** | Apache-2.0 | Apache-2.0 | ✅ |
| **Chatterbox** | MIT | MIT | ✅ |
| **Kokoro-82M** | Apache-2.0 | Apache-2.0 | ✅ |

The licensing guard in this repo (`vajravoice/utils/licensing.py`) is the
machine-checkable form of this audit.

---

## Verify before commercial release

1. **Code licence vs weights licence** — they differ often. The weights
   licence is the binding term.
2. **Training-data provenance** — model licence ≠ dataset licence. VoxCeleb
   is the canonical trap.
3. **Provider community licences** (Llama, Gemma) — read against the actual
   deployment; they permit commercial use under conditions (usage thresholds,
   acceptable-use policies).
4. **Watermarking obligations** — StyleTTS2 pretrained use carries a
   disclose-synthesis + voice-consent term; AudioSeal is MIT on both.
