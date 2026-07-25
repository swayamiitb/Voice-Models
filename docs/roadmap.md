# Roadmap

The two-phase program, sequenced so every phase ends in something **you can
hear** — not a status update.

---

## Phase 1 — Prove the diagnosis (~4–6 weeks)

Stand up the commercial-safe baseline and validate the architectural diagnosis.

- ✅ Commercial-safe baseline: CosyVoice 2 or Chatterbox + AudioSeal watermark
  + ECAPA consent gate
- ✅ Indic TN grammar (lakh/crore, ₹, DD/MM) — fastest win in the project
- ✅ Ship an A/B demo: vendor Hinglish vs ours. The whole phase is judged on
  one sound file.
- ✅ Measure the VoxCeleb EER gap on Indian speakers. Publish the number
  internally.

**Exit criteria:** you can hear the difference, and we can show you the number.

---

## Phase 2 — Fix the five blocks (~2–3 months)

The five Indic-specific blocks, in order of leverage:

1. **Duration model** (M5.1) — fine-tune on syllable-timed speech. The single
   highest-leverage change for perceived nativeness.
2. **G2P + schwa deletion** (M1.2) — and the structural gift: the Indo-Aryan
   family shares a Sanskritic phoneme inventory, so build once and transfer
   across Hindi, Marathi, Gujarati, Bengali, Punjabi, Odia.
3. **Code-switching, architecturally** (M1.3) — per-token language ID,
   Romanized-Latin detection, intra-word mixing (`WhatsApp करोगे`).
4. **Speaker encoder bias** (M2.2) — fine-tune on Indic speaker data to
   close the VoxCeleb EER gap.
5. **Code-switched safety** (M4.1) — a Hinglish scam script sails straight
   through an English toxicity model. This block is simultaneously the
   responsible thing to build AND a moat.

**Exit criteria:** native speakers prefer ours in a blind test. Nothing softer.

---

## Phase 3 — Productionise (~2–3 months)

- ✅ Streaming under 300 ms TTFA on **Indian network conditions** — not on fibre
- ✅ Long-form stabilization (audiobooks, lectures, IVR trees)
- ✅ Edge caching, on-prem deployment, data residency
- ✅ Language expansion — each one is cheaper than the last

---

## Phase 1 representation core (research track)

In parallel to the engineering phases above, Phase 1 of the *architecture*
builds the disentangled self-supervised speech-understanding plane:

1. **Frozen-teacher feature extraction** — precompute XLS-R, WavLM, W2v-BERT
   layer features for the training corpus.
2. **TriFactorSSL distillation** — train the 100–140 M student + factorization
   heads (gradient reversal + HSIC).
3. **Semantic codebook + alignment** — freeze the SSL student, train attention
   pooling + quantizer + text-semantic alignment.
4. **Generator adaptation** — initialize the shared transformer from
   Qwen3-TTS; freeze 75–80%; train Indic experts + Prosody Director.
5. **Speed distillation** — distil 8 semantic-refinement steps into 4 then 2;
   apply consistency distillation to the studio refiner; introduce FP8 only
   after BF16 quality locks.

Expected training scale: adapter / student / distillation — short 4–8 GPU
jobs, not a new multi-million-hour foundation pretraining run.

---

## Why this compounds

1. **The licence map.** Verified per-component against actual LICENSE files.
   Boring, unglamorous, and the difference between a product and a lawsuit.
2. **The phoneme space compounds.** Every Indic language we add makes the
   next one cheaper. By language six we're faster than anyone starting at
   language one.
3. **Data flywheel.** Indic reference audio, consented, in real field
   conditions — noisy, phone-recorded, code-switched. Not studio English.
   That corpus doesn't exist, and it gets more valuable every month.
4. **The diagnosis itself.** Nobody else is talking about syllable-timing.
   When they finally notice, they'll have to rebuild their duration model.
