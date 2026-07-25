# Architectural Decision Records

The six load-bearing decisions, in priority order. Each is falsifiable.

---

## ADR-001 · Representation-first ordering

**Context.** Every other open TTS pipeline reaches for the cheapest generator
first and inherits an entangled representation it can never fully correct.
Neural-codec tokens and pooled speaker embeddings entangle phonetic content,
speaker identity, prosody, accent, and recording conditions. A faster
generator applied to an entangled representation yields fast, confidently
wrong, non-native speech.

**Decision.** Build the disentangled self-supervised speech-understanding
core **first** (Phase 1), and only then pursue raw speed (Phase 2).

**Consequence.** Longer time-to-first-audio than a generator-first build.
The thesis is **falsifiable**: see the mandatory SSL ablation below. If the
ablation does not show the student preserving the ensemble's gains, the
two-phase ordering is wrong and the architecture is revisited before any
speed work proceeds.

**The mandatory SSL ablation.** Six configurations:
1. Codec conditioning without SSL
2. XLS-R only
3. WavLM only
4. XLS-R + WavLM teachers
5. Full teacher ensemble
6. Distilled TriFactorSSL student

The system must demonstrate (a) the student preserves the ensemble's gains,
and (b) a Phase-2-speed generator on the disentangled representation beats
the same generator on an entangled one on native-listener preference.

**Status.** Accepted.

---

## ADR-002 · Flow matching over iterative diffusion

**Context.** Iterative diffusion models a curved, stochastic trajectory from
noise to data, requiring 50–1000 small solver steps. The step count is the
single largest term in the generation budget.

**Decision.** Use **conditional flow matching** (F5-TTS / Matcha-TTS OT-CFM).
Flow matching regresses a velocity field along a near-straight transport
path, so the ODE can be integrated in 4–8 large steps. The objective
differs from diffusion — the reported results indicate parity, not a
trade-off.

**Consequence.** F5-TTS released weights are CC-BY-NC (non-commercial). A
commercial build needs a permissively-licensed replacement checkpoint
(OpenF5-TTS, Apache) or a retrain on owned data. The architecture is
unaffected; only the checkpoint is.

**Status.** Accepted. Licensing guard enforces the checkpoint replacement.

---

## ADR-003 · Vocos iSTFT over HiFi-GAN

**Context.** Adversarial vocoders in the HiFi-GAN lineage reconstruct the
waveform in the time domain, upsampling through transposed convolutions up to
24,000 samples/sec. This is the dominant cost of the vocoder stage.

**Decision.** Use **Vocos** — predict STFT magnitude and phase, perform a
single inverse STFT. The waveform is produced in one pass instead of
progressively upsampled.

**Consequence.** ~90% vocoder latency cut at comparable perceptual quality,
with superior high-frequency phase reconstruction. Footprint ~50 MB (the
smallest module in the pipeline).

**Status.** Accepted.

---

## ADR-004 · ECAPA-TDNN + WavLM over a pooled embedding

**Context.** A pooled speaker embedding reproduces timbre without behaviour.
Pooling destroys exactly the time-varying detail (cadence, micro-formants,
breath) that makes a voice sound like a specific person rather than a
generic speaker of that timbre.

**Decision.** Pair **ECAPA-TDNN** (timbre anchor, 256-dim) with a frozen
**FP16 WavLM-Large** feature extractor (time-varying detail). ECAPA answers
*who*; WavLM restores *how*. Phase-1 uses the TriFactorSSL dual-path
realization for uncached references.

**Consequence.** ~1.2 GB VRAM. VoxCeleb provenance requires verification
before commercial deployment — the model licence does not launder the
dataset's research-only terms.

**Status.** Accepted. The VoxCeleb EER gap on Indian speakers is a known,
measurable issue; mitigation is retraining on consented data.

---

## ADR-005 · Fail-closed pre-generation guardrail

**Context.** Module 4 must answer two independent questions: is the voice
authorized, and is the content permitted? These must not be collapsed — an
authorized voice can still be directed to produce impermissible content.

**Decision.** The gate sits **before** generation. Generation is the dominant
cost, so refusal before it is nearly free. AND audio that is never generated
cannot subsequently be disclosed through any buffer, log, or cache. When the
cost argument and the security argument agree, you have found the right
design.

The gate **fails closed**: an unreachable consent service denies the request.
An availability failure must never become a security failure.

**Consequence.** Every decision is written to an idempotent, append-only
audit log. Cryptographic tokens live only in the control plane — they are
never injected into semantic or acoustic tensors.

**Status.** Accepted. This single design decision is also why DPDP / voice-
personal-data compliance is structurally built in, not bolted on.

---

## ADR-006 · Permissively-licensed stack

**Context.** The pattern across open-source TTS is that the code licence is
a decoy — the weights are where you get killed. F5-TTS, XTTS-v2, Fish
Speech, MaskGCT, Spark-TTS all have permissive code and non-commercial
weights. eSpeak-ng and phonemizer are GPL.

**Decision.** MIT/Apache/BSD on shipped weights only. F5-TTS checkpoint
replaced by design (OpenF5-TTS or retrain). eSpeak-ng replaced by design
(permissive Indic articulatory ruleset). No copyleft in the front end.

**Consequence.** The licensing guard (`vajravoice/utils/licensing.py`) raises
`ShipWarning` at load time if a non-ship component is selected in a
`commercial: true` config. The audit table in `docs/licensing_audit.md` is
the machine-checkable form of this ADR.

**Status.** Accepted. This is the difference between a product and a lawsuit.

---

## The five Indic-specific blocks

The structural insight that makes the architecture economically tractable:
**only 5 of the 18 sub-blocks are genuinely language-specific.** Vocoders
don't care what language you speak — physics is physics. Flow-matching
generators clone in-context — they generalize for free. Cross-attention is a
mechanism, not a language.

| Block | Indic work? |
|---|---|
| 1.1 Text Normalization | 🔴 YES — lakh/crore, ₹, DD/MM |
| 1.2 G2P | 🔴 YES — schwa deletion, conjuncts |
| 1.3 Unified space / LID | 🔴 YES — per-token, Romanized input |
| 2.2 Speaker encoder | 🟡 partly — VoxCeleb bias |
| 4.2 Safety classifier | 🔴 YES — code-switched vishing |
| 5.1 Duration model | 🔴 YES — **syllable timing** |
| *Everything else* | 🟢 inherited free |

Knowing this is the difference between a fundable plan and a fantasy.
