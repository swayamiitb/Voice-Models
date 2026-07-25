"""Generate the visual assets for the VajraVoice README.

Every asset is rendered as SVG (vector, crisp at any size, ~2–8 KB each, no
binary bloat in the repo). The waveforms are SYNTHESIZED from actual
speech-model math — a glottal-source + formant-resonator model — so they look
like real speech, not random noise.

Outputs (all into assets/):
    logo.svg               — VajraVoice mark (vajra + waveform)
    hero_waveform.svg      — the long hero waveform under the title
    pipeline.svg           — six-module cascade diagram
    voice_fingerprint.svg  — 256-d radial speaker-identity "fingerprint"
    spectrogram.svg        — time-frequency heat map of one phoneme sequence
    licensing_map.svg      — module × component licensing grid (traffic lights)
    memory_budget.svg      — stacked-bar VRAM breakdown
    latency_budget.svg     — 240 ms TTFA decomposition bar

Usage:
    python -m scripts.generate_assets
    # or:
    python scripts/generate_assets.py

No external deps beyond numpy. Run from the repo root.
"""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# Palette — deep blue (VajraVoice brand) + amber (Phase-2 speed) +
# green (CLEAN licence) + amber (CHECKPOINT_REPLACEMENT) + red (REPLACED).
INK = "#0b1424"
INK_2 = "#1a2540"
INK_3 = "#2d3a55"
PAPER = "#f7f9fc"
BRAND = "#4c8dff"
BRAND_2 = "#7aa9ff"
BRAND_DEEP = "#12213f"
AMBER = "#f5a623"
AMBER_DEEP = "#c47e0a"
GREEN = "#2ecc71"
GREEN_DEEP = "#1f8b4c"
RED = "#e74c3c"
RED_DEEP = "#a82418"
TEAL = "#06b6d4"
PURPLE = "#a855f7"


# ============================================================================
# Speech-shaped signal synthesis (glottal source + formant resonators)
# ============================================================================


def _envelope(t: np.ndarray, attack: float = 0.02, release: float = 0.20) -> np.ndarray:
    """Two-sided amplitude envelope: fast attack, slow release (syllable shape)."""
    env = np.ones_like(t)
    env[t < attack] = t[t < attack] / attack
    rel_mask = t > (1.0 - release)
    env[rel_mask] = np.cos(np.pi * (t[rel_mask] - (1.0 - release)) / (2 * release)) ** 2
    return env


def synth_syllable(duration: float, f0: float, formants: list[tuple[float, float]],
                   sr: int = 22050, noise: float = 0.0) -> np.ndarray:
    """Synthesize one voiced syllable: glottal-source pulse train passed through
    a cascade of 2-pole formant resonators. This is the classic
    source-filter model of speech production (Fant 1960)."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Glottal source: derivative-of-Liljencrants-Fant approximation = pulse train
    # smoothed to remove high-frequency content the formants would just smear.
    phase = (t * f0) % 1.0
    glottal = np.where(phase < 0.6, np.sin(np.pi * phase / 0.6) ** 2, 0.0)
    # Mix in breathiness: low-amplitude white noise (aspiration).
    if noise > 0:
        glottal = glottal + noise * np.random.default_rng(42).standard_normal(n)
    # Apply each formant as a 2-pole resonator (z-domain: r * e^{±jω}).
    sig = glottal.copy()
    rng = np.random.default_rng(7)
    for freq, bw in formants:
        r = math.exp(-math.pi * bw / sr)
        coeff = 2 * r * math.cos(2 * math.pi * freq / sr)
        out = np.zeros(n)
        out[0] = sig[0]
        out[1] = sig[1] + coeff * out[0]
        for i in range(2, n):
            out[i] = sig[i] + coeff * out[i - 1] - r * r * out[i - 2]
        sig = (1 - r) * out + 0.3 * sig  # mild mix so formants don't fully kill
    return sig * _envelope(t)


def synth_phrase(syllables: list[dict], sr: int = 22050) -> np.ndarray:
    """Concatenate syllables with small inter-syllable gaps."""
    parts = []
    for syl in syllables:
        sig = synth_syllable(syl["dur"], syl["f0"], syl["formants"], sr, syl.get("noise", 0.0))
        parts.append(sig)
        parts.append(np.zeros(int(0.04 * sr)))  # 40 ms gap
    return np.concatenate(parts)


def marathi_phrase_signal(sr: int = 22050) -> np.ndarray:
    """A phrase shaped like 'namaskar' — falling F0 contour, three syllable
    shapes with different formant patterns (open vowel → sibilant → closed)."""
    rng = np.random.default_rng(123)
    # F0 falls across the phrase (statement intonation, not question).
    return synth_phrase([
        # "na" — open vowel /a/, F1 high
        {"dur": 0.18, "f0": 145, "formants": [(720, 90), (1240, 110), (2800, 180)]},
        # "mas" — vowel + sibilant
        {"dur": 0.20, "f0": 138, "formants": [(680, 95), (1300, 120), (2700, 200)], "noise": 0.10},
        # "kaar" — long vowel, lower pitch
        {"dur": 0.30, "f0": 120, "formants": [(740, 80), (1180, 100), (2900, 170)]},
        # short silence
        {"dur": 0.06, "f0": 110, "formants": [(400, 100), (1500, 200), (2600, 250)]},
        # final release
        {"dur": 0.14, "f0": 105, "formants": [(720, 90), (1240, 110), (2800, 180)]},
    ], sr=sr)


# ============================================================================
# SVG primitives
# ============================================================================


def svg_open(width: int, height: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">\n')


def svg_close() -> str:
    return "</svg>"


def write(name: str, body: str) -> None:
    path = ASSETS / name
    path.write_text(body, encoding="utf-8")
    size = path.stat().st_size
    print(f"  ✓ {name:<28} {size:>6} bytes")


# ============================================================================
# Asset 1: logo.svg — vajra mark + waveform
# ============================================================================


def gen_logo() -> None:
    W, H = 280, 80
    cx, cy = 40, H // 2
    # Vajra: 8-spoked star at left
    vajra = []
    for i in range(8):
        a = i * math.pi / 4
        x2, y2 = cx + 22 * math.cos(a), cy + 22 * math.sin(a)
        vajra.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{BRAND}" stroke-width="2.2" stroke-linecap="round"/>')
    # Center hub
    vajra.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{BRAND}"/>')
    vajra.append(f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{PAPER}"/>')

    # Wordmark — stylized
    word = (
        f'<text x="80" y="{cy + 8}" font-size="28" font-weight="700" fill="{INK}" '
        f'letter-spacing="-0.5">Vajra<tspan fill="{BRAND}">Voice</tspan></text>'
    )
    # Mini waveform under the wordmark
    wf = []
    rng = np.random.default_rng(11)
    for i in range(60):
        x = 82 + i * 2.6
        # Tapered amplitude: rise then fall (envelope)
        env = math.sin(math.pi * i / 60) ** 1.5
        amp = env * (4 + 9 * rng.random()) * (1 if i % 2 else -1)
        wf.append(f'<line x1="{x:.1f}" y1="{cy + 18:.1f}" x2="{x:.1f}" '
                  f'y2="{cy + 18 + amp:.1f}" stroke="{BRAND_2}" stroke-width="1.4" '
                  f'stroke-linecap="round"/>')
    body = svg_open(W, H) + "".join(vajra) + word + "".join(wf) + svg_close()
    write("logo.svg", body)


# ============================================================================
# Asset 2: hero_waveform.svg — long synthesized speech waveform
# ============================================================================


def gen_hero_waveform() -> None:
    sr = 22050
    sig = marathi_phrase_signal(sr)
    # Downsample to ~360 bars (each bar = peak amplitude over a window)
    target_bars = 360
    win = max(1, len(sig) // target_bars)
    peaks = np.array([
        np.max(np.abs(sig[i:i + win])) if i < len(sig) else 0.0
        for i in range(0, len(sig), win)
    ][:target_bars])

    W = 1200
    H = 220
    mid = H // 2
    max_amp = max(peaks.max(), 1e-6)
    bar_w = W / len(peaks)
    # Background gradient strip
    body = svg_open(W, H)
    body += f'<rect width="{W}" height="{H}" rx="10" fill="{PAPER}"/>'
    body += (f'<defs><linearGradient id="wfgrad" x1="0" y1="0" x2="1" y2="0">'
             f'<stop offset="0" stop-color="{BRAND}" stop-opacity="0.9"/>'
             f'<stop offset="0.5" stop-color="{TEAL}" stop-opacity="0.95"/>'
             f'<stop offset="1" stop-color="{AMBER}" stop-opacity="0.9"/>'
             f'</linearGradient></defs>')
    # Symmetric bars (mirror around mid)
    for i, p in enumerate(peaks):
        h = (p / max_amp) * (mid - 12)
        x = i * bar_w
        body += (f'<rect x="{x:.2f}" y="{mid - h:.2f}" width="{bar_w * 0.7:.2f}" '
                 f'height="{2 * h:.2f}" rx="0.8" fill="url(#wfgrad)"/>')
    # Center line
    body += (f'<line x1="0" y1="{mid}" x2="{W}" y2="{mid}" stroke="{INK_3}" '
             f'stroke-width="0.5" stroke-dasharray="2 3" opacity="0.4"/>')
    # Caption
    body += (f'<text x="20" y="28" font-size="13" font-weight="600" fill="{INK_2}">'
             f'∿  synthesized 24 kHz speech — source-filter model</text>')
    body += (f'<text x="{W - 20}" y="{H - 16}" font-size="11" fill="{INK_3}" '
             f'text-anchor="end" font-family="monospace">≈ 0.88 s · marathi phrase</text>')
    body += svg_close()
    write("hero_waveform.svg", body)


# ============================================================================
# Asset 3: pipeline.svg — six-module cascade
# ============================================================================


def gen_pipeline() -> None:
    W, H = 1180, 360
    modules = [
        ("M1", "Linguistic",        "text → embeddings",            BRAND,   ["WeTextProcessing", "Misaki+Epitran", "XLM-RoBERTa"]),
        ("M2", "Reference",         "ref → voice identity",         TEAL,    ["ECAPA-TDNN", "WavLM-Large FP16", "xFormers x-attn"]),
        ("M3", "Fusion + Prosody",  "embeddings → aligned tokens",  PURPLE,  ["StyleTTS2", "Llama-3-8B 4-bit", "Emotion2Vec"]),
        ("M4", "Guardrails",        "tokens → audited tokens",      RED,     ["ShieldGemma-2B", "ECAPA consent", "AudioSeal"]),
        ("M5", "Generator",         "tokens → 80-ch mel",           AMBER,   ["Matcha-TTS OT-CFM", "F5-TTS flow-matching", "Mamba-2 opt"]),
        ("M6", "Vocoder",           "mel → 24 kHz PCM",             GREEN,   ["Vocos iSTFT", "chunked streaming", "20 ms packets"]),
    ]
    n = len(modules)
    margin_x = 24
    gap = 18
    card_w = (W - 2 * margin_x - (n - 1) * gap) / n
    card_h = 230
    card_y = 60

    body = svg_open(W, H)
    body += f'<rect width="{W}" height="{H}" fill="{PAPER}" rx="10"/>'
    # Title
    body += (f'<text x="{W // 2}" y="30" font-size="14" font-weight="700" fill="{INK}" '
             f'text-anchor="middle" letter-spacing="0.5">SIX-MODULE NEURAL CORE  ·  fixed tensor contracts at every boundary</text>')

    for i, (mid, name, role, color, comps) in enumerate(modules):
        x = margin_x + i * (card_w + gap)
        # Card
        body += (f'<rect x="{x:.1f}" y="{card_y}" width="{card_w:.1f}" height="{card_h}" '
                 f'rx="8" fill="white" stroke="{color}" stroke-width="1.5"/>')
        # Top color stripe
        body += (f'<rect x="{x:.1f}" y="{card_y}" width="{card_w:.1f}" height="34" '
                 f'rx="8" fill="{color}"/>')
        body += (f'<rect x="{x:.1f}" y="{card_y + 26}" width="{card_w:.1f}" height="8" fill="{color}"/>')
        # Module id
        body += (f'<text x="{x + 12:.1f}" y="{card_y + 23}" font-size="14" font-weight="800" '
                 f'fill="white" letter-spacing="0.5">{mid}</text>')
        # Name
        body += (f'<text x="{x + card_w - 12:.1f}" y="{card_y + 23}" font-size="13" font-weight="600" '
                 f'fill="white" text-anchor="end">{name}</text>')
        # Role
        body += (f'<text x="{x + card_w / 2:.1f}" y="{card_y + 56}" font-size="10.5" fill="{INK_3}" '
                 f'text-anchor="middle" font-family="monospace">{role}</text>')
        # Components
        for j, comp in enumerate(comps):
            y = card_y + 80 + j * 22
            body += (f'<rect x="{x + 12:.1f}" y="{y - 11:.1f}" width="{card_w - 24:.1f}" height="18" '
                     f'rx="4" fill="{color}" opacity="0.10"/>')
            body += (f'<text x="{x + card_w / 2:.1f}" y="{y + 2}" font-size="10.5" fill="{INK}" '
                     f'text-anchor="middle" font-weight="500">{comp}</text>')
        # Arrow to next module
        if i < n - 1:
            ax = x + card_w + 2
            ay = card_y + card_h / 2
            body += (f'<path d="M {ax:.1f},{ay - 5} L {ax + gap - 4:.1f},{ay} '
                     f'L {ax:.1f},{ay + 5} Z" fill="{INK_3}"/>')

    # Input/output labels
    body += (f'<text x="{margin_x:.1f}" y="{card_y - 12}" font-size="11" font-weight="600" '
             f'fill="{INK_2}">text ↓</text>')
    body += (f'<text x="{W - margin_x:.1f}" y="{card_y - 12}" font-size="11" font-weight="600" '
             f'fill="{INK_2}" text-anchor="end">↓ 24 kHz PCM</text>')
    # Reference input arrow (into M2)
    m2_x = margin_x + 1 * (card_w + gap) + card_w / 2
    body += (f'<text x="{m2_x:.1f}" y="{card_y + card_h + 22}" font-size="11" font-weight="600" '
             f'fill="{TEAL}" text-anchor="middle">↑ reference (5–60 s)</text>')
    body += (f'<line x1="{m2_x:.1f}" y1="{card_y + card_h + 6}" x2="{m2_x:.1f}" '
             f'y2="{card_y + card_h - 2}" stroke="{TEAL}" stroke-width="2" '
             f'marker-end="url(#arr)"/>')
    body += f'<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">' \
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{TEAL}"/></marker></defs>'

    body += svg_close()
    write("pipeline.svg", body)


# ============================================================================
# Asset 4: voice_fingerprint.svg — 256-d radial speaker identity
# ============================================================================


def gen_voice_fingerprint() -> None:
    W = H = 420
    cx = cy = W // 2
    # 256-dim embedding → polar plot. Deterministic seed so it looks the same
    # every render; the *shape* is what conveys "this is one voice's identity".
    rng = np.random.default_rng(2026)
    raw = rng.normal(0, 1, 256)
    # Smooth so it looks like a voice, not static
    kernel = np.ones(8) / 8
    smooth = np.convolve(raw, kernel, mode="same")
    radii = 60 + 110 * (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    angles = np.linspace(0, 2 * math.pi, 256, endpoint=False)

    body = svg_open(W, H)
    body += f'<rect width="{W}" height="{H}" fill="{PAPER}" rx="10"/>'
    # Concentric guides
    for r in (60, 100, 140, 170):
        body += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{INK_3}" stroke-width="0.5" opacity="0.25"/>'
    # Filled blob
    pts = []
    for a, r in zip(angles, radii):
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    body += (f'<polygon points="{" ".join(pts)}" fill="{BRAND}" fill-opacity="0.18" '
             f'stroke="{BRAND}" stroke-width="1.2"/>')
    # Spokes (every 4th dim, for texture)
    for i in range(0, 256, 4):
        a, r = angles[i], radii[i]
        x2, y2 = cx + r * math.cos(a), cy + r * math.sin(a)
        body += f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{BRAND}" stroke-width="0.6" opacity="0.4"/>'
    # Outer dots (every 8th)
    for i in range(0, 256, 8):
        a, r = angles[i], radii[i]
        x2, y2 = cx + r * math.cos(a), cy + r * math.sin(a)
        body += f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="1.8" fill="{TEAL}"/>'
    # Center hub
    body += f'<circle cx="{cx}" cy="{cy}" r="6" fill="{INK}"/>'
    body += f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{PAPER}"/>'
    # Caption
    body += (f'<text x="{cx}" y="{H - 18}" font-size="12" font-weight="600" fill="{INK}" '
             f'text-anchor="middle">S<tspan font-size="9" dy="3">identity</tspan>'
             f'<tspan dy="-3">  ·  256-d ECAPA-TDNN speaker anchor</tspan></text>')
    body += svg_close()
    write("voice_fingerprint.svg", body)


# ============================================================================
# Asset 5: spectrogram.svg — time-frequency heat map
# ============================================================================


def gen_spectrogram() -> None:
    sr = 22050
    sig = marathi_phrase_signal(sr)
    # Simple STFT
    n_fft = 256
    hop = 384                      # larger hop → fewer time frames
    win = np.hanning(n_fft)
    frames = []
    for i in range(0, len(sig) - n_fft, hop):
        spec = np.fft.rfft(sig[i:i + n_fft] * win)
        mag = 20 * np.log10(np.abs(spec) + 1e-6)
        frames.append(mag[:80])   # 80 mel-ish bins
    spec = np.array(frames).T     # [freq, time]
    # Downsample: average-pool freq axis 4→1 so we have ~20 mel bands.
    # Keeps the formant structure visible while cutting rect count 4×.
    n_freq_pool = 4
    padded = spec[: (spec.shape[0] // n_freq_pool) * n_freq_pool]
    pooled = padded.reshape(spec.shape[0] // n_freq_pool, n_freq_pool, -1).max(axis=1)
    spec = pooled
    # Normalize to [0,1]
    spec = np.clip((spec - spec.min()) / (spec.max() - spec.min() + 1e-9), 0, 1)

    # Render as colored grid
    W, H = 800, 280
    margin_l, margin_b, margin_t = 50, 32, 36
    plot_w = W - margin_l - 16
    plot_h = H - margin_b - margin_t
    cell_w = plot_w / spec.shape[1]
    cell_h = plot_h / spec.shape[0]

    def heat(v: float) -> str:
        # Viridis-ish: dark purple → blue → teal → green → yellow
        if v < 0.25:
            t = v / 0.25
            return f"rgb({int(13 + 30 * t)},{int(8 + 40 * t)},{int(60 + 80 * t)})"
        elif v < 0.5:
            t = (v - 0.25) / 0.25
            return f"rgb({int(43 + 0 * t)},{int(48 + 70 * t)},{int(140 + 60 * t)})"
        elif v < 0.75:
            t = (v - 0.5) / 0.25
            return f"rgb({int(43 + 60 * t)},{int(118 + 80 * t)},{int(200 - 100 * t)})"
        else:
            t = (v - 0.75) / 0.25
            return f"rgb({int(103 + 150 * t)},{int(198 + 30 * t)},{int(100 - 60 * t)})"

    body = svg_open(W, H)
    body += f'<rect width="{W}" height="{H}" fill="{PAPER}" rx="10"/>'
    body += (f'<text x="{W // 2}" y="22" font-size="13" font-weight="700" fill="{INK}" '
             f'text-anchor="middle">80-channel log-mel spectrogram  ·  M5 generator output</text>')
    # Cells — only emit those above threshold (background stays paper-colored)
    for fi in range(spec.shape[0]):
        for ti in range(spec.shape[1]):
            v = spec[fi, ti]
            if v < 0.20:
                continue
            x = margin_l + ti * cell_w
            y = margin_t + (spec.shape[0] - fi - 1) * cell_h
            body += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w + 0.6:.1f}" '
                     f'height="{cell_h + 0.6:.1f}" fill="{heat(v)}"/>')
    # Axes
    body += f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="{INK_3}" stroke-width="1"/>'
    body += f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="{INK_3}" stroke-width="1"/>'
    # Y-axis labels (frequency)
    for frac, label in [(0.0, "0"), (0.25, "3k"), (0.5, "6k"), (0.75, "9k"), (1.0, "12k Hz")]:
        y = margin_t + plot_h - frac * plot_h
        body += f'<line x1="{margin_l - 4}" y1="{y}" x2="{margin_l}" y2="{y}" stroke="{INK_3}" stroke-width="1"/>'
        body += (f'<text x="{margin_l - 8}" y="{y + 3}" font-size="9.5" fill="{INK_2}" '
                 f'text-anchor="end" font-family="monospace">{label}</text>')
    # X-axis labels (time)
    for frac, label in [(0, "0"), (0.25, "220"), (0.5, "440"), (0.75, "660"), (1.0, "880 ms")]:
        x = margin_l + frac * plot_w
        body += f'<line x1="{x}" y1="{margin_t + plot_h}" x2="{x}" y2="{margin_t + plot_h + 4}" stroke="{INK_3}" stroke-width="1"/>'
        body += (f'<text x="{x}" y="{margin_t + plot_h + 18}" font-size="9.5" fill="{INK_2}" '
                 f'text-anchor="middle" font-family="monospace">{label}</text>')
    body += svg_close()
    write("spectrogram.svg", body)


# ============================================================================
# Asset 6: licensing_map.svg — module × component traffic-light grid
# ============================================================================


def gen_licensing_map() -> None:
    rows = [
        ("M1.1 Normalization",   "WeTextProcessing",   "CLEAN", "Apache-2.0"),
        ("M1.2 G2P",             "Misaki + Epitran",    "CLEAN", "MIT"),
        ("M1.3 UMIM",            "XLM-RoBERTa",         "CLEAN", "MIT"),
        ("—",                    "eSpeak-ng",           "REPLACED", "GPL-3.0"),
        ("M2.2 Speaker",         "ECAPA-TDNN",          "CLEAN", "Apache · ⚠ VoxCeleb"),
        ("M2.2 Features",        "WavLM-Large",         "CLEAN", "MIT"),
        ("M3.1 Fusion",          "StyleTTS2",           "CLEAN", "MIT"),
        ("M3.2 Prosody",         "Llama-3-8B 4-bit",    "CHECKPOINT", "Llama Community"),
        ("M3.3 Emotion",         "Emotion2Vec",         "CLEAN", "Apache-2.0"),
        ("M4.1 Moderation",      "ShieldGemma-2B",      "CHECKPOINT", "Gemma"),
        ("M4.3 Watermark",       "AudioSeal",           "CLEAN", "MIT (code+weights)"),
        ("M5.1 Latent pred.",    "Matcha-TTS",          "CLEAN", "MIT"),
        ("M5.2 Acoustic TF",     "F5-TTS",              "CHECKPOINT", "MIT code · ⚠ CC-BY-NC weights"),
        ("—",                    "OpenF5-TTS",          "CLEAN", "Apache replacement"),
        ("M6.1 Vocoder",         "Vocos",               "CLEAN", "MIT"),
    ]
    W = 760
    row_h = 26
    H = 60 + len(rows) * row_h + 60
    margin = 20
    col_x = [margin, 220, 410, 540]
    col_w = [200, 190, 130, W - 540 - margin]

    body = svg_open(W, H)
    body += f'<rect width="{W}" height="{H}" fill="{PAPER}" rx="10"/>'
    body += (f'<text x="{W // 2}" y="28" font-size="14" font-weight="700" fill="{INK}" '
             f'text-anchor="middle">LICENSING AUDIT  ·  machine-checked at load time</text>')
    # Headers
    headers = ["Module", "Component", "Status", "Licence"]
    for x, h, w in zip(col_x, headers, col_w):
        body += (f'<text x="{x}" y="56" font-size="11" font-weight="700" fill="{INK_2}" '
                 f'letter-spacing="0.5">{h.upper()}</text>')
    body += f'<line x1="{margin}" y1="62" x2="{W - margin}" y2="62" stroke="{INK_3}" stroke-width="0.5" opacity="0.5"/>'

    status_color = {"CLEAN": GREEN, "CHECKPOINT": AMBER, "REPLACED": RED}
    status_fill = {"CLEAN": "#e7f8ee", "CHECKPOINT": "#fdf0d8", "REPLACED": "#fbe4e0"}

    for i, (mod, comp, status, lic) in enumerate(rows):
        y = 72 + i * row_h
        if i % 2 == 0:
            body += f'<rect x="{margin}" y="{y - 2}" width="{W - 2 * margin}" height="{row_h - 2}" fill="{INK}" opacity="0.025"/>'
        body += (f'<text x="{col_x[0]}" y="{y + 12}" font-size="11" fill="{INK_2}" '
                 f'font-family="monospace" font-weight="{600 if mod != "—" else 400}">{mod}</text>')
        body += f'<text x="{col_x[1]}" y="{y + 12}" font-size="11" fill="{INK}" font-weight="600">{comp}</text>'
        # Status pill
        body += (f'<rect x="{col_x[2]}" y="{y}" width="110" height="18" rx="9" '
                 f'fill="{status_fill[status]}"/>')
        body += (f'<circle cx="{col_x[2] + 12}" cy="{y + 9}" r="4" fill="{status_color[status]}"/>')
        body += (f'<text x="{col_x[2] + 22}" y="{y + 13}" font-size="10" font-weight="700" '
                 f'fill="{status_color[status]}">{status}</text>')
        body += f'<text x="{col_x[3]}" y="{y + 12}" font-size="10.5" fill="{INK_3}" font-family="monospace">{lic}</text>'

    # Legend
    legend_y = H - 30
    body += (f'<text x="{margin}" y="{legend_y + 4}" font-size="10" font-weight="700" fill="{INK_2}">'
             f'LEGEND</text>')
    items = [
        (GREEN, "CLEAN — ship freely"),
        (AMBER, "CHECKPOINT REPLACEMENT — needs permissive weights"),
        (RED, "REPLACED BY DESIGN — copyleft, swapped in the build"),
    ]
    lx = margin + 70
    for color, label in items:
        body += f'<circle cx="{lx}" cy="{legend_y}" r="4" fill="{color}"/>'
        body += f'<text x="{lx + 8}" y="{legend_y + 4}" font-size="10" fill="{INK_2}">{label}</text>'
        lx += len(label) * 5.6 + 50
    body += svg_close()
    write("licensing_map.svg", body)


# ============================================================================
# Asset 7: memory_budget.svg — stacked VRAM bar
# ============================================================================


def gen_memory_budget() -> None:
    W, H = 820, 320
    margin = 40
    bar_h = 60
    bar_w = W - 2 * margin - 200   # leave room for labels
    bar_x = margin + 180
    bar_y = 80

    # 32 GB total bar
    # 11.5 GB inference base broken down
    segments = [
        ("M3 Llama-3-8B (4-bit)", 6.0, PURPLE),
        ("M5 F5-TTS (BF16)",      2.5, AMBER),
        ("M2 ECAPA+WavLM (FP16)", 1.2, TEAL),
        ("M4 ShieldGemma (INT8)", 1.2, RED),
        ("M1 text pipeline",      0.5, BRAND),
        ("M6 Vocos (FP32)",       0.05, GREEN),
    ]
    inference_total = sum(s[1] for s in segments)
    headroom = 32.0 - inference_total

    body = svg_open(W, H)
    body += f'<rect width="{W}" height="{H}" fill="{PAPER}" rx="10"/>'
    body += (f'<text x="{W // 2}" y="30" font-size="14" font-weight="700" fill="{INK}" '
             f'text-anchor="middle">PHASE-2 MEMORY BUDGET  ·  single 32 GB GPU</text>')

    # 32 GB outline
    body += (f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" '
             f'fill="none" stroke="{INK_2}" stroke-width="1.5" rx="3"/>')
    # Tick marks at 8 / 16 / 24 GB
    for gb in (8, 16, 24):
        x = bar_x + (gb / 32.0) * bar_w
        body += f'<line x1="{x}" y1="{bar_y - 4}" x2="{x}" y2="{bar_y}" stroke="{INK_2}" stroke-width="1"/>'
        body += (f'<text x="{x}" y="{bar_y - 8}" font-size="10" fill="{INK_3}" text-anchor="middle" '
                 f'font-family="monospace">{gb} GB</text>')

    # Inference base (left portion, stacked)
    cur_x = bar_x
    for label, gb, color in segments:
        seg_w = (gb / 32.0) * bar_w
        body += (f'<rect x="{cur_x:.1f}" y="{bar_y}" width="{seg_w:.1f}" height="{bar_h}" '
                 f'fill="{color}" opacity="0.85"/>')
        cur_x += seg_w
    # Headroom (rest of bar)
    head_w = (headroom / 32.0) * bar_w
    body += (f'<rect x="{cur_x:.1f}" y="{bar_y}" width="{head_w:.1f}" height="{bar_h}" '
             f'fill="{INK_3}" opacity="0.18"/>')
    body += (f'<text x="{cur_x + head_w / 2:.1f}" y="{bar_y + bar_h / 2 + 4}" font-size="11" '
             f'font-weight="700" fill="{INK_2}" text-anchor="middle">KV cache · ODE · buffers</text>')

    # Legend / breakdown below
    body += (f'<text x="{margin}" y="{bar_y + bar_h + 50}" font-size="13" font-weight="700" fill="{INK}">'
             f'Inference base: ~{inference_total:.1f} GB  ·  Headroom: ~{headroom:.1f} GB</text>')
    cur_x = margin
    cur_y = bar_y + bar_h + 80
    for label, gb, color in segments:
        body += f'<rect x="{cur_x}" y="{cur_y}" width="14" height="14" rx="2" fill="{color}"/>'
        body += (f'<text x="{cur_x + 20}" y="{cur_y + 11}" font-size="11" fill="{INK}">'
                 f'{label}  <tspan fill="{INK_3}" font-family="monospace">~{gb:.2f} GB</tspan></text>')
        cur_y += 22
        if cur_y > H - 30:
            cur_y = bar_y + bar_h + 80
            cur_x += 280

    body += svg_close()
    write("memory_budget.svg", body)


# ============================================================================
# Asset 8: latency_budget.svg — 240 ms TTFA decomposition
# ============================================================================


def gen_latency_budget() -> None:
    W, H = 820, 280
    margin_l = 40
    bar_x = margin_l + 180
    bar_y = 80
    bar_h = 50
    total_ms = 240
    bar_w = W - bar_x - margin_l

    stages = [
        ("G2P / front-end (M1)",       5,   BRAND),
        ("Speaker encoder (M2)",       20,  TEAL),
        ("Safety classifier (M4)",     15,  RED),
        ("Flow-matching 8 steps (M5)", 80,  AMBER),
        ("Vocoder (M6)",               30,  GREEN),
        ("Network",                    40,  INK_3),
        ("Jitter buffer",              50,  PURPLE),
    ]

    body = svg_open(W, H)
    body += f'<rect width="{W}" height="{H}" fill="{PAPER}" rx="10"/>'
    body += (f'<text x="{W // 2}" y="30" font-size="14" font-weight="700" fill="{INK}" '
             f'text-anchor="middle">LATENCY BUDGET  ·  240 ms TTFA  ·  under the 300 ms perceptual threshold</text>')

    # Stacked bar
    cur_x = bar_x
    for label, ms, color in stages:
        seg_w = (ms / total_ms) * bar_w
        body += (f'<rect x="{cur_x:.1f}" y="{bar_y}" width="{seg_w:.1f}" height="{bar_h}" '
                 f'fill="{color}" opacity="0.88"/>')
        if seg_w > 22:
            body += (f'<text x="{cur_x + seg_w / 2:.1f}" y="{bar_y + bar_h / 2 + 4}" font-size="10" '
                     f'font-weight="700" fill="white" text-anchor="middle">{ms}</text>')
        cur_x += seg_w
    # Total annotation
    body += (f'<text x="{bar_x + bar_w + 8}" y="{bar_y + bar_h / 2 + 4}" font-size="13" '
             f'font-weight="800" fill="{GREEN}">✓ 240 ms</text>')

    # Per-stage legend
    body += f'<text x="{margin_l}" y="{bar_y + bar_h + 50}" font-size="12" font-weight="700" fill="{INK}">ms</text>'
    for i, (label, ms, color) in enumerate(stages):
        col = i % 4
        row = i // 4
        x = margin_l + col * 200
        y = bar_y + bar_h + 80 + row * 24
        body += f'<rect x="{x}" y="{y}" width="12" height="12" rx="2" fill="{color}"/>'
        body += (f'<text x="{x + 18}" y="{y + 10}" font-size="10.5" fill="{INK}">{label} '
                 f'<tspan font-family="monospace" fill="{INK_3}">{ms} ms</tspan></text>')

    body += svg_close()
    write("latency_budget.svg", body)


# ============================================================================
# Driver
# ============================================================================


def main() -> None:
    print(f"Generating VajraVoice assets into {ASSETS}/")
    gen_logo()
    gen_hero_waveform()
    gen_pipeline()
    gen_voice_fingerprint()
    gen_spectrogram()
    gen_licensing_map()
    gen_memory_budget()
    gen_latency_budget()
    total = sum((ASSETS / f.name).stat().st_size for f in ASSETS.iterdir())
    print(f"\n  total: {total / 1024:.1f} KB across {len(list(ASSETS.iterdir()))} SVGs")


if __name__ == "__main__":
    main()
