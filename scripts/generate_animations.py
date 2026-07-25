"""Generate animated GIFs for the VajraVoice README.

Each animation is driven by real speech-shaped DSP synthesized in code — no
stock footage, no external assets. Four visualizations:

    oscilloscope.gif       — waveform of synthesized speech, scrolling left
    eq_bars.gif            — 16-band spectrum analyzer, peak-hold markers
    fingerprint_pulse.gif  — radial speaker-identity blob, breathing
    spectrogram_scroll.gif — waterfall spectrogram, new frames entering right

Outputs into assets/ (alongside the static SVGs). Total size kept under
~600 KB so they load instantly on GitHub.

Usage:
    python scripts/generate_animations.py

Deps: numpy + Pillow. No torch, no ffmpeg.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# ─── palette ───────────────────────────────────────────────────────────────
INK        = (11, 20, 36)
INK_2      = (26, 37, 64)
INK_3      = (45, 58, 85)
PAPER      = (247, 249, 252)
PAPER_2    = (228, 234, 244)
BRAND      = (76, 141, 255)
BRAND_2    = (122, 169, 255)
BRAND_DEEP = (18, 33, 63)
AMBER      = (245, 166, 35)
AMBER_2    = (255, 196, 92)
GREEN      = (46, 204, 113)
GREEN_2    = (110, 230, 160)
RED        = (231, 76, 60)
TEAL       = (6, 182, 212)
TEAL_2     = (90, 210, 230)
PURPLE     = (168, 85, 247)
PURPLE_2   = (200, 130, 250)


# ═══════════════════════════════════════════════════════════════════════════
# Speech synthesis (reused from generate_assets, simplified)
# ═══════════════════════════════════════════════════════════════════════════


def _envelope(t, attack=0.02, release=0.20):
    env = np.ones_like(t)
    env[t < attack] = t[t < attack] / attack
    rel = t > (1.0 - release)
    env[rel] = np.cos(np.pi * (t[rel] - (1.0 - release)) / (2 * release)) ** 2
    return env


def synth_syllable(duration, f0, formants, sr=22050, noise=0.0):
    n = int(duration * sr)
    t = np.arange(n) / sr
    phase = (t * f0) % 1.0
    glottal = np.where(phase < 0.6, np.sin(np.pi * phase / 0.6) ** 2, 0.0)
    if noise > 0:
        glottal = glottal + noise * np.random.default_rng(42).standard_normal(n)
    sig = glottal.copy()
    for freq, bw in formants:
        r = math.exp(-math.pi * bw / sr)
        c = 2 * r * math.cos(2 * math.pi * freq / sr)
        out = np.zeros(n); out[0] = sig[0]; out[1] = sig[1] + c * out[0]
        for i in range(2, n):
            out[i] = sig[i] + c * out[i - 1] - r * r * out[i - 2]
        sig = (1 - r) * out + 0.3 * sig
    return sig * _envelope(t)


def synth_phrase(sr=22050):
    parts = []
    for dur, f0, fmt, noise in [
        (0.18, 145, [(720, 90), (1240, 110), (2800, 180)], 0.0),
        (0.20, 138, [(680, 95), (1300, 120), (2700, 200)], 0.10),
        (0.30, 120, [(740, 80), (1180, 100), (2900, 170)], 0.0),
        (0.06, 110, [(400, 100), (1500, 200), (2600, 250)], 0.0),
        (0.14, 105, [(720, 90), (1240, 110), (2800, 180)], 0.0),
    ]:
        parts.append(synth_syllable(dur, f0, fmt, sr, noise))
        parts.append(np.zeros(int(0.04 * sr)))
    return np.concatenate(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _font(size: int):
    """Try to load a real font; fall back to the default bitmap font."""
    for path in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _save_gif(name: str, frames: list, duration_ms: int = 60, loop: int = 0) -> None:
    """Save a list of PIL frames as an animated GIF."""
    path = ASSETS / name
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=loop, optimize=True, disposal=2,
    )
    size = path.stat().st_size
    print(f"  ✓ {name:<28} {len(frames)} frames  {size / 1024:.1f} KB")


# ═══════════════════════════════════════════════════════════════════════════
# 1) oscilloscope.gif — scrolling waveform of real speech
# ═══════════════════════════════════════════════════════════════════════════


def gen_oscilloscope() -> None:
    W, H = 800, 200
    sr = 22050
    sig = synth_phrase(sr)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    n_frames = 60
    win = 400  # samples shown per frame (a few glottal pulses)
    mid = H // 2
    f_h = _font(11)
    frames = []
    for fi in range(n_frames):
        # Loop the phrase smoothly
        start = int((fi / n_frames) * (len(sig) - win - 1))
        chunk = sig[start : start + win]
        img = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(img)
        # Grid
        for y in range(0, H, 40):
            d.line([(0, y), (W, y)], fill=PAPER_2, width=1)
        for x in range(0, W, 80):
            d.line([(x, 0), (x, H)], fill=PAPER_2, width=1)
        d.line([(0, mid), (W, mid)], fill=INK_3, width=1)
        # Waveform — gradient effect via overlapping strokes
        pts = [(int(i * W / win), int(mid - chunk[i] * (mid - 16))) for i in range(win)]
        # Glow underlay
        d.line(pts, fill=BRAND_2, width=5, joint="curve")
        d.line(pts, fill=BRAND, width=2, joint="curve")
        # Caption
        d.text((12, 8), "OSCILLOSCOPE  ·  synthesized 24 kHz speech",
               fill=INK_2, font=f_h)
        frames.append(img)
    _save_gif("oscilloscope.gif", frames, duration_ms=60)


# ═══════════════════════════════════════════════════════════════════════════
# 2) eq_bars.gif — 16-band spectrum analyzer with peak-hold markers
# ═══════════════════════════════════════════════════════════════════════════


def gen_eq_bars() -> None:
    W, H = 800, 280
    sr = 22050
    sig = synth_phrase(sr)
    n_bands = 16
    n_frames = 60
    n_fft = 512
    win = np.hanning(n_fft)
    # Pre-compute spectrum frames
    spec_frames = []
    hop = (len(sig) - n_fft) // n_frames
    for i in range(n_frames):
        s = sig[i * hop : i * hop + n_fft] * win
        mag = np.abs(np.fft.rfft(s))
        # Log-spaced band aggregation
        bands = np.zeros(n_bands)
        freqs = np.linspace(0, sr / 2, len(mag))
        edges = np.logspace(np.log10(80), np.log10(sr / 2), n_bands + 1)
        for b in range(n_bands):
            mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
            bands[b] = mag[mask].max() if mask.any() else 0.0
        spec_frames.append(bands / (bands.max() + 1e-9))

    # Peak-hold state
    peaks = np.zeros(n_bands)
    peak_vel = np.zeros(n_bands)
    bar_w = (W - 40) // n_bands
    f_h = _font(11)
    frames = []
    for fi in range(n_frames):
        bands = spec_frames[fi]
        # Update peak-hold
        for b in range(n_bands):
            if bands[b] > peaks[b]:
                peaks[b] = bands[b]; peak_vel[b] = 0
            else:
                peak_vel[b] += 0.012
                peaks[b] = max(0, peaks[b] - peak_vel[b])
        img = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(img)
        # Baseline
        d.line([(20, H - 32), (W - 20, H - 32)], fill=INK_3, width=1)
        for b in range(n_bands):
            x = 20 + b * bar_w + 4
            h = max(2, int(bands[b] * (H - 60)))
            y = H - 32 - h
            # Gradient: low=teal, mid=brand, high=amber
            if bands[b] < 0.4:
                col = TEAL
            elif bands[b] < 0.75:
                col = BRAND
            else:
                col = AMBER
            d.rectangle([x, y, x + bar_w - 8, H - 33], fill=col)
            # Cap highlight
            d.rectangle([x, y, x + bar_w - 8, y + 3], fill=(255, 255, 255))
            # Peak-hold marker
            py = max(2, H - 32 - int(peaks[b] * (H - 60)))
            d.rectangle([x, py - 2, x + bar_w - 8, py + 1], fill=RED)
        d.text((12, 8), "SPECTRUM  ·  16-band analyzer with peak-hold",
               fill=INK_2, font=f_h)
        frames.append(img)
    _save_gif("eq_bars.gif", frames, duration_ms=70)


# ═══════════════════════════════════════════════════════════════════════════
# 3) fingerprint_pulse.gif — radial ECAPA-TDNN speaker blob breathing
# ═══════════════════════════════════════════════════════════════════════════


def gen_fingerprint_pulse() -> None:
    W = H = 320
    cx = cy = W // 2
    # Base embedding (deterministic)
    rng = np.random.default_rng(2026)
    raw = rng.normal(0, 1, 256)
    kernel = np.ones(8) / 8
    smooth = np.convolve(raw, kernel, mode="same")
    base = 60 + 100 * (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    angles = np.linspace(0, 2 * math.pi, 256, endpoint=False)
    n_frames = 32
    f_h = _font(11)
    frames = []
    for fi in range(n_frames):
        phase = fi / n_frames * 2 * math.pi
        # Two-frequency breathing: subtle pulsation
        breathe = 1.0 + 0.08 * math.sin(phase) + 0.04 * math.sin(2 * phase + 0.7)
        # Slight rotation
        rot = phase * 0.15
        radii = base * breathe
        img = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(img)
        # Concentric guides
        for r in (60, 100, 140, 170):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=PAPER_2)
        # Outer glow ring
        glow_r = int(170 * breathe + 8)
        d.ellipse([cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r],
                  outline=(*BRAND_2, 80), width=2)
        # Polygon
        pts = []
        for a, r in zip(angles, radii):
            aa = a + rot
            pts.append((cx + r * math.cos(aa), cy + r * math.sin(aa)))
        d.polygon(pts, outline=BRAND)
        # Fill with low opacity by stacking multiple scaled copies
        for scale, alpha_col in [(0.96, (180, 210, 255)), (0.85, (200, 220, 255))]:
            inner = [(cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale) for p in pts]
            d.polygon(inner, fill=alpha_col)
        # Spokes
        for i in range(0, 256, 8):
            a = angles[i] + rot
            r = radii[i]
            x2, y2 = cx + r * math.cos(a), cy + r * math.sin(a)
            d.line([(cx, cy), (x2, y2)], fill=BRAND_2, width=1)
        # Outer dots
        for i in range(0, 256, 8):
            a = angles[i] + rot
            r = radii[i]
            x2, y2 = cx + r * math.cos(a), cy + r * math.sin(a)
            d.ellipse([x2 - 2, y2 - 2, x2 + 2, y2 + 2], fill=TEAL)
        # Hub
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=INK)
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=PAPER)
        d.text((10, 10), "VOICE FINGERPRINT  ·  256-d ECAPA-TDNN", fill=INK_2, font=f_h)
        frames.append(img)
    _save_gif("fingerprint_pulse.gif", frames, duration_ms=60)


# ═══════════════════════════════════════════════════════════════════════════
# 4) spectrogram_scroll.gif — waterfall, new frames enter from the right
# ═══════════════════════════════════════════════════════════════════════════


def _heat_rgb(v: float) -> tuple[int, int, int]:
    """Viridis-like colormap, v in [0,1]."""
    if v < 0.25:
        t = v / 0.25
        return (int(13 + 30 * t), int(8 + 40 * t), int(60 + 80 * t))
    elif v < 0.5:
        t = (v - 0.25) / 0.25
        return (43, int(48 + 70 * t), int(140 + 60 * t))
    elif v < 0.75:
        t = (v - 0.5) / 0.25
        return (int(43 + 60 * t), int(118 + 80 * t), int(200 - 100 * t))
    else:
        t = (v - 0.75) / 0.25
        return (int(103 + 150 * t), int(198 + 30 * t), int(100 - 60 * t))


def gen_spectrogram_scroll() -> None:
    W, H = 800, 280
    sr = 22050
    sig = synth_phrase(sr)
    n_fft = 256
    hop = 200
    win = np.hanning(n_fft)
    # Compute full spectrogram once
    cols = []
    for i in range(0, len(sig) - n_fft, hop):
        spec = np.abs(np.fft.rfft(sig[i:i + n_fft] * win))[:80]
        cols.append(spec)
    spec = np.array(cols).T  # [80 freq bins, T time frames]
    spec = np.clip((spec - spec.min()) / (spec.max() - spec.min() + 1e-9), 0, 1)
    n_freq_bins = 20
    spec_pooled = spec[: (spec.shape[0] // 4) * 4].reshape(
        spec.shape[0] // 4, 4, -1
    ).max(axis=1)[:n_freq_bins]
    spec_pooled = spec_pooled[:, ::-1]  # so it scrolls forward in time

    n_total_cols = spec_pooled.shape[1]
    cols_visible = 50
    n_frames = 60
    f_h = _font(11)
    frames = []
    for fi in range(n_frames):
        img = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(img)
        # Title strip
        d.rectangle([0, 0, W, 28], fill=INK)
        d.text((12, 8), "SPECTROGRAM  ·  waterfall · formant tracks visible",
               fill=PAPER, font=f_h)
        plot_top = 36
        plot_h = H - plot_top - 28
        plot_w = W - 20
        cell_w = plot_w / cols_visible
        cell_h = plot_h / n_freq_bins
        # Window of columns: slide through spec, looping
        for ti in range(cols_visible):
            src_i = (fi - cols_visible + ti) % n_total_cols
            for fi_bin in range(n_freq_bins):
                v = spec_pooled[n_freq_bins - 1 - fi_bin, src_i]
                if v < 0.18:
                    continue
                x = 10 + ti * cell_w
                y = plot_top + fi_bin * cell_h
                d.rectangle([x, y, x + cell_w + 1, y + cell_h + 1], fill=_heat_rgb(v))
        # X axis (time)
        d.line([(10, plot_top + plot_h), (10 + plot_w, plot_top + plot_h)],
               fill=INK_3, width=1)
        d.text((10 + plot_w - 60, H - 20), "time →", fill=INK_3, font=f_h)
        frames.append(img)
    _save_gif("spectrogram_scroll.gif", frames, duration_ms=50)


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"Generating animated GIFs into {ASSETS}/")
    gen_oscilloscope()
    gen_eq_bars()
    gen_fingerprint_pulse()
    gen_spectrogram_scroll()
    gif_total = sum((ASSETS / f.name).stat().st_size for f in ASSETS.glob("*.gif"))
    svg_total = sum((ASSETS / f.name).stat().st_size for f in ASSETS.glob("*.svg"))
    print(f"\n  GIFs: {gif_total / 1024:.1f} KB across {len(list(ASSETS.glob('*.gif')))} files")
    print(f"  SVGs: {svg_total / 1024:.1f} KB across {len(list(ASSETS.glob('*.svg')))} files")
    print(f"  total assets: {(gif_total + svg_total) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
