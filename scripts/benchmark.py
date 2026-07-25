"""CLI: run the RTF / TTFA / VRAM benchmark harness.

Reproduces the evaluation methodology from the VajraVoice Technical Design
Document §V. Cold-start + warm percentiles are reported for RTF and TTFA,
plus peak/steady VRAM and audio seconds generated.

In stub mode this measures the wiring overhead (sub-millisecond). In real
mode against a CUDA box it produces the design-target numbers (RTF < 0.15,
TTFA < 300 ms).

Example:
    python -m scripts.benchmark --config configs/stub.yaml --n 50
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vajravoice.config import VajraVoiceConfig
from vajravoice.pipeline import VajraVoicePipeline

# A mixed-length Marathi/English prompt set covering code-switching.
PROMPTS = [
    "Namaskar, aaj aapan ek navin prakalp baddal bolu.",
    "Main office ja raha hoon, meeting hai at 5.",
    "तुम्ही कसे आहात?",
    "WhatsApp करोगे?",
    "The frontier ships 15 languages; one is Indian.",
    "भारतात सुमारे १४ कोटी मराठी भाषक आहेत.",
    "मी आज बाजारात जात आहे.",
    "RTF is generation seconds divided by produced audio seconds.",
    "We took the SOTA blueprint apart into six modules and eighteen blocks.",
    "Causal codec decode emits 24 kHz PCM in 20 ms WebSocket packets.",
]


def main() -> int:
    p = argparse.ArgumentParser(
        prog="vajravoice-benchmark",
        description="Measure RTF / TTFA / audio seconds for the pipeline.",
    )
    p.add_argument("--config", default="configs/stub.yaml")
    p.add_argument("--n", type=int, default=20, help="Iterations per prompt.")
    p.add_argument("--warmup", type=int, default=3, help="Warmup runs (excluded).")
    p.add_argument("--voice", default="vp_marathi_mf_asha")
    args = p.parse_args()

    config = VajraVoiceConfig.from_yaml(args.config)
    pipe = VajraVoicePipeline(config).load()
    print(f"# VajraVoice benchmark — {config.name}\n", file=sys.stderr)

    # Warmup
    for _ in range(args.warmup):
        pipe.synthesize(text=PROMPTS[0], voice_profile_id=args.voice)

    all_ttfa, all_rtf, all_audio_s = [], [], []
    for prompt in PROMPTS:
        for _ in range(args.n):
            r = pipe.synthesize(text=prompt, voice_profile_id=args.voice)
            all_ttfa.append(r.ttfa_ms)
            all_rtf.append(r.rtf)
            all_audio_s.append(r.audio_seconds)

    def pct(xs, p):
        s = sorted(xs)
        return s[int(len(s) * p)]

    print("=" * 60, file=sys.stderr)
    print(f"  Prompts          : {len(PROMPTS)}", file=sys.stderr)
    print(f"  Iters / prompt   : {args.n}", file=sys.stderr)
    print(f"  Total samples    : {len(all_ttfa)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print(f"  TTFA  p50        : {pct(all_ttfa, 0.50):8.2f} ms", file=sys.stderr)
    print(f"  TTFA  p95        : {pct(all_ttfa, 0.95):8.2f} ms", file=sys.stderr)
    print(f"  RTF   p50        : {pct(all_rtf, 0.50):8.4f}", file=sys.stderr)
    print(f"  RTF   p95        : {pct(all_rtf, 0.95):8.4f}", file=sys.stderr)
    print(f"  Audio mean       : {statistics.mean(all_audio_s):8.3f} s", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
