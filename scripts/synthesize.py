"""CLI: synthesize speech from text.

Examples:

    # stub mode (no GPU, no weights — for demo / wiring verification)
    python -m scripts.synthesize --text "Namaskar" --config configs/stub.yaml

    # real mode (needs the heavy ML stack on a CUDA box)
    python -m scripts.synthesize --text "नमस्कार" \\
        --voice vp_marathi_mf_asha --config configs/default.yaml \\
        --reference ref.wav --out out.mp3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m scripts.synthesize` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vajravoice.config import VajraVoiceConfig
from vajravoice.pipeline import VajraVoicePipeline


def main() -> int:
    p = argparse.ArgumentParser(
        prog="vajravoice-synthesize",
        description="Synthesize speech with the VajraVoice pipeline.",
    )
    p.add_argument("--text", required=True, help="Text to synthesize.")
    p.add_argument("--config", default="configs/stub.yaml",
                   help="Pipeline config YAML (default: configs/stub.yaml).")
    p.add_argument("--voice", default="vp_marathi_mf_asha",
                   help="Signed voice profile id (default: Marathi Asha).")
    p.add_argument("--reference", help="Reference audio for uncached voice (5–60s wav).")
    p.add_argument("--language", default="auto")
    p.add_argument("--out", default="out.wav", help="Output audio path (stub emits PCM s16le).")
    p.add_argument("--stream", action="store_true",
                   help="Stream 20ms packets instead of one batch buffer.")
    args = p.parse_args()

    config = VajraVoiceConfig.from_yaml(args.config)
    pipeline = VajraVoicePipeline(config).load()
    print(f"VajraVoice pipeline: {pipeline}", file=sys.stderr)

    if args.stream:
        n_packets = 0
        with open(args.out, "wb") as f:
            for packet in pipeline.stream(
                text=args.text,
                voice_profile_id=args.voice,
                reference_audio=_read_ref(args.reference),
                language=args.language,
            ):
                f.write(packet)
                n_packets += 1
        seconds = n_packets * 0.020
        print(f"Streamed {n_packets} packets ({seconds:.2f}s) → {args.out}", file=sys.stderr)
        return 0

    result = pipeline.synthesize(
        text=args.text,
        voice_profile_id=args.voice,
        reference_audio=_read_ref(args.reference),
        language=args.language,
    )
    with open(args.out, "wb") as f:
        f.write(result.audio)
    print(
        f"\n✓ Synthesized {result.audio_seconds:.2f}s of audio → {args.out}\n"
        f"  request_id : {result.request_id}\n"
        f"  generation : {result.generation_ms:.1f} ms\n"
        f"  TTFA       : {result.ttfa_ms:.1f} ms\n"
        f"  RTF        : {result.rtf:.3f}\n"
        f"  watermark  : {result.watermark['scheme']} (p={result.watermark['detectable_post_transform_p']})\n"
        f"  audit_id   : {result.audit_id}",
        file=sys.stderr,
    )
    return 0


def _read_ref(path: str | None) -> bytes | None:
    if path is None:
        return None
    return Path(path).read_bytes()


if __name__ == "__main__":
    sys.exit(main())
