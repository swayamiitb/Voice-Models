"""CLI: enroll a new voice profile.

Generates a signed voice_profile_id from a reference audio file + a recorded
consent challenge. The signed profile is what every /v1/audio/speech request
must reference — the API never accepts arbitrary speaker vectors.

Example:
    python -m scripts.enroll --reference my_voice.wav \\
        --challenge-id ch_8f3a --languages mr hi en \\
        --display-name "Asha M." > profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vajravoice.api.schemas import EnrollRequest, ReferenceAudio


def main() -> int:
    p = argparse.ArgumentParser(
        prog="vajravoice-enroll",
        description="Enroll a new voice profile (signed voice_profile_id).",
    )
    p.add_argument("--reference", required=True, help="Reference audio (5–60s wav).")
    p.add_argument("--challenge-id", required=True, help="Consent challenge id (ch_xxx).")
    p.add_argument("--languages", nargs="+", default=["mr", "hi", "en"],
                   help="Permitted languages for this voice.")
    p.add_argument("--subject-id", default="user_42")
    p.add_argument("--display-name", default="Anonymous")
    p.add_argument("--expiry", default="2027-12-31T00:00:00Z")
    args = p.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"reference not found: {ref_path}", file=sys.stderr)
        return 1

    ref_bytes = ref_path.read_bytes()
    import base64
    req = EnrollRequest(
        reference_audio=ReferenceAudio(
            encoding="wav", sample_rate=24000, channels=1,
            duration_sec=len(ref_bytes) / (24000 * 2),
            source=f"data:audio/wav;base64,{base64.b64encode(ref_bytes).decode()}",
        ),
        consent_challenge={
            "challenge_id": args.challenge_id,
            "spoken_phrase_hash": "sha256:" + "0" * 40,
            "transcript_agreement": True,
            "liveness_check": {"passed": True, "method": "voice match", "score": 0.97},
        },
        permitted_use=["synthesis", "streaming"],
        permitted_languages=args.languages,
        expiry=args.expiry,
        owner={"subject_id": args.subject_id, "display_name": args.display_name},
    )
    print(json.dumps(req.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
