"""Pydantic schemas for the VajraVoice API (matches ADD §VIII data contracts).

These are the wire-level request/response models. They mirror the JSON
payloads in the ADD exactly (enroll req/resp, speech req/200/403, WS stream
protocol). Sample payloads from the ADD are reproduced in the docstrings so
the API surface is self-documenting.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /v1/voices/enroll
# ---------------------------------------------------------------------------


class ConsentChallenge(BaseModel):
    challenge_id: str
    spoken_phrase_hash: str
    transcript_agreement: bool = True
    liveness_check: Optional[dict] = None


class ReferenceAudio(BaseModel):
    encoding: str = "wav"
    sample_rate: int = 24000
    channels: int = 1
    duration_sec: float
    source: str                          # data:audio/wav;base64,<...>


class EnrollRequest(BaseModel):
    """POST /v1/voices/enroll — request.

    Example payload (ADD Contract 1):
        {
          "reference_audio": {"encoding":"wav","sample_rate":24000,"channels":1,
                              "duration_sec":32.5,"source":"data:audio/wav;base64,<...>"},
          "consent_challenge": {"challenge_id":"ch_8f3a...","spoken_phrase_hash":"sha256:9c1b...",
                                "transcript_agreement":true,
                                "liveness_check":{"passed":true,"method":"face + voice match","score":0.97}},
          "permitted_use": ["synthesis","streaming"],
          "permitted_languages": ["mr","hi","en"],
          "expiry": "2027-07-17T00:00:00Z",
          "owner": {"subject_id":"user_42","display_name":"Asha M."}
        }
    """

    reference_audio: ReferenceAudio
    consent_challenge: ConsentChallenge
    permitted_use: list[str] = Field(default_factory=lambda: ["synthesis", "streaming"])
    permitted_languages: list[str] = Field(default_factory=lambda: ["mr", "hi", "en"])
    expiry: str
    owner: dict


class EnrollResponse(BaseModel):
    """POST /v1/voices/enroll — signed response (ADD Contract 2)."""

    voice_profile_id: str
    status: str = "enrolled"
    identity_anchor: dict                 # {algorithm, dim, hash}
    style_anchor: Optional[dict] = None   # {arousal, valence, speaking_rate}
    language_distribution: Optional[dict] = None
    consent: dict                         # {token, issued, expires, revocable}
    diagnostics: Optional[dict] = None    # {snr_db, cloning_confidence}


# ---------------------------------------------------------------------------
# /v1/audio/speech
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech — request (ADD Contract 3).

    Example:
        {
          "text": "नमस्कार, आज आपण एक नवीन प्रकल्प बद्दल बोलू.",
          "voice_profile_id": "vp_01HZX...",
          "language": "auto",
          "profile": "interactive",
          "emotion": {"arousal": 0.5, "valence": 0.6},
          "pronunciation_overrides": [{"word": "प्रकल्प", "ipa": "pɾəkəlp"}],
          "output": {"format": "mp3", "sample_rate": 24000},
          "seed": 17
        }
    """

    text: str
    voice_profile_id: str
    language: str = "auto"
    profile: str = "interactive"           # "interactive" | "studio"
    emotion: Optional[dict] = None
    pronunciation_overrides: list[dict] = Field(default_factory=list)
    output: dict = Field(default_factory=lambda: {"format": "mp3", "sample_rate": 24000})
    stream: bool = False
    seed: int = 17


class SpeechResponse(BaseModel):
    """POST /v1/audio/speech — 200 OK (ADD Contract 3)."""

    request_id: str
    audio_seconds: float
    generation_ms: float
    ttfa_ms: float
    rtf: float
    watermark: dict
    audit_id: str
    # audio bytes returned out-of-band (HTTP body / base64) to keep the JSON
    # metadata payload small.


class ConsentDeniedResponse(BaseModel):
    """POST /v1/audio/speech — 403 (fail-closed; no audio generated)."""

    request_id: str
    error: str = "consent_denied"
    reason: str
    audit_id: str
    audio_generated: bool = False


# ---------------------------------------------------------------------------
# /v1/audio/stream (WebSocket) — message envelopes
# ---------------------------------------------------------------------------


class StreamClientMessage(BaseModel):
    """Client → Server messages (ADD Contract 4).

    Message types: session.start · text.append · text.commit · style.update · cancel
    """

    type: str
    voice_profile_id: Optional[str] = None
    text: Optional[str] = None
    profile: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[dict] = None


class StreamServerMessage(BaseModel):
    """Server → Client messages (ADD Contract 4).

    Message types: audio.chunk · alignment · session.end
    """

    type: str
    seq: Optional[int] = None
    encoding: Optional[str] = "pcm_s16le"
    sample_rate: Optional[int] = 24000
    duration_ms: Optional[int] = 20
    data: Optional[str] = None            # base64 for JSON; raw bytes for WS binary frames
    word: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    watermark_detectable_p: Optional[float] = None
    audit_id: Optional[str] = None
