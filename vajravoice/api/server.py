"""FastAPI server exposing the three ADD endpoints.

  POST /v1/voices/enroll   — voice enrollment + signed profile issuance
  POST /v1/audio/speech    — synchronous synthesis (or fail-closed 403)
  WS   /v1/audio/stream    — bidirectional streaming synthesis

Start in stub mode (no GPU, no weights):

    uvicorn vajravoice.api.server:app --reload

The pipeline is built once at app startup from VAJRAVOICE_CONFIG (defaults to
configs/stub.yaml). Switch configs by setting the env var.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import VajraVoiceConfig
from ..modules.m4_guardrails import ConsentDenied
from ..pipeline import VajraVoicePipeline
from .schemas import (
    ConsentDeniedResponse,
    EnrollRequest,
    EnrollResponse,
    SpeechRequest,
    SpeechResponse,
)

log = logging.getLogger("vajravoice.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

# ---------------------------------------------------------------------------
# App + pipeline bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VajraVoice — Neural TTS Engine",
    description=(
        "Open-source MAI-Voice-2-class neural TTS for Indian languages. "
        "Six-module pipeline: linguistic → reference → fusion+prosody → "
        "guardrails → generator → vocoder. MIT/Apache stack, consent-gated, "
        "watermarked, streaming 24 kHz."
    ),
    version="0.1.0",
    contact={"name": "swayamiitb", "url": "https://github.com/swayamiitb/Voice-Models"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_pipeline() -> None:
    """Build + load the pipeline once at app startup."""
    cfg = VajraVoiceConfig.from_env()
    app.state.pipeline = VajraVoicePipeline(cfg).load()
    log.info("VajraVoice pipeline ready: %s", app.state.pipeline)


def _pipeline() -> VajraVoicePipeline:
    return app.state.pipeline


# ---------------------------------------------------------------------------
# Health & metadata
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "vajravoice", "version": "0.1.0"}


@app.get("/v1/voices")
def list_voices() -> dict:
    """List enrolled voice profiles (placeholder: returns the four Marathi voices
    shipped with the bol-tts-marathi Phase-1 prototype)."""
    return {
        "voices": [
            {"voice_profile_id": "vp_marathi_mf_asha",     "lang": "mr", "gender": "F"},
            {"voice_profile_id": "vp_marathi_mf_mukta",    "lang": "mr", "gender": "F"},
            {"voice_profile_id": "vp_marathi_mm_vivek",    "lang": "mr", "gender": "M"},
            {"voice_profile_id": "vp_marathi_mm_dnyanesh", "lang": "mr", "gender": "M"},
        ]
    }


# ---------------------------------------------------------------------------
# POST /v1/voices/enroll
# ---------------------------------------------------------------------------


@app.post("/v1/voices/enroll", response_model=EnrollResponse)
def enroll(req: EnrollRequest) -> EnrollResponse:
    """Enroll a new voice. Requires randomized spoken challenge, transcript
    agreement, liveness checks, and identity consistency. Returns a signed
    voice_profile_id — the API never accepts arbitrary speaker vectors."""
    import hashlib, time

    # In production this would invoke M2 on the reference audio, store the
    # signed profile, and issue a consent token from the Consent Service.
    # The stub path synthesises a deterministic profile id from the request.
    vp_id = "vp_" + hashlib.sha1(
        (req.owner.get("subject_id", "anon") + str(req.reference_audio.duration_sec)).encode()
    ).hexdigest()[:13]

    return EnrollResponse(
        voice_profile_id=vp_id,
        status="enrolled",
        identity_anchor={"algorithm": "ECAPA-TDNN", "dim": 256, "hash": "sha256:..."},
        style_anchor={"arousal": 0.42, "valence": 0.61, "speaking_rate": 0.88},
        language_distribution={"mr": 0.71, "hi": 0.22, "en": 0.07},
        consent={
            "token": "tok_signed_by_consent_service",
            "issued": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "expires": req.expiry,
            "revocable": True,
        },
        diagnostics={"snr_db": 31.2, "cloning_confidence": 0.94},
    )


# ---------------------------------------------------------------------------
# POST /v1/audio/speech
# ---------------------------------------------------------------------------


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    """Synchronous synthesis. Returns audio bytes + metadata, or 403 fail-closed."""
    try:
        result = _pipeline().synthesize(
            text=req.text,
            voice_profile_id=req.voice_profile_id,
            language=req.language,
            emotion=req.emotion,
            pronunciation_overrides=req.pronunciation_overrides,
            seed=req.seed,
        )
    except ConsentDenied as e:
        # Fail-closed: no audio generated. Availability failure ≠ security failure.
        return JSONResponse(
            status_code=403,
            content=ConsentDeniedResponse(
                request_id="req_" + req.voice_profile_id[-8:],
                reason=str(e),
                audit_id="aud_" + req.voice_profile_id[-12:],
            ).model_dump(),
        )

    # Audio bytes go in the body; metadata as a custom header (or base64 inline).
    meta = SpeechResponse(
        request_id=result.request_id,
        audio_seconds=result.audio_seconds,
        generation_ms=result.generation_ms,
        ttfa_ms=result.ttfa_ms,
        rtf=result.rtf,
        watermark=result.watermark,
        audit_id=result.audit_id,
    )
    return JSONResponse(
        content={
            **meta.model_dump(),
            "audio_base64": _b64(result.audio),
        }
    )


# ---------------------------------------------------------------------------
# WS /v1/audio/stream
# ---------------------------------------------------------------------------


@app.websocket("/v1/audio/stream")
async def stream(ws: WebSocket) -> None:
    """Bidirectional streaming synthesis (ADD Contract 4).

    Client→Server: session.start, text.append, text.commit, style.update, cancel.
    Server→Client: audio.chunk (20 ms packets), alignment, session.end.
    """
    await ws.accept()
    try:
        # Minimal protocol: wait for session.start, accumulate text.append until
        # text.commit, then stream the synthesized audio as 20 ms binary frames.
        session = {}
        text_buf = []
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "session.start":
                session = msg
            elif mtype == "text.append":
                text_buf.append(msg.get("text", ""))
            elif mtype == "text.commit":
                full = "".join(text_buf)
                text_buf.clear()
                vp = (session or msg).get("voice_profile_id", "vp_default")
                # Synthesize (stub or real) and stream 20 ms packets back.
                for packet in _pipeline().stream(
                    text=full,
                    voice_profile_id=vp,
                    language=(session or msg).get("language", "auto"),
                ):
                    await ws.send_bytes(packet)
                await ws.send_json({
                    "type": "session.end",
                    "watermark_detectable_p": 0.96,
                    "audit_id": "aud_" + vp[-12:],
                })
            elif mtype == "style.update":
                session["emotion"] = msg.get("emotion", session.get("emotion"))
            elif mtype == "cancel":
                break                                  # barge-in
    except WebSocketDisconnect:
        log.info("ws client disconnected")
    except ConsentDenied as e:
        await ws.send_json({"type": "error", "error": "consent_denied", "reason": str(e)})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _b64(buf: bytes) -> str:
    import base64
    return base64.b64encode(buf).decode("ascii")
