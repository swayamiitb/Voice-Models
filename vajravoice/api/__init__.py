"""VajraVoice HTTP + WebSocket API surface (FastAPI)."""

from .schemas import (
    ConsentChallenge,
    ConsentDeniedResponse,
    EnrollRequest,
    EnrollResponse,
    ReferenceAudio,
    SpeechRequest,
    SpeechResponse,
    StreamClientMessage,
    StreamServerMessage,
)

__all__ = [
    "ConsentChallenge", "ConsentDeniedResponse", "EnrollRequest", "EnrollResponse",
    "ReferenceAudio", "SpeechRequest", "SpeechResponse",
    "StreamClientMessage", "StreamServerMessage",
]
