"""M6 — Neural Vocoder & Streaming (sub-300 ms latency).

Decomposition (per Part2.md / VajraVoice-ADD-Part1):

    MelSpectrogram
      → 6.1 Vocoder Core              (Vocos ConvNeXt, inverse STFT)
      → 6.2 Long-Form Stabilization   (chunked inference + 50 ms crossfade)
      → 6.3 Streaming Buffer          (FastAPI WS + torch.cuda.Stream + PyAV)

The vocoding swap is the second load-bearing Phase-2 latency decision:

  Adversarial vocoders (HiFi-GAN lineage) reconstruct the waveform in the
  time domain — upsampling through transposed convolutions up to 24,000
  samples/sec. Vocos instead predicts the magnitude AND phase of the STFT
  and performs a SINGLE inverse STFT. The waveform is produced in one pass
  rather than progressively upsampled, removing the dominant cost of the
  vocoder stage while improving high-frequency phase reconstruction.
  Reported speedups are order-of-magnitude at comparable quality.

The third decision — FlashAttention-2 + torch.compile — is applied across
every transformer stack in the pipeline; it reduces the constant factor
rather than the number of passes, and is required for the budget to close.

Streaming detail: once Module 5 has produced ~300 ms of mel frames, the
vocoder emits audio while generation continues; packets are framed at 20 ms
and shipped over a dedicated CUDA stream. The listener hears the opening of
the utterance while its remainder is still being generated — that's what
makes the perceptual latency target achievable without making the model
itself faster.

Licensing: Vocos MIT. PyAV (FFmpeg wrapper) GPL/LGPL — handled at the
packaging boundary, not in this file.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..contracts import MelSpectrogram
from .base import VocoderModule


# 24 kHz mono s16le, 20 ms packets: 24000 * 0.020 * 2 = 960 bytes per packet.
_BYTES_PER_PACKET = 960
# Frame rate = hop_ms / 1000. At 10 ms hop, 20 ms = 2 frames per packet.
_PACKET_FRAMES = 2


class Part2VocoderModule(VocoderModule):
    """Concrete M6 per Part2.md §Module 6."""

    component_key = "Part2VocoderModule"
    phase = "phase2"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vocos = None        # Vocos (ConvNeXt backbone, iSTFT)
        self._chunk_state = None  # persistent hidden state across chunks
        self._stream = None       # dedicated torch.cuda.Stream for audio packets

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "Part2VocoderModule":
        if self._loaded:
            return self
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Part2VocoderModule needs the heavy ML stack: "
                "`pip install -e '.[models]'`. Missing: " + str(e.name)
            ) from e

        # Assert the mel contract at load time: disagreement between the
        # generator's mel configuration and the vocoder's training config is
        # the single most common cause of degraded output in a composed
        # pipeline — prevented by assertion, not detected by listening.
        self._assert_mel_contract()
        self._vocos = self._load_vocos()
        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, mel: MelSpectrogram, stream: bool = False) -> Any:
        self.load()
        if not stream:
            # Batch path: synthesize full waveform in one pass.
            return self._synthesize_full(mel)
        # Stream path: yield 20 ms packets as they're produced.
        return self._synthesize_streaming(mel)

    # ------------------------------------------------------------------
    # synthesis paths
    # ------------------------------------------------------------------

    def _synthesize_full(self, mel: MelSpectrogram) -> bytes:
        """Vocos: STFT magnitude+phase prediction, single inverse STFT."""
        # Vocos predict + inverse STFT (placeholder zeros for the demo path).
        T = mel.mel.shape[0] if hasattr(mel.mel, "shape") else len(mel.mel)
        n_samples = int(T * mel.sample_rate * (mel.hop_ms / 1000.0))
        return b"\x00\x00" * n_samples

    def _synthesize_streaming(self, mel: MelSpectrogram) -> Iterator[bytes]:
        """Chunked inference: emit 20 ms packets as mel frames become available.

        Once ~300 ms of mel is buffered (the perceptual threshold), the vocoder
        starts emitting while generation continues. Crossfade 50 ms between
        chunks to suppress volume drift and timbre migration.
        """
        T = mel.mel.shape[0] if hasattr(mel.mel, "shape") else len(mel.mel)
        for i in range(0, T, _PACKET_FRAMES):
            chunk_mel = self._slice_mel(mel, i, i + _PACKET_FRAMES)
            packet = self._synthesize_full(chunk_mel)
            # Apply 50 ms crossfade at chunk boundaries to suppress drift.
            packet = self._crossfade(packet, prev=self._chunk_state)
            self._chunk_state = packet
            # Pad/truncate to a clean 20 ms packet (960 bytes).
            yield packet[:_BYTES_PER_PACKET].ljust(_BYTES_PER_PACKET, b"\x00")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _assert_mel_contract(self) -> None:
        """Assert Vocos's training mel params match M5's output params."""
        # Real impl: load Vocos config; assert n_mels=80, sample_rate=24000,
        # hop=240, win=400, f_min, f_max, log-mag normalization all match the
        # M5 mel head. Mismatch here ⇒ silent quality degradation.
        pass

    @staticmethod
    def _slice_mel(mel: MelSpectrogram, start: int, end: int) -> MelSpectrogram:
        return MelSpectrogram(
            mel=mel.mel[start:end] if hasattr(mel.mel, "__getitem__") else mel.mel,
            sample_rate=mel.sample_rate,
            hop_ms=mel.hop_ms,
            window_ms=mel.window_ms,
            n_mels=mel.n_mels,
        )

    @staticmethod
    def _crossfade(packet: bytes, prev: "bytes | None", fade_ms: int = 50) -> bytes:
        """Equal-power crossfade at chunk boundaries.

        Suppresses the slow accumulation of state error across chunk
        boundaries, which otherwise manifests as loudness drift and timbre
        migration over multi-minute output.
        """
        # Real impl: 50 ms overlap-add with equal-power weights. Placeholder
        # returns the packet unchanged for the demo path.
        return packet

    # ------------------------------------------------------------------
    # weight loaders
    # ------------------------------------------------------------------

    def _load_vocos(self):  # pragma: no cover
        # from vocos import Vocos
        # return Vocos.from_pretrained("charactr/vocos-mel-24khz")
        return None
