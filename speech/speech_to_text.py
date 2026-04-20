import numpy as np
from faster_whisper import WhisperModel
from numpy.typing import NDArray
from utils.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_MIN_SILENCE_MS,
)


class STT:

    def __init__(self, model_size: str = "small") -> None:
        self.model = WhisperModel(model_size, compute_type="int8")

    def transcribe_audio(self, audio_array: NDArray[np.float32]) -> str:
        print("\nTranscribing...")

        # Normalisieren
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)

        segments, info = self.model.transcribe(
            audio_array,
            beam_size=WHISPER_BEAM_SIZE,
            temperature=0,
            language=WHISPER_LANGUAGE,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS},
        )

        text: list[str] = []

        for segment in segments:
            text.append(segment.text.strip())

        return " ".join(text)
