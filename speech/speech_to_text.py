# speech/speech_to_text.py

import numpy as np
from faster_whisper import WhisperModel
from numpy.typing import NDArray
from utils.config import Config


class STT:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = WhisperModel(config.WHISPER_MODEL_SIZE, compute_type="int8")

    def transcribe_audio(self, audio_array: NDArray[np.float32]) -> str:
        print("\nTranscribing...")

        # Normalisieren
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)

        segments, info = self.model.transcribe(
            audio_array,
            beam_size=self.config.WHISPER_BEAM_SIZE,
            temperature=0,
            language=self.config.WHISPER_LANGUAGE,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": self.config.WHISPER_VAD_MIN_SILENCE_MS
            },
        )

        text: list[str] = []

        for segment in segments:
            text.append(segment.text.strip())

        return " ".join(text)
