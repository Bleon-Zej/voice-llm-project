# utils/config.py
from dataclasses import dataclass


@dataclass(
    frozen=True
)  # frozen=True macht die Config read-only (Schutz vor Änderungen)
class Config:
    # LLM
    MODEL_NAME: str = "llama3"
    # MODEL_NAME: str = "gemma4:e2b"

    # STT
    WHISPER_MODEL_SIZE: str = "small"
    WHISPER_LANGUAGE: str = "de"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_VAD_MIN_SILENCE_MS: int = 300

    # Memory
    MAX_MESSAGES: int = 20
    MESSAGES_TO_SUMMARIZE: int = 10
    MEMORY_FILE_PATH: str = "data/chat_history.json"

    # Recorder
    SAMPLE_RATE: int = 16000
    CHUNK_DURATION: float = 0.03

    # Models
    KOKORO_MODEL_PATH: str = "data/models/model.onnx"
    KOKORO_VOICES_PATH: str = "data/models/voices.bin"

    # Main / Buffer-Management
    SILENCE_THRESHOLD_TIME: float = 2.0
    MIN_AUDIO_TIME: float = 0.5
    ENERGY_THRESHOLD: float = 0.01
