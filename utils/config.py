# utils/config.py
from dataclasses import dataclass


@dataclass(
    frozen=True
)  # frozen=True macht die Config read-only (Schutz vor Änderungen)
class Config:
    # LLM
    MODEL_NAME: str = "llama3"

    # STT
    WHISPER_MODEL_SIZE: str = "small"
    WHISPER_LANGUAGE: str = "de"

    # Memory
    MAX_MESSAGES: int = 10
    MEMORY_FILE_PATH: str = "data/chat_history.json"

    # Recorder
    SAMPLE_RATE: int = 16000

    # Main / Buffer-Management
    SILENCE_THRESHOLD_TIME: float = 2.0
    MIN_AUDIO_TIME: float = 0.5
    ENERGY_THRESHOLD: float = 0.01
