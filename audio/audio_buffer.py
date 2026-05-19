import numpy as np
from utils.config import Config


class AudioBuffer:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.buffer = []
        self.silent_chunks = 0
        self.SILENCE_CHUNKS_NEEDED = int(
            self.config.SILENCE_THRESHOLD_TIME / self.config.CHUNK_DURATION
        )

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        energy = np.sqrt(np.mean(chunk**2))
        is_speech = energy > self.config.ENERGY_THRESHOLD

        if is_speech:
            self.buffer.append(chunk)
            self.silent_chunks = 0

        elif self.buffer:
            self.silent_chunks += 1

            if self.silent_chunks >= self.SILENCE_CHUNKS_NEEDED:
                audio = np.concatenate(self.buffer)
                self.buffer.clear()
                self.silent_chunks = 0
                return audio

        return None
