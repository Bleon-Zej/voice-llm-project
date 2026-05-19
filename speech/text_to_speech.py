from utils.config import Config
from kokoro_onnx import Kokoro
import sounddevice as sd


class TTS:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.kokoro = Kokoro(
            self.config.KOKORO_MODEL_PATH, self.config.KOKORO_VOICES_PATH
        )

    def speak(self, text: str) -> None:
        samples, sample_rate = self.kokoro.create(
            text=text,
            voice=self.config.TTS_VOICE,
            speed=self.config.TTS_SPEED,
            lang=self.config.TTS_LANG,
        )
        sd.play(data=samples, samplerate=sample_rate)
        sd.wait()
