from utils.config import Config
from kokoro_onnx import Kokoro  # TODO: ZU requirements hinzufügen
import sounddevice as sd


class TTS:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.kokoro = Kokoro(
            self.config.KOKORO_MODEL_PATH, self.config.KOKORO_VOICES_PATH
        )

    def speak(self, text: str) -> None:
        samples, sample_rate = self.kokoro.create(
            text=text, voice="am_onyx", speed=1.3, lang="de"
        )
        sd.play(data=samples, samplerate=sample_rate)
        sd.wait()
