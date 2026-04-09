import numpy as np
from faster_whisper import WhisperModel


class STT:

    def __init__(self, model_size="small"):
        self.model = WhisperModel(model_size, compute_type="int8")

    # https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_voicechatbot/2_setup/
    def transcribe_audio(self, audio_array):
        print("Transcribing...")

        # Normalisieren
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)

        segments, info = self.model.transcribe(
            audio_array,
            beam_size=5,  # bessere Genauigkeit
            temperature=0,
            language="de",
        )

        text = []

        for segment in segments:
            text.append(segment.text.strip())

        return " ".join(text)
