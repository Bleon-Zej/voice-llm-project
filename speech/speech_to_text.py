import numpy as np
from faster_whisper import WhisperModel

class STT:

  def __init__(self, model_size="small"):
        self.model = WhisperModel(model_size)
#https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_voicechatbot/2_setup/
  def transcribe_audio(self, filename="input.wav"):
    print("Transcribing...")
    segments, info = self.model.transcribe(filename)
    print("Detected language:", info.language)
    text = []
    for segment in segments: 
       print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}")
       text.append(segment.text.strip())
    full_text = " ".join(text)
    print("Done\n")
    return full_text