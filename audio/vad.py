import sounddevice as sd


class VAD:


  def __init__(self, threshold=0.5):
    self.threshold = threshold

  
#decides per audio Chunk if someone is speaking
  def is_speech(self, chunk) -> bool:
    return chunk.max() > self.threshold