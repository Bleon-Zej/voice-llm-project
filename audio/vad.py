import numpy as np


class VAD:

    def __init__(self, speech_threshold=0.01, silence_threshold=0.015):
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.is_speaking = False

    def is_speech(self, chunk):
        energy = np.sqrt(np.mean(chunk**2))

        # Hysterese
        if self.is_speaking:
            if energy < self.silence_threshold:
                self.is_speaking = False
        else:
            if energy > self.speech_threshold:
                self.is_speaking = True

        return self.is_speaking
