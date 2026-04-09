import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import queue


class Recorder:

    def __init__(self, fs=16000, chunk_duration=0.03):
        self.fs = fs
        self.chunk_duration = chunk_duration
        self.blocksize = int(fs * chunk_duration)
        self.audio_queue = queue.Queue()

    def callback(self, indata, frames, time, status):
        chunk = indata.flatten()

        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def stream_audio(self):
        with sd.InputStream(
            samplerate=self.fs,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.callback,
        ):
            while True:
                yield self.audio_queue.get()
