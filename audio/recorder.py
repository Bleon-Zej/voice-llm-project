# audio/recorder.py

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import queue
from typing import Generator
from numpy.typing import NDArray
from utils.config import Config


class Recorder:

    def __init__(self, config: Config) -> None:
        self.fs: int = config.SAMPLE_RATE
        self.chunk_duration: float = config.CHUNK_DURATION
        self.blocksize: int = int(self.fs * self.chunk_duration)
        self.audio_queue: queue.Queue[NDArray[np.float32]] = queue.Queue()

    def callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time: sd.CallbackFlags,
        status: sd.CallbackFlags,
    ) -> None:
        chunk = indata.flatten()

        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def stream_audio(self) -> Generator[NDArray[np.float32], None, None]:
        with sd.InputStream(
            samplerate=self.fs,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.callback,
        ):
            while True:
                yield self.audio_queue.get()
