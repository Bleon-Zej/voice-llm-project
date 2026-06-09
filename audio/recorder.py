import asyncio
import sounddevice as sd
import numpy as np
import queue
from typing import AsyncGenerator
from numpy.typing import NDArray
from utils.config import Config


class Recorder:
    def __init__(self, config: Config) -> None:
        self.fs: int = config.SAMPLE_RATE
        self.chunk_duration: float = config.CHUNK_DURATION
        self.blocksize: int = int(self.fs * self.chunk_duration)

        self.audio_queue: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=50)

        # CONTROL FLAGS
        self.paused: bool = False
        self.drop_until: float = 0.0

        self.stream = None

    def callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info,
        status,
    ) -> None:

        if self.paused:
            return

        chunk = indata.flatten()

        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    async def stream_audio(self) -> AsyncGenerator[NDArray[np.float32], None]:

        with sd.InputStream(
            samplerate=self.fs,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.callback,
        ) as stream:

            self.stream = stream

            while True:
                if self.paused:
                    await asyncio.sleep(0.01)
                    continue

                chunk = await asyncio.to_thread(self.audio_queue.get)
                yield chunk

    def pause(self) -> None:
        self.paused = True
        self._flush_queue()

    def resume(self) -> None:
        self.paused = False

    def _flush_queue(self) -> None:
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass
