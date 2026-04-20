from llm.llm_client import LLMClient
from audio.recorder import Recorder
from speech.speech_to_text import STT
from utils.config import SILENCE_THRESHOLD_TIME, MIN_AUDIO_TIME, ENERGY_THRESHOLD
import numpy as np

llm = LLMClient()
stt = STT()
recorder = Recorder()

buffer = []
silent_chunks = 0
SILENCE_CHUNKS_NEEDED = int(SILENCE_THRESHOLD_TIME / recorder.chunk_duration)

print("Listening...")

for chunk in recorder.stream_audio():
    energy = np.sqrt(np.mean(chunk**2))
    is_speech = energy > ENERGY_THRESHOLD

    if is_speech:
        buffer.append(chunk)
        silent_chunks = 0

    elif buffer:
        silent_chunks += 1

        if silent_chunks >= SILENCE_CHUNKS_NEEDED:
            audio = np.concatenate(buffer)

            if len(audio) / recorder.fs >= MIN_AUDIO_TIME:
                text = stt.transcribe_audio(audio)

                if text.strip():
                    print(f"\nYOU: {text}")
                    response = llm.ask(text)
                    print()

            buffer.clear()
            silent_chunks = 0
