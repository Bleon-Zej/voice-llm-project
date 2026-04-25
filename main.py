# main.py

from llm.llm_client import LLMClient
from audio.recorder import Recorder
from speech.speech_to_text import STT
from utils.config import Config
import numpy as np

# Config erstellen
config = Config()

# Komponenten initialisieren
llm = LLMClient(config=config)
stt = STT(config=config)
recorder = Recorder(config=config)

# Buffer-Management
buffer = []
silent_chunks = 0
SILENCE_CHUNKS_NEEDED = int(config.SILENCE_THRESHOLD_TIME / config.CHUNK_DURATION)

print("Listening...")

for chunk in recorder.stream_audio():
    energy = np.sqrt(np.mean(chunk**2))
    is_speech = energy > config.ENERGY_THRESHOLD

    if is_speech:
        buffer.append(chunk)
        silent_chunks = 0

    elif buffer:
        silent_chunks += 1

        if silent_chunks >= SILENCE_CHUNKS_NEEDED:
            audio = np.concatenate(buffer)

            if len(audio) / recorder.fs >= config.MIN_AUDIO_TIME:
                text = stt.transcribe_audio(audio)

                if text.strip():
                    print(f"\nYOU: {text}")
                    response = llm.generate(messages=text)
                    print()

            buffer.clear()
            silent_chunks = 0
