from llm.llm_client import LLMClient
from audio.recorder import Recorder
from speech.speech_to_text import STT
from audio.vad import VAD
import numpy as np

llm = LLMClient()
stt = STT()
recorder = Recorder()
vad = VAD()

buffer = []
silent_time = 0.0

CHUNK_DURATION = 0.03
SILENCE_THRESHOLD_TIME = 2
MIN_AUDIO_TIME = 0.5

for chunk in recorder.stream_audio():

    is_speech = vad.is_speech(chunk)

    if is_speech:
        buffer.append(chunk)
        silent_time = 0.0

    else:
        silent_time += CHUNK_DURATION

        # erst verarbeiten wenn genug Sprache gesammelt
        if silent_time >= SILENCE_THRESHOLD_TIME:

            if len(buffer) > 0:

                audio = np.concatenate(buffer)

                # nur verarbeiten wenn genug Länge
                if len(audio) / recorder.fs >= MIN_AUDIO_TIME:
                    text = stt.transcribe_audio(audio)
                    print("YOU:", text)

                    response = llm.ask_stream(text)

                buffer.clear()
                silent_time = 0.0
