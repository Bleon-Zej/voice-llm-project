import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from math import ceil

class Recorder:
  
    def __init__(self, fs=44100, silent_counter=0, recorded=[]):
        self.fs = fs  # sample rate
        self.silent_counter = silent_counter
        self.recorded = recorded
        print(f"[Recorder] Initialisiert mit fs={self.fs}")

    def callback(self, indata, frames, time, status, vad, threshold, max_silence, chunk_duration):
        # indata kommt als 2D-Array (frames x channels)
        chunk = indata.flatten()
        self.recorded.append(chunk)

        if vad.is_speech(chunk):
            self.silent_counter = 0
            print("[Recorder] Sprache erkannt")
        else:
            self.silent_counter += 1
            print(f"[Recorder] Stille erkannt ({self.silent_counter * chunk_duration:.2f}s)")

        # Stream stoppen, wenn max_silence überschritten
        if self.silent_counter * chunk_duration >= max_silence:
            raise sd.CallbackStop()
  
    def record_until_silence(self, vad, threshold=0.02, max_silence=1.0, chunk_duration=0.03):
        self.recorded = []
        self.silent_counter = 0

        blocksize = int(chunk_duration * self.fs)

        print(f"[Recorder] Starte Aufnahme bis zu {max_silence}s Stille, threshold={threshold}")
        with sd.InputStream(
            samplerate=self.fs,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=lambda indata, frames, time, status: self.callback(
                indata, frames, time, status, vad, threshold, max_silence, chunk_duration
            )
        ):
            sd.sleep(int(1000 * (max_silence + 10)))  # warte, bis Callback stoppt

        audio = np.concatenate(self.recorded)
        print(f"[Recorder] Aufnahme beendet, Gesamtlänge={len(audio)/self.fs:.2f}s")
        return audio

  
    def save_wav(self, audio, filename="input.wav"):
        # Float32 -> Int16
        audio_int16 = (audio * 32767).astype("int16")
        write(filename, self.fs, audio_int16)
        print(f"[Recorder] WAV-Datei gespeichert: {filename}, Länge={len(audio)/self.fs:.2f}s")
        return filename