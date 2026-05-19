import numpy as np
from llm.llm_client import LLMClient
from audio.recorder import Recorder
from speech.speech_to_text import STT
from memory.memory_manager import MemoryManager
from utils.config import Config
from speech.text_to_speech import TTS


class Assistant:

    def __init__(self) -> None:
        self.config = Config()
        self.recorder = Recorder(config=self.config)
        self.stt = STT(config=self.config)
        self.llm = LLMClient(config=self.config)
        self.memory = MemoryManager(config=self.config)
        self.tts = TTS(config=self.config)

    def run(self) -> None:
        buffer = []
        silent_chunks = 0
        SILENCE_CHUNKS_NEEDED = int(
            self.config.SILENCE_THRESHOLD_TIME / self.config.CHUNK_DURATION
        )

        print("Listening...")

        for chunk in self.recorder.stream_audio():
            energy = np.sqrt(np.mean(chunk**2))
            is_speech = energy > self.config.ENERGY_THRESHOLD

            if is_speech:
                buffer.append(chunk)
                silent_chunks = 0

            elif buffer:
                silent_chunks += 1

                if silent_chunks >= SILENCE_CHUNKS_NEEDED:
                    audio = np.concatenate(buffer)
                    self._process_utterance(audio)

                    buffer.clear()
                    silent_chunks = 0

    def _process_utterance(self, audio: np.ndarray) -> None:
        if len(audio) / self.recorder.fs >= self.config.MIN_AUDIO_TIME:
            text = self.stt.transcribe_audio(audio)

            if text.strip():
                print(f"\nYOU: {text}")

                self.memory.add_message(role="user", content=text)

                if self.memory.needs_optimization():
                    print("\n[Optimiere Memory...]")
                    summary_data = self.memory.get_data_for_summarization()

                    summary_messages = [
                        {
                            "role": "system",
                            "content": "Fasse den Chatverlauf als kompakte Stichpunkte zusammen. "
                            "Nur Fakten, Namen, Entscheidungen. "
                            "Integriere die alte Zusammenfassung. "
                            "Kein Smalltalk. Keine Einleitungen. Nur die Stichpunkte.",
                        },
                        {"role": "user", "content": summary_data},
                    ]
                    new_summary = self.llm.generate(
                        messages=summary_messages, silent=True
                    )

                    self.memory.apply_summarization(
                        new_summary=new_summary,
                        count_to_remove=self.config.MESSAGES_TO_SUMMARIZE,
                    )

                context = self.memory.get_context_for_llm()
                response = self.llm.generate(messages=context)
                self.memory.add_message(role="assistant", content=response)
                self.memory.save()
                self.tts.speak(response)
                print()
