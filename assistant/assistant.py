# assistant.py

import numpy as np
import asyncio
from llm.llm_client import LLMClient
from audio.recorder import Recorder
from audio.audio_buffer import AudioBuffer
from speech.speech_to_text import STT
from memory.memory_manager import MemoryManager
from utils.config import Config
from speech.text_to_speech import TTS


class Assistant:

    def __init__(self) -> None:
        self.config = Config()
        self.queue = asyncio.Queue()
        self.recorder = Recorder(config=self.config)
        self.buffer = AudioBuffer(config=self.config)
        self.stt = STT(config=self.config)
        self.llm = LLMClient(config=self.config)
        self.memory = MemoryManager(config=self.config)
        self.tts = TTS(config=self.config)

    async def run(self) -> None:

        print("Listening...")

        async for chunk in self.recorder.stream_audio():
            audio = self.buffer.process(chunk)

            if audio is not None:
                await self._process_utterance(audio=audio)

    async def _process_utterance(self, audio: np.ndarray) -> None:
        if len(audio) / self.recorder.fs >= self.config.MIN_AUDIO_TIME:
            text = await asyncio.to_thread(self.stt.transcribe_audio, audio)

            if text.strip():
                print(f"\nYOU: {text}")

                self.memory.add_message(role="user", content=text)
                self.recorder.pause()
                context = self.memory.get_context_for_llm()
                task = await asyncio.gather(
                    self.llm.generate(messages=context, queue=self.queue),
                    self.tts.consume(self.queue),
                )
                response = task[0]
                await asyncio.sleep(0.3)  # kurzer Drain (wichtig!)
                self.recorder.resume()
                self.memory.add_message(role="assistant", content=response)
                self.memory.save()

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

                    await self._summarize(
                        summary_massages=summary_messages, silent=True
                    )
                print()

    async def _summarize(self, summary_massages: str, silent: bool) -> None:
        try:
            new_summary = await self.llm.generate(
                messages=summary_massages, silent=silent
            )
            self.memory.apply_summarization(
                new_summary=new_summary,
                count_to_remove=self.config.MESSAGES_TO_SUMMARIZE,
            )
        except Exception as e:
            print(f"Summarization Fehler: {e}")
