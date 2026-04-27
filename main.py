# main.py

import numpy as np
from llm.llm_client import LLMClient
from audio.recorder import Recorder
from speech.speech_to_text import STT
from memory.memory_manager import MemoryManager
from utils.config import Config

# Config erstellen
config = Config()

# Komponenten initialisieren
llm = LLMClient(config=config)
stt = STT(config=config)
recorder = Recorder(config=config)
memory = MemoryManager(config=config)

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

                    # 1. User-Nachricht im Memory speichern
                    memory.add_message(role="user", content=text)

                    # 2. Prüfen ob Optimierung nötig ist
                    if memory.needs_optimization():
                        print("\n[Optimiere Memory...]")
                        summary_data = memory.get_data_for_summarization()

                        # 3. Zusammenfassung von KI generieren lassen
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
                        new_summary = llm.generate(
                            messages=summary_messages, silent=True
                        )

                        # 4. Summary anwenden
                        memory.apply_summarization(
                            new_summary=new_summary,
                            count_to_remove=Config.MESSAGES_TO_SUMMARIZE,
                        )

                    # 3. Kontext für LLM holen
                    context = memory.get_context_for_llm()

                    # 4. Antwort generieren
                    response = llm.generate(messages=context)

                    # 5. Antwort im Memory speichern
                    memory.add_message(role="assistant", content=response)

                    # 6. Memory speichern (Persistenz)
                    memory.save()

                    print()

            buffer.clear()
            silent_chunks = 0


# TODO: .md file nutzen für system prompts und persönlichkeit
