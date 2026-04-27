# memory/memory_manager.py
from datetime import datetime
import json
import os
from utils.config import Config
from memory.models import Message


class MemoryManager:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.file_path = config.MEMORY_FILE_PATH
        self.messages = []
        self.summary = ""
        self.load()

    def add_message(self, role: str, content: str) -> None:
        timestamp = datetime.now()
        message = Message(role=role, content=content, timestamp=timestamp)
        self.messages.append(message)

    def get_context_for_llm(self) -> list[dict[str, str]]:
        context = []
        # später systempromts
        if self.summary:
            context.append({"role": "system", "content": self.summary})
        for msg in self.messages:
            message = msg.to_ollama_format()
            context.append(message)
        return context

    def needs_optimization(self) -> bool:
        return len(self.messages) >= self.config.MAX_MESSAGES

    def get_data_for_summarization(self) -> str:
        # Nur die ältesten Nachrichten nehmen
        messages_to_summarize = self.messages[: Config.MESSAGES_TO_SUMMARIZE]
        text = "\n".join(f"{msg.role}: {msg.content}" for msg in messages_to_summarize)
        return f"Alte Zusammenfassung: {self.summary}\n\nNeue Nachrichten:\n{text}"

    def apply_summarization(self, new_summary: str, count_to_remove: int) -> None:
        self.summary = new_summary
        self.messages = self.messages[count_to_remove:]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        data = {
            "summary": self.summary,
            "messages": [msg.to_dict() for msg in self.messages],
        }

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.summary = data.get("summary", "")
            self.messages = [Message.from_dict(msg) for msg in data.get("messages", [])]

        except (json.JSONDecodeError, IOError) as e:
            print(f"Fehler beim Laden des Chats: {e}")
            # Bei Fehler einfach mit leerem Chat starten
            self.summary = ""
            self.messages = []

    def clear(self) -> None:
        self.messages = []
        self.summary = ""

        if os.path.exists(self.file_path):
            os.remove(self.file_path)
