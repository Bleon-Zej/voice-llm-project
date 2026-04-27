# llm/llm_client.py

import ollama
from utils.config import Config
from typing import Any


class LLMClient:

    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, messages: list[dict[str, Any]], silent: bool = False) -> str:
        full_response = ""
        try:
            stream = ollama.chat(
                model=self.config.MODEL_NAME,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if not silent:  # Nur ausgeben wenn nicht silent
                    print(token, end="", flush=True)
                full_response += token
        except Exception as e:
            print(f"Fehler bei LLM-Anfrage: {e}")
            full_response = f"Fehler: {e}"

        return full_response
