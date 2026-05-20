# llm/llm_client.py
import asyncio
from ollama import AsyncClient
from utils.config import Config
from utils.text_utils import is_sentence
from typing import Any


class LLMClient:

    def __init__(self, config: Config) -> None:
        self.config = config

    async def generate(
        self, messages: list[dict[str, Any]], queue: asyncio.Queue, silent: bool = False
    ) -> None:
        buffer = ""
        try:
            stream = await AsyncClient().chat(
                model=self.config.MODEL_NAME,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                token = chunk["message"]["content"]
                if not silent:  # Nur ausgeben wenn nicht silent
                    print(token, end="", flush=True)
                buffer += token
                if is_sentence(buffer=buffer, next_token=token):
                    await queue.put(buffer.strip())
                    buffer = ""  # Buffer leeren

            if buffer.strip():
                await queue.put(buffer.strip())

            await queue.put(None)  # Ende Signal
        except Exception as e:
            print(f"Fehler bei LLM-Anfrage: {e}")
            await queue.put(f"Fehler: {e}")
            await queue.put(None)
