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
        self,
        messages: list[dict[str, Any]],
        queue: asyncio.Queue | None = None,
        silent: bool = False,
    ) -> str:
        buffer = ""
        if silent:
            model = self.config.MODEL_NAME_SUMMARY
        else:
            model = self.config.MODEL_NAME

        try:
            stream = await AsyncClient().chat(
                model=model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                token = chunk["message"]["content"]
                buffer += token
                if not silent:  # Nur ausgeben wenn nicht silent
                    print(token, end="", flush=True)
                    if is_sentence(buffer=buffer, next_token=token):
                        await queue.put(buffer.strip())
                        buffer = ""  # Buffer leeren

            if not silent:
                if buffer.strip():
                    await queue.put(buffer.strip())
                await queue.put(None)

            return buffer
        except Exception as e:
            print(f"Fehler bei LLM-Anfrage: {e}")
            await queue.put(f"Fehler: {e}")
            await queue.put(None)
            return buffer
