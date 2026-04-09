import subprocess
import ollama
from utils.config import MODEL_NAME


class LLMClient:

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    def ask_stream(self, prompt: str):
        stream = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        full_response = ""

        for chunk in stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token

        return full_response
