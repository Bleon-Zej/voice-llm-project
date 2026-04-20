import ollama
from utils.config import MODEL_NAME


class LLMClient:

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name: str = model_name

    def ask(self, prompt: str) -> str:
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
