import ollama
from utils.config import MODEL_NAME


class LLMClient:

    def __init__(self, model_name: str = MODEL_NAME, messages=None) -> None:
        self.model_name: str = model_name
        self.messages = messages if messages is not None else []

    def ask(self, prompt: str) -> str:
        full_response = ""
        try:
            self.messages.append({"role": "user", "content": prompt})
            stream = ollama.chat(
                model=self.model_name,
                messages=self.messages,
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_response += token
        except Exception as e:
            print(f"Fehler: {e}")
            full_response = f"Fehler: {e}"

        assistant_msg = {"role": "assistant", "content": full_response}
        self.messages.append(assistant_msg)
        return full_response
