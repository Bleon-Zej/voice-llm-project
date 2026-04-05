import subprocess
from utils.config import MODEL_NAME

class LLMClient:

  def __init__(self, model_name = MODEL_NAME):
    self.model_name = model_name

  def ask(self, prompt: str) -> str:
    try:
      result = subprocess.run(
          ["ollama", "run", self.model_name, prompt],
          capture_output=True,
          text=True,
          check=True
      )
      return result.stdout.strip()
    except subprocess.CalledProcessError as e:
      print("Error calling Model", e)
      return ""

