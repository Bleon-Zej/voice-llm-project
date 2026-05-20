# Voice Assistant

An offline voice assistant for Apple Silicon. Speak — it listens, thinks, and talks back. No cloud, no API keys.

![Demo](assets/demo.gif)

---

## How it works

```
Microphone → STT (faster-whisper) → LLM (Ollama) → TTS (Kokoro ONNX)
                                         ↕
                                      Memory
```

Speech is captured and segmented by silence. faster-whisper transcribes locally, Ollama generates a response, Kokoro synthesizes it back to speech. Conversation history is compressed automatically when it grows too long.

---

## Requirements

- macOS with Apple Silicon
- Python 3.12
- [Ollama](https://ollama.ai) running locally
- Kokoro ONNX model files (see below)

---

## Setup

```bash
git clone https://github.com/Bleon-Zej/voice-llm-project.git
cd voice-llm-project

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3
```

Download `kokoro-v1.0.onnx` and `voices.bin` from [Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M) and place them in `data/models/`.

---

## Usage

```bash
python main.py
```

Speak. Pause. It responds.

---

## Configuration

All settings in `utils/config.py`:

| Parameter                | Default   | Description                          |
| ------------------------ | --------- | ------------------------------------ |
| `MODEL_NAME`             | `llama3`  | Ollama model                         |
| `WHISPER_MODEL_SIZE`     | `small`   | `tiny` / `small` / `medium`          |
| `TTS_VOICE`              | `am_onyx` | Kokoro voice                         |
| `ENERGY_THRESHOLD`       | `0.01`    | Mic sensitivity                      |
| `SILENCE_THRESHOLD_TIME` | `2.0`     | Seconds of silence before processing |
| `MAX_MESSAGES`           | `20`      | Turns before memory summarization    |

---

## Roadmap

- [x] Voice activity detection
- [x] Local speech-to-text
- [x] Local LLM inference
- [x] Local text-to-speech
- [x] Persistent memory with summarization
- [ ] Async pipeline (overlap LLM + TTS)
- [ ] Wake-word detection
- [ ] Voice cloning with F5-TTS

---

## License

MIT
