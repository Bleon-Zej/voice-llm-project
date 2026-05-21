# main.py
from assistant.assistant import Assistant
import asyncio

if __name__ == "__main__":
    asyncio.run(Assistant().run())
