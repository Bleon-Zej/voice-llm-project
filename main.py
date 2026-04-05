from llm.llm_client import LLMClient
from audio.recorder import Recorder
from speech.speech_to_text import STT
from audio.vad import VAD

llm = LLMClient()
stt = STT()

print("start")
rec = Recorder()
vad = VAD(threshold=0.02)
audio = rec.record_until_silence(vad=vad)
input = rec.save_wav(audio=audio)
prompt = stt.transcribe_audio(input)
response = llm.ask(prompt)
print(response)
print("Fertig")


  #audio_file = recorder.record_audio()
  #promt = speech_to_text.transcribe_audio(audio_file)
  #response = llm.ask(promt)
