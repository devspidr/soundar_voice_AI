# backend/agents/voice_agent.py
import os

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from TTS.api import TTS
except Exception:
    TTS = None

import soundfile as sf
import numpy as np

class VoiceAgent:
    def __init__(self):
        self.stt_model = None
        self.tts_model = None

    def transcribe(self, audio_path):
        if WhisperModel is None:
            return "(server STT not available - use browser STT)"
        if self.stt_model is None:
            self.stt_model = WhisperModel("small")
        segments, info = self.stt_model.transcribe(audio_path)
        text = "".join([s.text for s in segments])
        return text

    def speak(self, text, out_path="backend/out_response.wav"):
        if TTS is None:
            sr = 22050
            data = np.zeros(int(0.3 * sr), dtype="float32")
            sf.write(out_path, data, sr)
            return out_path

        if self.tts_model is None:
            models = TTS.list_models()
            self.tts_model = TTS(models[0])

        wav = self.tts_model.tts(text)
        sf.write(out_path, wav, 22050)
        return out_path
