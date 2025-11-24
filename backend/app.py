# # backend/app.py
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# try:
#     # For python -m backend.app
#     from .agents.personality_agent import PersonalityAgent
# except ImportError:
#     # For python app.py
#     from agents.personality_agent import PersonalityAgent

# from .agents.reasoning_agent import ReasoningAgent
# from .agents.retrieval_agent import RetrievalAgent


# from dotenv import load_dotenv
# load_dotenv()


# import sys
# from pathlib import Path

# if __package__ is None:
#     # allow running "python app.py"
#     ROOT = Path(__file__).resolve().parent
#     sys.path.insert(0, str(ROOT))



# app = FastAPI(title="Soundar Voice AI - Backend")

# # CORS for local dev (restrict in production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize agents (singletons)
# retriever = RetrievalAgent()
# personality = PersonalityAgent()
# reasoning = ReasoningAgent()

# class VoicePayload(BaseModel):
#     user_id: str
#     transcript: str
#     conversation_history: list = []

# @app.post("/api/voice_text")
# def voice_text(data: VoicePayload):
#     user_text = data.transcript or ""

#     # 1 — Retrieve context from Chroma
#     context = retriever.retrieve(user_text)

#     # 2 — Build user prompt and get the persona/system instruction
#     user_prompt, system_instructions = personality.apply_style(user_text, context)

#     # 3 — Generate final response using system instructions
#     answer = reasoning.generate_response(user_prompt, system_instructions)

#     return {"text": answer}


# @app.get("/health")
# def health():
#     return {"status": "ok"}









# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

try:
    # For python -m backend.app
    from .agents.personality_agent import PersonalityAgent
except ImportError:
    # For python app.py
    from agents.personality_agent import PersonalityAgent

from .agents.reasoning_agent import ReasoningAgent
from .agents.retrieval_agent import RetrievalAgent

from dotenv import load_dotenv
load_dotenv()

import sys

if __package__ is None:
    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="Soundar Voice AI - Backend")

# ---------------------------
# FRONTEND STATIC FILE SUPPORT
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Mount folder: http://127.0.0.1:8000/frontend/index.html
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# Redirect root → index.html
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/frontend/index.html")

# ---------------------------


# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
retriever = RetrievalAgent()
personality = PersonalityAgent()
reasoning = ReasoningAgent()

class VoicePayload(BaseModel):
    user_id: str
    transcript: str
    conversation_history: list = []

@app.post("/api/voice_text")
def voice_text(data: VoicePayload):
    user_text = data.transcript or ""

    context = retriever.retrieve(user_text)

    user_prompt, system_instructions = personality.apply_style(user_text, context)

    answer = reasoning.generate_response(user_prompt, system_instructions)

    return {"text": answer}

@app.get("/health")
def health():
    return {"status": "ok"}
