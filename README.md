# Soundar Voice AI — Fully Self-Hosted (FAISS + HuggingFace LLM) — MVP

This repository is an **MVP** for a fully local, self-hosted, voice-first assistant
that speaks as *Soundar*. It uses:

- Browser Web Speech API for STT & TTS (frontend)
- Backend FastAPI orchestrator
- FAISS vector store for RAG (profile-based)
- SentenceTransformers for embeddings
- HuggingFace Transformers local model for generation (default: gpt2; replaceable)
- faster-whisper for optional server-side STT (optional)
- Coqui TTS (optional) for server-side TTS

**Important:** This project aims to be runnable locally and free. The default generation
model is `gpt2` (small) so it runs on most machines. For higher quality, replace the
model with a larger HF instruct model (Mistral, Llama, etc.) — see notes below.

----
## Contents
- `backend/` - FastAPI backend + agents + RAG utilities
- `frontend/` - simple HTML/JS voice frontend (uses browser STT & TTS)
- `data/` - place your `profile_full.txt` here (Soundar profile)
- `requirements.txt` - Python dependencies
- `run_backend.sh` - script to run backend
- `run_frontend.sh` - simple static server for frontend

----
## Quick start (Linux / macOS)

1. Clone or unzip this repo and cd into it.
2. Create Python venv and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Edit `data/profile_full.txt` and paste your full Soundar profile (this is the RAG doc).
4. Build the FAISS index:
   ```bash
   python backend/rag/build_faiss.py
   ```
5. Start the backend:
   ```bash
   ./run_backend.sh
   ```
   The server will start at `http://localhost:8000`.
6. Serve the frontend:
   ```bash
   ./run_frontend.sh
   ```
   Open `http://localhost:8001` in Chrome/Edge.
7. Click **Record** and speak. The browser will do STT and send a transcript to the backend.
   The backend will:
     - retrieve relevant profile chunks from FAISS,
     - construct a prompt and call the local HF model,
     - return text which the browser will speak using SpeechSynthesis.

----
## Notes & Tips
- **Improve generation quality:** Replace the HF model in `backend/agents/reasoning_agent.py`
  by downloading a stronger local model and updating `HF_MODEL` env var or code.
- **Server-side STT/TTS:** This repo uses browser STT/TTS for simplicity. If you want
  server-side, enable `faster-whisper` and `TTS` usage in `voice_agent.py`.
- **Memory:** The repo includes a simple FAISS profile store. To add long-term
  conversational memory, create a `memory_store` FAISS table and store summary vectors.
- **Privacy:** Everything runs locally. Keep `data/profile_full.txt` safe.

----
## Troubleshooting
- If sentence-transformers or torch fail to install, check CPU/GPU compatibility and
  install the appropriate `torch` wheel from https://pytorch.org.
- If the frontend microphone fails, allow microphone permissions in the browser.

----
If you want, I can:
- swap gpt2 to a specific HF model and add an automatic download script,
- include Whisper-based server STT & Coqui TTS examples,
- or prepare a Dockerfile for easier deployment.
