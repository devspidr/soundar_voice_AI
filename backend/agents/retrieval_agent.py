# backend/agents/retrieval_agent.py
"""
Minimal safe RetrievalAgent (stub) — avoids chromadb so backend can start.
Replace with a proper chroma-backed implementation later.
"""
import logging
from pathlib import Path

log = logging.getLogger(__name__)

class RetrievalAgent:
    def __init__(self, collection_name="soundar_memory"):
        log.info("Using stub RetrievalAgent (no chromadb).")
        self.collection = None

    def retrieve(self, query: str, k: int = 3):
        if not query or not query.strip():
            return ""
        # Return empty string (no retrieved memory). Keeps behavior stable.
        return ""
