# Simple memory agent: heuristic to save facts like 'I am', 'I live in', etc.
import uuid
from sentence_transformers import SentenceTransformer
import os

EMB_MODEL = 'all-MiniLM-L6-v2'
model = SentenceTransformer(EMB_MODEL)
STORE_DIR = 'backend/rag/vector_store'

def should_save(text):
    triggers = ['i am', 'i was', 'my ', 'i have', 'i live', 'born', 'work at', 'job', 'favorite', 'favourite', 'love']
    t = text.lower()
    return any(tr in t for tr in triggers)

def summarize(text):
    s = text.strip()
    return s if len(s) < 300 else s[:300]

def save_memory_if_needed(user_id, text):
    if not should_save(text):
        return {'saved': False}
    summary = summarize(text)
    # append to memories file
    os.makedirs(STORE_DIR, exist_ok=True)
    mem_file = os.path.join(STORE_DIR, 'memories.txt')
    with open(mem_file, 'a', encoding='utf-8') as f:
        f.write(f"{user_id}\t{summary}\n")
    return {'saved': True, 'summary': summary}
