# Build FAISS index from backend/rag/soundar_profile.txt
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

PROFILE = "backend/rag/soundar_profile.txt"
OUT_INDEX = "backend/rag/vector_store/faiss_index.bin"

os.makedirs(os.path.dirname(OUT_INDEX), exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

with open(PROFILE, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f.read().split('\n') if l.strip()]

if not lines:
    raise SystemExit('Please put your profile text into backend/rag/soundar_profile.txt')

embeddings = model.encode(lines)
embeddings = np.array(embeddings).astype('float32')
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
faiss.write_index(index, OUT_INDEX)
print('Built FAISS index with', len(lines), 'vectors ->', OUT_INDEX)
