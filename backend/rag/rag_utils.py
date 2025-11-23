# backend/rag/rag_utils.py
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "backend/rag/chroma_db"
COLLECTION_NAME = "soundar_profile"

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return collection

def get_embedding_model():
    # return an instantiated SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model
