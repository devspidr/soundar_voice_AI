# backend/rag/build_chroma.py

import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def main():
    # Get base directory -> backend/
    base_dir = Path(__file__).resolve().parent.parent

    # Path to your profile text
    profile_path = base_dir / "data" / "soundar_profile.txt"

    if not profile_path.exists():
        print("❌ ERROR: soundar_profile.txt not found at:", profile_path)
        return

    print("Loading:", profile_path)
    text = profile_path.read_text(encoding="utf-8")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    print(f"Total chunks created: {len(chunks)}")

    # Embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks).tolist()

    # Path where backend reads the Chroma DB
    chroma_dir = base_dir / "chroma_db"
    chroma_dir.mkdir(exist_ok=True)

    print("Saving ChromaDB to:", chroma_dir)

    # Create client
    client = chromadb.Client(Settings(persist_directory=str(chroma_dir)))

    # GET SAME COLLECTION NAME AS BACKEND
    collection = client.get_or_create_collection(
        name="soundar_memory",
        metadata={"hnsw:space": "cosine"}
    )

    # Insert into DB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    print("✅ ChromaDB build complete!")
    print("Location:", chroma_dir)


if __name__ == "__main__":
    main()
