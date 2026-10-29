"""
populate_kb.py — Run this ONCE to load domain knowledge into ChromaDB.
Usage: python populate_kb.py
"""

import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_into_chunks(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def load_knowledge_file(file_path: str) -> str:
    """Read a plain text knowledge base file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def populate_collection(
    collection: chromadb.Collection,
    file_path: str,
    source_label: str,
    chunk_size: int,
    overlap: int,
) -> int:
    """
    Load one knowledge base file into ChromaDB.
    Returns the number of chunks added.
    """
    text = load_knowledge_file(file_path)
    chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)

    documents = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{source_label}_chunk_{idx:04d}"
        documents.append(chunk)
        metadatas.append({"source": source_label, "chunk_index": idx})
        ids.append(chunk_id)

    # Add in batches of 100 to avoid memory issues
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[start : start + batch_size],
            metadatas=metadatas[start : start + batch_size],
            ids=ids[start : start + batch_size],
        )

    return len(chunks)


def main():
    config = load_config()

    chroma_path = config["chromadb"]["path"]
    collection_name = config["chromadb"]["collection_name"]
    embedding_model = config["embeddings"]["model"]
    chunk_size = config["embeddings"]["chunk_size"]
    overlap = config["embeddings"]["chunk_overlap"]

    kb_dir = os.path.join("data", "knowledge_base")

    knowledge_files = {
        "ISA-18.2-Alarms": os.path.join(kb_dir, "isa_18_2_alarms.txt"),
        "ISA-88-Modes": os.path.join(kb_dir, "isa_88_modes.txt"),
        "ISA-5.1-Tags": os.path.join(kb_dir, "isa_5_1_tags.txt"),
        "Control-Glossary": os.path.join(kb_dir, "control_glossary.txt"),
    }

    print("Initialising ChromaDB...")
    client = chromadb.PersistentClient(path=chroma_path)

    print(f"Loading embedding model: {embedding_model}")
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    # Delete existing collection to start fresh
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for source_label, file_path in knowledge_files.items():
        if not os.path.exists(file_path):
            print(f"  WARNING: File not found — {file_path}")
            continue
        count = populate_collection(collection, file_path, source_label, chunk_size, overlap)
        total_chunks += count
        print(f"  Loaded {count} chunks from {source_label}")

    print(f"\nDone. Total chunks in domain knowledge base: {total_chunks}")
    print(f"ChromaDB stored at: {os.path.abspath(chroma_path)}")
    print("\nYou can now run: streamlit run src/ui/app.py")


if __name__ == "__main__":
    main()
