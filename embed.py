#!/usr/bin/env python3
"""
Generate embeddings for PDF chunks and build a FAISS index for semantic search.
"""

import os
import json
import pickle
import argparse
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
from tqdm import tqdm
import faiss
from openai import OpenAI


def _hydrate_openai_env_from_streamlit_secrets() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return

    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        try:
            openai_section = st.secrets.get("openai") or {}
            api_key = openai_section.get("api_key")
        except Exception:
            api_key = None

    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)


def load_chunks(input_file: str = "chunks.pkl") -> List[Dict]:
    """Load chunks from a pickle file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Chunks file {input_file} not found. Run ingest.py first.")
    
    with open(input_file, "rb") as f:
        return pickle.load(f)


def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Generate embeddings for texts using OpenAI's embedding model."""
    _hydrate_openai_env_from_streamlit_secrets()
    client = OpenAI()
    embeddings = []
    
    # Process in batches to avoid rate limits and improve speed
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch_texts
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return embeddings


def create_citation_map(chunks: List[Dict]) -> Dict[int, Dict]:
    """Create a mapping from chunk ID to citation information."""
    citation_map = {}
    for chunk in chunks:
        chunk_id = chunk["id"]
        citation_map[chunk_id] = {
            "source_pdf": chunk["source_pdf"],
            "page_num": chunk["page_num"],
            "text": chunk["text"]
        }
    return citation_map


def save_citation_map(citation_map: Dict[int, Dict], output_file: str = "citations.json") -> None:
    """Save the citation map to a JSON file."""
    # Convert integer keys to strings for JSON compatibility
    serializable_map = {str(k): v for k, v in citation_map.items()}
    with open(output_file, "w") as f:
        json.dump(serializable_map, f, indent=2)
    print(f"Saved citation map to {output_file}")


def build_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    """Build a FAISS index from embeddings."""
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Get dimensionality of embeddings
    dimension = embeddings_array.shape[1]
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings_array)
    
    return index


def save_faiss_index(index: faiss.Index, output_file: str = "index.faiss") -> None:
    """Save the FAISS index to disk."""
    faiss.write_index(index, output_file)
    print(f"Saved FAISS index to {output_file}")


def build_index(force_rebuild: bool = False) -> None:
    """Build the embedding index and citation map."""
    # Check if index already exists
    if not force_rebuild and os.path.exists("index.faiss") and os.path.exists("citations.json"):
        print("Index and citation map already exist. Use --rebuild to regenerate.")
        return
    
    # Load chunks
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from chunks.pkl")
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    embeddings = get_embeddings(texts)
    
    # Build and save FAISS index
    index = build_faiss_index(embeddings)
    save_faiss_index(index)
    
    # Create and save citation map
    citation_map = create_citation_map(chunks)
    save_citation_map(citation_map)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and build search index")
    parser.add_argument(
        "--rebuild", 
        action="store_true", 
        help="Force rebuilding the index even if it already exists"
    )
    args = parser.parse_args()
    
    build_index(force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()
