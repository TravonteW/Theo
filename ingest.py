#!/usr/bin/env python3
"""
Ingest PDFs and prepare them for search and question answering.
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from pypdf import PdfReader
from tqdm import tqdm


def find_pdf_files(pdf_dir: str = "./PDF Files") -> List[Path]:
    """Find all PDF files in the given directory recursively."""
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        raise FileNotFoundError(f"Directory {pdf_dir} not found")
    
    pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
    return pdf_files


def extract_text_from_pdf(pdf_path: Path) -> List[str]:
    """Extract text from each page of the PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text.strip():  # Only add non-empty pages
            pages.append(text)
    
    return pages


def split_text_into_chunks(
    text: str, 
    encoder_name: str = "cl100k_base", 
    max_tokens: int = 500, 
    overlap: int = 15
) -> List[str]:
    """Split text into chunks with a maximum token count and slight overlap."""
    encoder = tiktoken.get_encoding(encoder_name)
    tokens = encoder.encode(text)
    
    chunks = []
    i = 0
    
    while i < len(tokens):
        # Get chunk with max_tokens
        chunk_end = min(i + max_tokens, len(tokens))
        chunk_tokens = tokens[i:chunk_end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move to next chunk with overlap
        i += max_tokens - overlap
    
    return chunks


def process_pdfs(pdf_dir: str = "./PDF Files") -> List[Dict]:
    """Process all PDFs in the directory and return a list of chunks with metadata."""
    pdf_files = find_pdf_files(pdf_dir)
    all_chunks = []
    chunk_id = 0
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_name = pdf_path.name
        pages = extract_text_from_pdf(pdf_path)
        
        for page_num, page_text in enumerate(pages):
            chunks = split_text_into_chunks(page_text)
            
            for chunk in chunks:
                chunk_data = {
                    "id": chunk_id,
                    "source_pdf": pdf_name,
                    "page_num": page_num + 1,  # 1-indexed for user-friendliness
                    "text": chunk
                }
                all_chunks.append(chunk_data)
                chunk_id += 1
    
    return all_chunks


def save_chunks(chunks: List[Dict], output_file: str = "chunks.pkl") -> None:
    """Save the chunks to a pickle file."""
    with open(output_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {output_file}")


def load_chunks(input_file: str = "chunks.pkl") -> Optional[List[Dict]]:
    """Load chunks from a pickle file if it exists."""
    if os.path.exists(input_file):
        with open(input_file, "rb") as f:
            return pickle.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF files for search and QA")
    parser.add_argument(
        "--rebuild", 
        action="store_true", 
        help="Force rebuilding the chunks even if they already exist"
    )
    args = parser.parse_args()
    
    if not args.rebuild:
        chunks = load_chunks()
        if chunks:
            print(f"Loaded {len(chunks)} existing chunks. Use --rebuild to reprocess PDFs.")
            return
    
    chunks = process_pdfs()
    save_chunks(chunks)


if __name__ == "__main__":
    main()