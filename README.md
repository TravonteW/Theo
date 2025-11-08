# PDF Question Answering System

This application allows users to ask questions about a collection of PDF documents and receive answers with inline citations pointing back to the source material.

## Features

- Extracts and processes text from PDF files
- Chunks text into semantically meaningful segments
- Creates embeddings for efficient semantic search
- Retrieves relevant passages based on user questions
- Generates answers with citations to source documents
- Simple web interface for asking questions

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your-api-key
   ```

## Usage

1. Process your PDF files:
   ```
   python ingest.py
   ```
   This will scan the `./PDF Files` directory for PDFs, extract text, and save chunks to `chunks.pkl`.

2. Create embeddings and build search index:
   ```
   python embed.py
   ```
   This will generate embeddings for each text chunk and build a FAISS index.

3. Start the web application:
   ```
   python -m streamlit run app.py
   ```
   Then open your browser to the URL provided in the terminal.

4. Ask questions through the web interface or use the CLI:
   ```
   python ask.py
   ```

## Rebuilding the Index

If you add new PDFs or want to rebuild the index:
```
python ingest.py --rebuild
python embed.py --rebuild
```

## File Structure

- `ingest.py` - Processes PDF files into text chunks
- `embed.py` - Creates embeddings and builds search index
- `ask.py` - Handles question answering functionality
- `app.py` - Streamlit web interface
- `requirements.txt` - Required Python packages
- `chunks.pkl` - Processed text chunks (generated)
- `index.faiss` - FAISS vector index (generated)
- `citations.json` - Mapping for citation lookup (generated)