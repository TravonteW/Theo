# Theo

Theo is a theological research assistant that answers questions across a curated library of sacred and philosophical texts. Responses include strict inline citations (e.g. `[1]`, `[2]`) that map back to the exact passages used.

## What Theo includes

- Chat-style Streamlit UI with conversation threads
- Inline citations + citation cards/snippets
- "Focus Sources" filter to restrict retrieval to selected texts
- Safe defaults for public deployments (disk persistence is off unless you enable it)

## Quickstart (local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI key:
   - Copy `.env.example` to `.env`
   - Fill in `OPENAI_API_KEY` (do not commit `.env`)
3. Put PDFs in `PDF Files/`
4. Build retrieval artifacts:
   ```bash
   python ingest.py
   python embed.py
   ```
5. Run Theo:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud

Do not commit secrets. This repo ignores `.env` and `.streamlit/secrets.toml`.

Theo also requires retrieval artifacts (`index.faiss` and `citations.json`). Because these can be large, the recommended approach is to host them as a zip and let the app download them at runtime.

### Recommended: host an asset bundle zip

1. Generate artifacts locally (run `python ingest.py` then `python embed.py`)
2. Create `theo_assets.zip` containing `index.faiss` and `citations.json`:
   - `make assets`
   - or PowerShell: `Compress-Archive -Path index.faiss,citations.json -DestinationPath theo_assets.zip -Force`
3. Upload `theo_assets.zip` somewhere Streamlit can fetch (a GitHub Release asset works well)
4. In Streamlit Cloud -> Settings -> Secrets, set:
   ```toml
   OPENAI_API_KEY = "your-api-key"
   THEO_ASSET_BUNDLE_URL = "https://host/path/theo_assets.zip"

   # Recommended for public deployments (prevents cross-user conversation leakage):
   THEO_PERSIST_CONVERSATIONS = "0"
   ```
5. Reboot the app.

## Settings (env vars / Streamlit Secrets)

- `OPENAI_API_KEY` (required)
- Retrieval artifacts:
  - `THEO_ASSET_BUNDLE_URL` (zip containing `index.faiss` + `citations.json`)
  - or `THEO_INDEX_URL` + `THEO_CITATIONS_URL` (separate files)
  - `THEO_ASSET_DIR` (optional cache directory for downloads)
- Deployment/privacy:
  - `THEO_PERSIST_CONVERSATIONS` (`1` to enable saving `conversations.json`; default is off)
  - `THEO_DEBUG` (`1` to show full exception details; default is off)
- Response controls (optional):
  - `THEO_REASONING_EFFORT`
  - `THEO_TEXT_VERBOSITY`

## Security notes

- If an API key was ever committed to Git, treat it as compromised: rotate it and rewrite Git history before publishing.
- Disk persistence is disabled by default to avoid cross-user leaks on public deployments.

## Project layout

- `app.py` - Streamlit web UI (Theo)
- `ask.py` - Retrieval + prompting + citation extraction
- `ingest.py` - PDF ingestion -> `chunks.pkl`
- `embed.py` - Embeddings + index build -> `citations.json` + `index.faiss`
- `requirements.txt` - Runtime dependencies
