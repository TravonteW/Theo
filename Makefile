.PHONY: help install ingest embed build assets run clean

help:
	@echo "Available commands:"
	@echo "  install  Install dependencies"
	@echo "  ingest   Process PDFs into chunks.pkl"
	@echo "  embed    Build citations.json + index.faiss"
	@echo "  build    Rebuild chunks + index"
	@echo "  assets   Create theo_assets.zip (for Streamlit Cloud)"
	@echo "  run      Run the Streamlit app"
	@echo "  clean    Remove generated artifacts"

install:
	pip install -r requirements.txt

ingest:
	python ingest.py

embed:
	python embed.py

build:
	python ingest.py --rebuild
	python embed.py --rebuild

assets:
	python -c "import sys, zipfile; from pathlib import Path; out=Path('theo_assets.zip'); files=['index.faiss','citations.json']; missing=[f for f in files if not Path(f).exists()]; sys.exit('Missing: '+', '.join(missing)) if missing else None; with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as zf: [zf.write(f, arcname=Path(f).name) for f in files]; print('Wrote', out)"

run:
	streamlit run app.py

clean:
	python -c "import shutil; from pathlib import Path; [Path(p).unlink(missing_ok=True) for p in ['chunks.pkl','citations.json','index.faiss','conversations.json','theo_assets.zip']]; shutil.rmtree('__pycache__', ignore_errors=True)"
