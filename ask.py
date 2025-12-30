#!/usr/bin/env python3
"""
Question answering system with cited sources from a collection of PDFs.
"""

import os
import json
import shutil
import tempfile
import urllib.request
import zipfile
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from openai import OpenAI


# Cache for loaded resources to avoid re-reading from disk
_cached_index = None
_cached_citation_map = None
_cache_timestamps = {}


def _get_streamlit_secrets():
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return None
    try:
        return st.secrets
    except Exception:
        return None


def _get_theo_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is not None and str(value).strip() != "":
        return value

    secrets = _get_streamlit_secrets()
    if secrets is None:
        return default

    try:
        if key in secrets:
            return str(secrets[key])
    except Exception:
        pass

    try:
        value = secrets.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    except Exception:
        pass

    return default


def _get_asset_dir() -> Path:
    configured_dir = _get_theo_setting("THEO_ASSET_DIR")
    if configured_dir:
        base = Path(str(configured_dir)).expanduser()
    else:
        xdg_cache = os.getenv("XDG_CACHE_HOME")
        base = Path(xdg_cache) / "theo" if xdg_cache else (Path.home() / ".cache" / "theo")

    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "theo"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def get_asset_path(filename: str) -> Path:
    """Resolve an asset path from CWD first, then THEO_ASSET_DIR cache."""
    local = Path(filename)
    if local.exists():
        return local
    return _get_asset_dir() / filename


def _download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp_path.replace(dest_path)


def _download_and_extract_zip(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "theo_assets.zip"
    _download_file(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    try:
        zip_path.unlink()
    except Exception:
        pass


def ensure_retrieval_assets() -> Tuple[Path, Path]:
    """Ensure index.faiss and citations.json are present, downloading if configured."""
    index_path = get_asset_path("index.faiss")
    citations_path = get_asset_path("citations.json")
    if index_path.exists() and citations_path.exists():
        return index_path, citations_path

    asset_dir = _get_asset_dir()
    bundle_url = _get_theo_setting("THEO_ASSET_BUNDLE_URL") or _get_theo_setting("THEO_ASSET_ZIP_URL")
    if bundle_url:
        _download_and_extract_zip(str(bundle_url), asset_dir)
    else:
        # Optional: allow separate URLs for each artifact.
        index_url = _get_theo_setting("THEO_INDEX_URL")
        citations_url = _get_theo_setting("THEO_CITATIONS_URL")
        if index_url and not Path("index.faiss").exists() and not (asset_dir / "index.faiss").exists():
            _download_file(str(index_url), asset_dir / "index.faiss")
        if citations_url and not Path("citations.json").exists() and not (asset_dir / "citations.json").exists():
            _download_file(str(citations_url), asset_dir / "citations.json")

    index_path = get_asset_path("index.faiss")
    citations_path = get_asset_path("citations.json")
    if not index_path.exists() or not citations_path.exists():
        raise FileNotFoundError(
            "Missing retrieval assets: index.faiss and citations.json. "
            "Generate them locally (embed.py), then either (a) host them as a zip and set "
            "THEO_ASSET_BUNDLE_URL in Streamlit secrets, or (b) provide THEO_INDEX_URL and "
            "THEO_CITATIONS_URL."
        )

    return index_path, citations_path


def _hydrate_openai_env_from_streamlit_secrets() -> None:
    """Load OPENAI_API_KEY from Streamlit secrets when env vars are not set."""
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


def _openai_client() -> OpenAI:
    _hydrate_openai_env_from_streamlit_secrets()
    return OpenAI()

def _normalize_source_name(name: str) -> str:
    return Path(name).stem.replace("_", " ").lower().strip()

def load_resources() -> Tuple[faiss.Index, Dict[str, Dict]]:
    """Load the FAISS index and citation map with caching."""
    global _cached_index, _cached_citation_map, _cache_timestamps

    index_path, citations_path = ensure_retrieval_assets()

    # Check file modification times
    index_key = str(index_path.resolve())
    citations_key = str(citations_path.resolve())
    index_mtime = index_path.stat().st_mtime
    citations_mtime = citations_path.stat().st_mtime

    # Load from cache if files haven't changed
    if (
        _cached_index is not None
        and _cached_citation_map is not None
        and _cache_timestamps.get("index_path") == index_key
        and _cache_timestamps.get("citations_path") == citations_key
        and _cache_timestamps.get("index_mtime") == index_mtime
        and _cache_timestamps.get("citations_mtime") == citations_mtime
    ):
        return _cached_index, _cached_citation_map

    # Load FAISS index
    _cached_index = faiss.read_index(str(index_path))

    # Load citation map
    with citations_path.open("r", encoding="utf-8") as f:
        _cached_citation_map = json.load(f)

    # Update cache timestamps
    _cache_timestamps["index_path"] = index_key
    _cache_timestamps["citations_path"] = citations_key
    _cache_timestamps["index_mtime"] = index_mtime
    _cache_timestamps["citations_mtime"] = citations_mtime

    return _cached_index, _cached_citation_map


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate embedding for a single text using OpenAI's embedding model."""
    client = _openai_client()
    response = client.embeddings.create(
        model=model,
        input=[text]
    )
    return response.data[0].embedding


def _collect_diverse_contexts_from_indices(
    indices: np.ndarray,
    citation_map: Dict[str, Dict],
    limit: int,
    per_source_k: int,
    allowed_sources: Optional[Set[str]] = None,
) -> List[Dict]:
    results: List[Dict] = []
    seen: Set[str] = set()
    per_source_counts: Dict[str, int] = {}

    def _add(chunk_id: str, info: Dict):
        results.append({
            "id": chunk_id,
            "source_pdf": info["source_pdf"],
            "page_num": info["page_num"],
            "text": info["text"],
        })
        seen.add(chunk_id)
        per_source_counts[info["source_pdf"]] = per_source_counts.get(info["source_pdf"], 0) + 1

    # First pass: enforce per-source cap to encourage source diversity.
    for idx in indices[0]:
        chunk_id = str(int(idx))
        if chunk_id in seen or chunk_id not in citation_map:
            continue
        info = citation_map[chunk_id]
        source = info["source_pdf"]
        if allowed_sources and source not in allowed_sources:
            continue
        if per_source_counts.get(source, 0) >= per_source_k:
            continue
        _add(chunk_id, info)
        if len(results) >= limit:
            return results

    # Second pass: top up ignoring per-source cap (still de-duping).
    for idx in indices[0]:
        chunk_id = str(int(idx))
        if chunk_id in seen or chunk_id not in citation_map:
            continue
        info = citation_map[chunk_id]
        source = info["source_pdf"]
        if allowed_sources and source not in allowed_sources:
            continue
        _add(chunk_id, info)
        if len(results) >= limit:
            break

    return results

def _available_sources(citation_map: Dict[str, Dict]) -> List[str]:
    return sorted({v["source_pdf"] for v in citation_map.values()})


def _extract_target_sources(question: str, sources: List[str]) -> List[str]:
    """Heuristically detect which sources are mentioned in the question."""
    q = question.lower()
    src_lc = [(s, s.lower()) for s in sources]

    # Simple alias patterns
    aliases = {
        "quran": ["quran", "koran"],
        "king james": ["kjv", "king james"],
        "bible": ["bible"],
        "septuagint": ["septuagint", "lxx"],
        "vulgate": ["vulgate"],
        "talmud": ["talmud"],
        "tanakh": ["tanakh"],
        "gita": ["bhagavad", "gita"],
        "upanishads": ["upanishad"],
        "dhammapada": ["dhammapada"],
    }

    matched: List[str] = []
    for label, keys in aliases.items():
        if any(k in q for k in keys):
            # Find best matching available filename containing any alias token
            for orig, lc in src_lc:
                if any(k in lc for k in keys):
                    matched.append(orig)
    # Also match direct filename substrings if user typed exact-ish title words
    for orig, lc in src_lc:
        terms = [t for t in lc.replace(".pdf", "").split() if len(t) > 3]
        if any(t in q for t in terms):
            matched.append(orig)

    # Deduplicate while preserving order
    seen_set: Set[str] = set()
    ordered = []
    for m in matched:
        if m not in seen_set:
            ordered.append(m)
            seen_set.add(m)
    return ordered



def determine_response_style(question: str) -> str:
    """Heuristically decide whether to answer in a casual or research tone."""
    q = question.lower()
    tokens = q.split()
    research_markers = {
        "compare", "contrast", "analysis", "analyze", "difference", "differences",
        "evidence", "explain", "exposition", "discuss", "evaluate", "historical",
        "context", "significance", "implications", "sources", "citations", "outline",
        "summarize", "what does", "according", "interpret"
    }
    casual_markers = {
        "quick", "brief", "short", "simple", "hey", "hi", "hello",
        "what's", "whats", "tell me", "remind", "give me", "in a sentence",
        "tl;dr", "tldr", "plain", "casual"
    }
    if any(marker in q for marker in research_markers) or len(tokens) >= 18:
        return "research"
    if len(tokens) <= 10 or any(marker in q for marker in casual_markers):
        return "casual"
    return "research"

def retrieve_multi_source_contexts(
    question: str,
    index: faiss.Index,
    citation_map: Dict[str, Dict],
    k_total: int = 8,
    per_source_k: int = 3,
    breadth: int = 200,
    focus_sources: Optional[List[str]] = None,
) -> List[Dict]:
    """Retrieve contexts ensuring coverage across mentioned sources when possible."""
    sources = _available_sources(citation_map)
    normalized_lookup = {src: _normalize_source_name(src) for src in sources}

    matched_focus = []
    if focus_sources:
        for requested in focus_sources:
            req_norm = _normalize_source_name(requested)
            match = next((src for src, norm in normalized_lookup.items() if req_norm == norm or req_norm in norm or norm in req_norm), None)
            if match and match not in matched_focus:
                matched_focus.append(match)

    if matched_focus:
        targets = matched_focus
    else:
        targets = _extract_target_sources(question, sources)

    forced_mode = bool(matched_focus)
    allowed_sources = set(matched_focus) if matched_focus else None

    if not targets:
        qe = get_embedding(question)
        qv = np.array([qe]).astype('float32')
        _, inds = index.search(qv, max(breadth, k_total * 60))
        diverse = _collect_diverse_contexts_from_indices(
            inds,
            citation_map,
            limit=k_total,
            per_source_k=max(1, per_source_k),
            allowed_sources=allowed_sources,
        )
        return diverse

    if not forced_mode and len(targets) < 2:
        qe = get_embedding(question)
        qv = np.array([qe]).astype('float32')
        _, inds = index.search(qv, max(breadth, k_total * 60))
        return _collect_diverse_contexts_from_indices(
            inds,
            citation_map,
            limit=min(k_total, 12),
            per_source_k=max(1, min(per_source_k, 3)),
        )

    collected: List[Dict] = []
    used_ids: Set[str] = set()

    # First pass: guarantee at least one chunk per target source
    source_counts = {src: 0 for src in targets}

    # For each target source, bias retrieval by appending a focus hint
    for src in targets:
        sub_q = f"{question} (focus on {src})"
        qe = get_embedding(sub_q)
        qv = np.array([qe]).astype('float32')
        _, inds = index.search(qv, breadth)

        # Filter to the desired source and collect top per_source_k unique chunks
        count = 0
        for idx_val in inds[0]:
            cid = str(int(idx_val))
            if cid in citation_map and cid not in used_ids:
                info = citation_map[cid]
                if info.get("source_pdf") == src:
                    collected.append({
                        "id": cid,
                        "source_pdf": info["source_pdf"],
                        "page_num": info["page_num"],
                        "text": info["text"],
                    })
                    used_ids.add(cid)
                    source_counts[src] += 1
                    count += 1
                    if count >= per_source_k:
                        break

    # Ensure at least one result per selected source before backfilling
    missing_sources = [src for src, count in source_counts.items() if count == 0]
    if missing_sources:
        # Try harder to find at least one result per missing source
        for src in missing_sources:
            # More specific search for this source
            specific_q = f"information from {src} about {question}"
            qe = get_embedding(specific_q)
            qv = np.array([qe]).astype('float32')
            _, inds = index.search(qv, breadth * 2)

            for idx_val in inds[0]:
                cid = str(int(idx_val))
                if cid in citation_map and cid not in used_ids:
                    info = citation_map[cid]
                    if info.get("source_pdf") == src:
                        collected.append({
                            "id": cid,
                            "source_pdf": info["source_pdf"],
                            "page_num": info["page_num"],
                            "text": info["text"],
                        })
                        used_ids.add(cid)
                        break  # Just need one per missing source

    # If not enough contexts, top up with general nearest neighbors
    if len(collected) < k_total:
        qe = get_embedding(question)
        qv = np.array([qe]).astype('float32')
        _, inds = index.search(qv, max(k_total * 4, 50))
        for idx_val in inds[0]:
            cid = str(int(idx_val))
            if cid in citation_map and cid not in used_ids:
                info = citation_map[cid]
                if forced_mode and info["source_pdf"] not in allowed_sources:
                    continue
                collected.append({
                    "id": cid,
                    "source_pdf": info["source_pdf"],
                    "page_num": info["page_num"],
                    "text": info["text"],
                })
                used_ids.add(cid)
            if len(collected) >= k_total:
                break

    return collected[:k_total]


def summarize_long_history(history: List[Dict[str, str]], max_turns: int = 6) -> List[Dict[str, str]]:
    """Summarize conversation history if it exceeds max_turns to keep focus."""
    if len(history) <= max_turns:
        return history

    # Keep the most recent max_turns-1 turns and create a summary of the rest
    recent_turns = history[-(max_turns-1):]
    older_turns = history[:-(max_turns-1)]

    # Create a summary of older turns
    if older_turns:
        user_questions = [turn["content"] for turn in older_turns if turn["role"] == "user"]
        if user_questions:
            summary_content = f"[Previous discussion covered: {', '.join(user_questions[:3])}{'...' if len(user_questions) > 3 else ''}]"
            summary_turn = {"role": "system", "content": summary_content}
            return [summary_turn] + recent_turns

    return recent_turns

def create_prompt_with_context(question: str, contexts: List[Dict], history: Optional[List[Dict[str, str]]] = None) -> List[Dict]:
    """Create a prompt with the question and retrieved contexts."""
    # System message
    response_style = determine_response_style(question)
    multi_source = len({c["source_pdf"] for c in contexts}) > 1
    num_contexts = len(contexts)
    base_instructions = (
        "You are Theo, a theological assistant specializing in Christianity, Islam, Buddhism, Hinduism, and Judaism. "
        "Your goal is to help users understand and explore religious and sacred texts from these traditions.\n\n"
        "Use the supplied passages (and any prior turns the user already saw) as your evidence. Do not invent details. "
        f"CRITICAL: Cite using only citation numbers [1] through [{num_contexts}]. Never invent citation numbers beyond this range.\n\n"
        "Citations:\n"
        "- Cite specific passages for complete context, using inline citations like [1] or [2][3].\n"
        "- If you make a claim, support it with a citation whenever the passages support it.\n\n"
        "Semantics and phrasing:\n"
        "- If the user's wording differs from the passages, map their terms to the closest concepts in the text (synonyms, paraphrases, related terms).\n"
        "- Avoid saying an idea is \"not in the text\" when the passages express it using different words; instead explain the connection and cite it.\n\n"
        "Hypotheticals and application:\n"
        "- If the user asks a hypothetical or how a principle applies to a scenario not explicitly described, first ground the answer in the passages (cite the relevant principles), "
        "then provide a careful, clearly-labeled interpretation or application that follows from those principles.\n"
        "- If the passages do not support the requested detail, say so plainly and ask a clarifying question or offer the closest supported framing.\n\n"
        "Interpretation:\n"
        "- When content is ambiguous or interpretive, provide a balanced view acknowledging different perspectives that are consistent with the passages.\n"
        "- Maintain a respectful tone toward diverse beliefs and interpretations.\n\n"
        "Do not mention limitations, training data, or the retrieval process."
    )

    casual_instructions = (
        " The user sounds conversational or is asking for a quick take. Keep the response concise (roughly 120-180 words unless asked otherwise), "
        "use a warm, plainspoken tone, and include only the most helpful citations."
    )
    research_instructions = (
        " The user is seeking a thorough or comparative explanation. Maintain an analytical tone, expand on key points with citations, "
        "and explicitly note meaningful disagreements or gaps when the passages conflict or do not address the user's request."
    )
    style_instructions = casual_instructions if response_style == "casual" else research_instructions
    compare_instructions = (
        " When multiple distinct sources are provided, explicitly compare them where relevant, "
        "highlighting both agreements and differences with proper citations."
    )
    system_message = {
        "role": "system",
        "content": base_instructions + style_instructions + (compare_instructions if multi_source else "")
    }

    messages = [system_message]

    # Add conversation history if provided, with summarization for long histories
    if history:
        processed_history = summarize_long_history(history, max_turns=6)
        messages.extend(processed_history)
    
    # Create context string with numbered citations
    context_str = "Here are some relevant passages:\n\n"
    for i, ctx in enumerate(contexts, 1):
        source = ctx["source_pdf"]
        page = ctx["page_num"]
        text = ctx["text"]
        context_str += f"[{i}] From '{source}', page {page}:\n{text}\n\n"
    
    # User message with context and question
    user_message = {
        "role": "user",
        "content": f"{context_str}Based on these sources, please answer: {question}"
    }
    
    # Add the current user question
    messages.append(user_message)
    return messages


def get_completion(
    messages: List[Dict],
    model: str = "gpt-5-mini",
    fallback_model: str = "gpt-5.2",
    reasoning_effort: Optional[str] = None,
    text_verbosity: Optional[str] = None,
) -> str:
    """Get a completion from the OpenAI API."""
    client = _openai_client()

    reasoning_effort = reasoning_effort or _get_theo_setting("THEO_REASONING_EFFORT", "medium")
    text_verbosity = text_verbosity or _get_theo_setting("THEO_TEXT_VERBOSITY", "medium")

    def _create_chat_completion(model_name: str):
        kwargs = {
            "model": model_name,
            "messages": messages,
            "reasoning": {"effort": reasoning_effort},
            "text": {"verbosity": text_verbosity},
        }
        try:
            return client.chat.completions.create(**kwargs)
        except TypeError:
            # Back-compat for older SDKs that don't accept GPT-5 controls yet.
            return client.chat.completions.create(model=model_name, messages=messages)
    
    try:
        response = _create_chat_completion(model)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with {model}, falling back to {fallback_model}: {e}")
        response = _create_chat_completion(fallback_model)
        return response.choices[0].message.content


def deduplicate_and_stitch_chunks(contexts: List[Dict]) -> List[Dict]:
    """De-duplicate chunks and stitch adjacent pages from same source."""
    from collections import defaultdict

    # Group by source and page
    by_source_page = defaultdict(lambda: defaultdict(list))
    for ctx in contexts:
        source = ctx["source_pdf"]
        page = ctx["page_num"]
        by_source_page[source][page].append(ctx)

    # Process each source
    result = []
    for source in sorted(by_source_page.keys()):
        pages = sorted(by_source_page[source].keys())
        i = 0
        while i < len(pages):
            current_page = pages[i]
            current_chunks = by_source_page[source][current_page]

            # Look for adjacent pages to stitch together
            stitched_text = current_chunks[0]["text"]
            stitched_pages = [current_page]
            j = i + 1

            # Check if next pages are adjacent and should be stitched
            while j < len(pages) and pages[j] == pages[j-1] + 1:
                next_chunks = by_source_page[source][pages[j]]
                if len(next_chunks) == 1 and len(current_chunks) == 1:
                    # Only stitch if both pages have single chunks
                    stitched_text += f" ... {next_chunks[0]['text']}"
                    stitched_pages.append(pages[j])
                    j += 1
                else:
                    break

            # Create stitched result
            result.append({
                "id": current_chunks[0]["id"],
                "source_pdf": source,
                "page_num": stitched_pages[0] if len(stitched_pages) == 1 else f"{stitched_pages[0]}-{stitched_pages[-1]}",
                "text": stitched_text,
                "pages_stitched": len(stitched_pages)
            })

            i = j

    return result

def group_and_deduplicate_citations(citations: List[Dict]) -> List[Dict]:
    """Group citations by source, then page; de-duplicate repeated pages."""
    from collections import defaultdict

    def _page_sort_key(page_key: str):
        head = str(page_key).split('-', 1)[0]
        return (0, int(head)) if head.isdigit() else (1, head.lower())

    # Group by source, then by normalized page key to avoid str/int mismatches
    grouped = defaultdict(lambda: defaultdict(list))
    for citation in citations:
        source = citation["pdf"]
        page_value = citation["page"]
        page_key = str(page_value)
        grouped[source][page_key].append({**citation, "page": page_value})

    # Flatten back to list, combining snippets from same source+page
    result = []
    for source in sorted(grouped.keys()):
        for page_key in sorted(grouped[source].keys(), key=_page_sort_key):
            page_citations = grouped[source][page_key]
            if len(page_citations) == 1:
                result.append(page_citations[0])
            else:
                # Combine snippets from same page
                combined_snippet = " ... ".join(c["snippet"] for c in page_citations)
                # Keep the first citation's index for numbering
                result.append({
                    "pdf": source,
                    "page": page_citations[0]["page"],
                    "snippet": combined_snippet,
                    "index": page_citations[0]["index"]
                })

    return result

def extract_citations(answer: str, contexts: List[Dict]) -> Dict:
    """Extract citations from the answer and map them to the context."""
    # Prepare the result structure
    result = {
        "answer": answer,
        "citations": []
    }

    # Create a mapping from citation numbers to contexts
    citation_map = {}
    for i, ctx in enumerate(contexts, 1):
        citation_map[i] = {
            "pdf": ctx["source_pdf"],
            "page": ctx["page_num"],
            "snippet": ctx["text"],
            "index": i
        }

    # Find all citations in the answer using a simple heuristic
    # This assumes citations are in the format [n] where n is a number
    import re
    citation_numbers = set(int(match) for match in re.findall(r'\[(\d+)\]', answer))

    # Add the citations in order
    raw_citations = []
    for i in sorted(citation_numbers):
        if i in citation_map:
            raw_citations.append(citation_map[i])

    # Group and deduplicate citations
    result["citations"] = group_and_deduplicate_citations(raw_citations)

    return result


def _count_unique_citations(answer_text: str) -> int:
    import re
    nums = set(int(match) for match in re.findall(r"\[(\d+)\]", answer_text))
    return len(nums)


def _answer_suggests_more_context(answer_text: str) -> bool:
    text = answer_text.lower()
    triggers = (
        "not addressed",
        "not mentioned",
        "not in the text",
        "not in the passages",
        "insufficient",
        "not enough information",
        "cannot determine",
        "can't determine",
        "unclear from the passages",
    )
    return any(t in text for t in triggers)


def answer(question: str, conversation: Optional[List[Dict[str, str]]] = None, focus_sources: Optional[List[str]] = None) -> Dict:
    """Answer a question with cited sources."""
    # Load resources
    index, citation_map = load_resources()
    
    response_style = determine_response_style(question)
    if focus_sources:
        k_total, per_source_k, breadth = 8, 3, 200
    else:
        is_long = len(question.split()) >= 18
        is_comparative = any(k in question.lower() for k in ("compare", "contrast", "difference", "differences", "across", "versus", "vs"))
        if response_style == "research" or is_long or is_comparative:
            k_total, per_source_k, breadth = 12, 3, 450
        else:
            k_total, per_source_k, breadth = 10, 2, 320

    # Retrieve contexts, ensuring multi-source coverage when applicable
    raw_contexts = retrieve_multi_source_contexts(
        question,
        index,
        citation_map,
        k_total=k_total,
        per_source_k=per_source_k,
        breadth=breadth,
        focus_sources=focus_sources,
    )

    # Apply chunk de-duplication and page-window stitching
    contexts = deduplicate_and_stitch_chunks(raw_contexts)
    
    # Create prompt with context
    messages = create_prompt_with_context(question, contexts, history=conversation)
    
    # Get completion
    answer_text = get_completion(messages)

    # If no specific sources were selected, allow the model to "pull wider" when needed:
    # if the answer has too few citations or signals missing context, re-run once with a broader retrieval.
    if not focus_sources:
        citation_count = _count_unique_citations(answer_text)
        if citation_count < 2 or _answer_suggests_more_context(answer_text):
            raw_contexts_wide = retrieve_multi_source_contexts(
                question,
                index,
                citation_map,
                k_total=max(k_total, 16),
                per_source_k=max(per_source_k, 3),
                breadth=max(breadth, 700),
                focus_sources=None,
            )
            contexts_wide = deduplicate_and_stitch_chunks(raw_contexts_wide)
            if len(contexts_wide) > len(contexts):
                messages_wide = create_prompt_with_context(question, contexts_wide, history=conversation)
                answer_text_wide = get_completion(messages_wide)
                if _count_unique_citations(answer_text_wide) >= citation_count:
                    contexts = contexts_wide
                    answer_text = answer_text_wide
    
    # Extract and format citations
    result = extract_citations(answer_text, contexts)
    
    return result


def main():
    """CLI interface for the question answering system."""
    print("Welcome to the PDF Question Answering System")
    print("Type 'exit' to quit")
    
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        if not question:
            continue
        
        try:
            result = answer(question)
            
            print("\nAnswer:")
            print(result["answer"])
            
            if result["citations"]:
                print("\nCitations:")
                for i, citation in enumerate(result["citations"], 1):
                    print(f"[{i}] {citation['pdf']}, Page {citation['page']}")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
