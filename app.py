#!/usr/bin/env python3
"""
Streamlit web interface for the PDF Q&A system.
"""

import base64
import os
import streamlit as st
import json
import textwrap
import html
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ask import answer, get_asset_path

# Text collections organized by religious tradition
TEXT_COLLECTIONS = {
    "Buddhist Texts": [
        "Amitabha",
        "The Buddhist Catechism",
        "The Life of Buddha",
        "The Creed of Buddha",
        "Sacred Books of the East Vol. 10 The Dhammapada and Sutta Nipata",
    ],
    "Christian Texts": [
        "Greek New Testament",
        "The King James Version",
        "The Jefferson Bible",
        "The Complete Sayings of Jesus",
        "The Apocrypha",
        "The Septuagint",
        "The Vulgate",
    ],
    "Jewish Texts": [
        "Tanakh 1917",
        "The Babylonian Talmud in Selection",
        "Eighteen Treatises from the Mishna",
        "The Talmud Selections",
        "The Wisdom of the Talmud",
    ],
    "Islamic Texts": [
        "The Holy Quran",
        "The Meaning of the Glorious Qur'an",
        "The Religion of the Koran",
    ],
    "Hindu Texts": [
        "Srimad-Bhagavad-Gita",
        "From the Upanishads",
        "The Ramayana and Mahabharata",
        "The Hindu Book of Astrology",
        "Vedic Hymns P1",
        "Vedic Hymns P2",
    ],
}

MAX_CONVERSATIONS = 20
QUICK_QUESTION_TEMPLATES = [
    "Summarize the core teaching of the Sermon on the Mount.",
    "Compare how the Bhagavad Gita and the Dhammapada describe duty.",
    "What does the Talmud say about justice?",
    "Explain the concept of mercy in the Quran."
]

TYPING_HTML = """
<div class='message-bubble assistant'>
    <div class='message-meta'><span>Theo</span><span>typing...</span></div>
    <div class='typing-indicator'>
        <span class='typing-dot'></span>
        <span class='typing-dot'></span>
        <span class='typing-dot'></span>
    </div>
</div>
"""


def disk_persistence_enabled() -> bool:
    """Controls whether conversations are saved/loaded from conversations.json.

    WARNING: In a public Streamlit deployment, disk persistence can mix different users'
    conversations on the server. Keep disabled unless you understand the privacy impact.
    """
    # Local import as a belt-and-suspenders guard for deploy environments where
    # module-level imports may have been altered during hot reloads.
    import os as _os

    value = str(_os.getenv("THEO_PERSIST_CONVERSATIONS", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}

def format_title(name: str) -> str:
    return name.removesuffix(".pdf")

def normalize_title(filename: str) -> str:
    return Path(filename).stem.replace("_", " ")

def sanitize_html(text: str) -> str:
    """Sanitize HTML content to prevent XSS attacks"""
    return html.escape(text)


def safe_js_text(text: str) -> str:
    return sanitize_html(text).replace('"', '&quot;').replace("'", "&#39;")


def encode_copy_payload(text: str) -> str:
    return base64.b64encode(text.encode('utf-8')).decode('ascii')


def ensure_conversation_ids():
    """Guarantee each saved conversation has a unique numeric ID and a safe counter."""

    def _parse_id(value):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    updated = False
    conversations = st.session_state.conversations

    # If duplicate IDs exist, keep the "best" conversation on the original ID and re-ID the rest.
    groups: Dict[int, List[Dict]] = {}
    for conv in conversations:
        cid = _parse_id(conv.get("id"))
        if cid is not None:
            groups.setdefault(cid, []).append(conv)

    for cid, group in groups.items():
        if len(group) <= 1:
            continue

        def _score(c: Dict):
            messages = c.get("messages") or []
            ts = c.get("timestamp") or ""
            return (len(messages), ts)

        group_sorted = sorted(group, key=_score, reverse=True)
        # Keep the winner's ID, clear IDs on the rest so they get reassigned.
        for dup in group_sorted[1:]:
            dup["id"] = None
            updated = True

    # Assign missing/invalid IDs and normalize IDs to ints.
    used: Set[int] = set()
    max_id = 0
    for conv in conversations:
        cid = _parse_id(conv.get("id"))
        if cid is not None:
            used.add(cid)
            max_id = max(max_id, cid)

    next_id = max(max_id + 1, int(st.session_state.conversation_counter or 1))
    for conv in conversations:
        cid = _parse_id(conv.get("id"))
        if cid is None:
            conv["id"] = next_id
            used.add(next_id)
            next_id += 1
            updated = True
        else:
            if conv.get("id") != cid:
                conv["id"] = cid
                updated = True

    if next_id != st.session_state.conversation_counter:
        st.session_state.conversation_counter = next_id
        updated = True

    # Clean up any selection/rename state referencing IDs that no longer exist.
    existing_ids = {str(conv.get("id")) for conv in conversations}
    if st.session_state.get("selected_conversations"):
        filtered = [cid for cid in st.session_state.selected_conversations if str(cid) in existing_ids]
        if filtered != st.session_state.selected_conversations:
            st.session_state.selected_conversations = filtered
            updated = True
    if st.session_state.get("rename_mode") and str(st.session_state.rename_mode) not in existing_ids:
        st.session_state.rename_mode = None
        updated = True

    if updated:
        save_conversations_to_disk(conversations)


def find_conversation_index(conv_id):
    for idx, conv in enumerate(st.session_state.conversations):
        if str(conv.get('id')) == str(conv_id):
            return idx
    return None


def ensure_active_conversation():
    """Create a placeholder conversation if none is active."""
    if st.session_state.active_conversation_id is None:
        new_id = st.session_state.conversation_counter
        st.session_state.conversation_counter += 1
        placeholder = {
            "id": new_id,
            "title": "New conversation",
            "messages": [],
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.conversations.append(placeholder)
        st.session_state.active_conversation_id = new_id
    return st.session_state.active_conversation_id


def persist_current_conversation():
    """Sync the in-progress conversation back into the sidebar list."""
    if not st.session_state.current_conversation:
        return
    active_id = ensure_active_conversation()
    idx = find_conversation_index(active_id)
    if idx is None:
        return
    st.session_state.conversations[idx]["messages"] = st.session_state.current_conversation.copy()
    st.session_state.conversations[idx]["title"] = generate_conversation_title(st.session_state.current_conversation)
    st.session_state.conversations[idx]["timestamp"] = datetime.now().isoformat()
    st.session_state.conversations = st.session_state.conversations[-MAX_CONVERSATIONS:]
    save_conversations_to_disk(st.session_state.conversations)


def ensure_active_conversation_loaded():
    active_id = st.session_state.active_conversation_id
    if active_id is None:
        return
    idx = find_conversation_index(active_id)
    if idx is None:
        st.session_state.active_conversation_id = None
        st.session_state.current_conversation = []
    else:
        st.session_state.current_conversation = st.session_state.conversations[idx]["messages"].copy()


def remove_conversation(base_id):
    idx = find_conversation_index(base_id)
    if idx is None:
        return False
    st.session_state.conversations.pop(idx)
    return True


def create_message(msg_type: str, content: str, citations=None, source_question: str | None = None):
    """Create a structured chat message with metadata"""
    citations = citations or []
    st.session_state.message_counter += 1
    return {
        "id": f"{msg_type}-{st.session_state.message_counter}",
        "type": msg_type,
        "content": content,
        "citations": citations,
        "timestamp": datetime.now().isoformat(),
        "source_question": source_question,
    }


def sync_message_counter():
    """Ensure the message counter stays ahead of any stored messages."""
    max_counter = st.session_state.message_counter
    for conv in st.session_state.conversations:
        for msg in conv.get("messages", []):
            msg_id = str(msg.get("id", "") or "0")
            try:
                suffix = int(msg_id.split("-")[-1])
                max_counter = max(max_counter, suffix)
            except ValueError:
                continue
    st.session_state.message_counter = max_counter


def format_timestamp(timestamp: str | None) -> str:
    if not timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%b %d - %H:%M")
    except ValueError:
        return timestamp or ""


def run_answer_flow(question: str, add_question: bool = True, focus_sources=None) -> bool:
    """Shared helper to call the backend, append messages, and keep metadata."""
    if not question:
        return False
    ensure_active_conversation()

    history_messages = []
    for msg in st.session_state.current_conversation[-6:]:
        if msg["type"] == "question":
            history_messages.append({"role": "user", "content": msg["content"]})
        elif msg["type"] == "answer":
            history_messages.append({"role": "assistant", "content": msg["content"]})

    focus = focus_sources if focus_sources is not None else (st.session_state.selected_sources or None)
    try:
        result = answer(question, conversation=history_messages, focus_sources=focus)
        st.session_state.last_error = None
    except Exception as exc:
        debug = str(os.getenv("THEO_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
        missing_assets = (
            isinstance(exc, FileNotFoundError)
            and ("index.faiss" in str(exc) or "citations.json" in str(exc))
        )
        friendly = (
            "Theo is missing its retrieval files (`index.faiss` / `citations.json`). "
            "Generate them locally (run `embed.py`), then host them and set "
            "`THEO_ASSET_BUNDLE_URL` (or `THEO_INDEX_URL` + `THEO_CITATIONS_URL`) in Streamlit Secrets, "
            "then reboot the app."
        )
        st.session_state.last_error = f"{type(exc).__name__}: {exc}" if debug else (friendly if missing_assets else "Theo hit an error generating a reply. Please try again.")
        if debug:
            raise
        return False

    if add_question:
        st.session_state.current_conversation.append(create_message("question", question))

    st.session_state.current_conversation.append(
        create_message("answer", result["answer"], result.get("citations", []), source_question=question)
    )
    persist_current_conversation()
    return True

def make_citations_clickable(text: str, answer_counter: int) -> str:
    """Convert citation numbers [1], [2] etc into clickable chips that scroll to citation cards"""
    def replace_citation(match):
        citation_num = match.group(1)
        citation_id = f"citation-{answer_counter}-{citation_num}"
        return f'<a href="#{citation_id}" class="citation-chip" onclick="document.getElementById(\'{citation_id}\').scrollIntoView({{behavior: \'smooth\', block: \'center\'}})">[{citation_num}]</a>'

    return re.sub(r'\[(\d+)\]', replace_citation, text)

def export_answer_as_markdown(answer_text: str, citations: list) -> str:
    """Export answer with citations as Markdown format"""
    markdown = f"# Answer\n\n{answer_text}\n\n## Citations\n\n"
    for citation in citations:
        markdown += f"[{citation['index']}] {citation['pdf']}, Page {citation['page']}\n"
        markdown += f"> {citation['snippet']}\n\n"
    return markdown

def export_citations_as_json(citations: list) -> str:
    """Export citations as JSON format"""
    return json.dumps(citations, indent=2, ensure_ascii=False)

def generate_conversation_title(conversation):
    """Generate AI-friendly conversation titles"""
    questions = [m["content"] for m in conversation if m["type"] == "question"]
    if not questions:
        return "Conversation"

    def tidy(snippet: str) -> str:
        cleaned = re.sub(r"\s+", " ", snippet).strip()
        if not cleaned:
            return "Conversation"
        words = cleaned.split()
        preview = " ".join(words[:8]).rstrip(".,;:!?")
        return preview[:1].upper() + preview[1:]

    if len(questions) == 1:
        return tidy(questions[0])
    return f"{tidy(questions[0])} <-> {tidy(questions[1])}"

def save_conversations_to_disk(conversations, max_conversations=MAX_CONVERSATIONS):
    """Save conversations to disk with size cap and rotation"""
    if not disk_persistence_enabled():
        return

    # Add timestamp to conversations if not present
    for conv in conversations:
        if 'timestamp' not in conv:
            conv['timestamp'] = datetime.now().isoformat()

    # Sort by timestamp (newest first) and limit to max_conversations
    conversations_sorted = sorted(conversations, key=lambda x: x.get('timestamp', ''), reverse=True)
    conversations_to_save = conversations_sorted[:max_conversations]

    try:
        with open('conversations.json', 'w', encoding='utf-8') as f:
            json.dump(conversations_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save conversations: {e}")

def load_conversations_from_disk():
    """Load conversations from disk"""
    if not disk_persistence_enabled():
        return []
    try:
        with open('conversations.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Failed to load conversations: {e}")
        return []

def init_session_state():
    """Initialize all session state keys once at startup to avoid key errors"""
    defaults = {
        "conversations": [],
        "current_conversation": [],
        "conversation_counter": 1,
        "selected_sources": [],
        "clear_sources_pending": False,
        "rename_mode": None,  # stores the ID of conversation being renamed
        "typing_indicator": False,
        "conversation_search": "",
        "selected_conversations": [],
        "pending_prefill": "",
        "question_input": "",
        "question_needs_reset": False,
        "message_counter": 0,
        "active_conversation_id": None,
        "user_settings": {
            "show_timestamps": True,
            "compact_mode": False,
            "auto_scroll": True,
        },
        "last_error": None,
        "pending_delete_confirmation": False,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@st.cache_data
def load_available_sources():
    """Return a mapping of normalized source names -> original source name.

    In Streamlit Cloud (or fresh checkouts), `citations.json` may not exist yet.
    In that case we fall back to the curated `TEXT_COLLECTIONS` list so the UI
    can render without crashing.
    """
    citations_path = get_asset_path("citations.json")
    try:
        with citations_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        lookup: Dict[str, str] = {}
        for info in data.values():
            original = info["source_pdf"]
            normalized = normalize_title(original)
            lookup[normalized] = original
        return lookup
    except FileNotFoundError:
        pass
    except Exception:
        # If the file is present but unreadable, fall back to curated titles.
        pass

    fallback: Dict[str, str] = {}
    for tradition_texts in TEXT_COLLECTIONS.values():
        for title in tradition_texts:
            normalized = normalize_title(title)
            fallback.setdefault(normalized, title)
    return fallback

# Set page configuration
st.set_page_config(
    page_title="Theo",
    page_icon="Theo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    :root {
        --page-bg: radial-gradient(circle at top, #1d1e25, #050505 65%);
        --panel-bg: rgba(15, 17, 25, 0.95);
        --panel-border: #272b3c;
        --accent: #4cc3ff;
        --accent-2: #f7e1a1;
        --text-color: #f5f5f5;
        --text-muted: #b0b7c8;
        --danger: #ff6b6b;
    }

    .stApp {
        background: var(--page-bg);
        color: var(--text-color);
        font-family: 'Inter','Segoe UI',sans-serif;
    }

    body, .stMarkdown, .stText, .stTextInput label, .stCaption {
        color: var(--text-color) !important;
    }

    .stApp > main .block-container {
        max-width: 1150px;
        padding: 2.5rem 2rem 3.5rem;
        gap: 1.5rem;
    }

    .hero-title h1 {
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #f7e1a1, #4cc3ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-title .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1.25rem;
        color: var(--text-muted);
    }

    .quick-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.4rem 0.85rem;
        margin: 0.25rem 0.35rem 0.35rem 0;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.03);
        color: var(--text-color);
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .quick-chip:hover {
        border-color: var(--accent);
        box-shadow: 0 0 12px rgba(76,195,255,0.25);
    }

    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 10px !important;
        border: 1px solid var(--panel-border) !important;
        padding: 0.5rem 0.75rem !important;
        color: var(--text-color) !important;
    }

    div[data-testid="stExpander"] > div[data-testid="stExpanderContent"] {
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: 16px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.45);
    }

    .source-counter {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.4rem;
    }

    .stTextInput > div > div {
        border-radius: 14px;
        padding: 0.65rem 0.85rem;
        background: rgba(0,0,0,0.6);
        border: 1px solid var(--panel-border);
    }

    .stTextInput input {
        color: var(--text-color) !important;
    }

    .stButton button {
        padding: 0.65rem 1rem;
        min-height: 42px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.1);
        background: linear-gradient(120deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        color: var(--text-color);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .stButton button:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
    }

    .chat-stream {
        background: rgba(0,0,0,0.15);
        border-radius: 18px;
        padding: 1.25rem;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        width: 100%;
        box-sizing: border-box;
    }

    .message-bubble {
        position: relative;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 32px rgba(0,0,0,0.35);
        width: 100%;
        box-sizing: border-box;
    }

    .message-bubble.user {
        background: rgba(76,195,255,0.08);
        border-color: rgba(76,195,255,0.3);
    }

    .message-bubble.assistant {
        background: rgba(255,255,255,0.02);
    }

    .message-meta {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }

    .action-btn.copy {
        background: linear-gradient(120deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        color: var(--text-color);
        border-radius: 999px;
        padding: 0.4rem 1.4rem;
        cursor: pointer;
        font-size: 0.85rem;
        text-align: center;
        min-width: 150px;
    }

    .action-btn.copy:hover,
    .action-btn.copy.copied {
        border-color: var(--accent);
        color: var(--accent);
    }

    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulse 1s infinite ease-in-out;
    }

    .typing-dot:nth-child(2) { animation-delay: 0.15s; }
    .typing-dot:nth-child(3) { animation-delay: 0.3s; }

    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }

    .citation-card {
        background: rgba(255,255,255,0.02);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        margin: 0.6rem 0;
        display: flex;
        gap: 0.75rem;
    }

    .citation-badge {
        background: var(--accent-2);
        color: #111;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: rgba(8,9,13,0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .conversation-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.75rem;
        margin-bottom: 0.6rem;
        transition: border-color 0.2s ease;
        width: 100%;
    }

    .conversation-card button {
        width: 100%;
    }

    .conversation-card:hover {
        border-color: var(--accent);
    }

    .conversation-meta {
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    .conversation-row {
        display: flex;
        gap: 0.4rem;
        align-items: center;
    }

    .conversation-search input {
        background: rgba(0,0,0,0.6) !important;
        border-radius: 999px !important;
    }

    .danger-zone {
        border: 1px solid rgba(255,107,107,0.3);
        background: rgba(255,107,107,0.1);
        border-radius: 12px;
        padding: 0.75rem;
    }

    .settings-toggle label {
        font-size: 0.9rem;
    }

    @media (max-width: 768px) {
        .chat-stream {
            padding: 1rem;
        }
        .message-bubble {
            padding: 0.85rem;
        }
        .quick-chip {
            width: 100%;
            justify-content: center;
        }
    }
</style>

<script>
// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+K to focus input
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        const input = document.querySelector('textarea, input[type="text"]');
        if (input) input.focus();
    }

    // Enter in form to submit (handled by browser by default, but ensuring it works)
    if (e.key === 'Enter' && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {
        // In textareas: Enter submits; Shift+Enter makes a newline.
        if (e.target.tagName === 'TEXTAREA' && e.shiftKey) return;
        if (e.target.tagName === 'TEXTAREA') e.preventDefault();
        const form = e.target.closest('form');
        if (form) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) submitBtn.click();
        }
    }
});
</script>

""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Load conversations from disk on startup
if not st.session_state.conversations:
    st.session_state.conversations = load_conversations_from_disk()

sync_message_counter()
ensure_conversation_ids()
st.session_state.selected_conversations = [str(x) for x in st.session_state.selected_conversations]
ensure_active_conversation_loaded()


# Hero section with lightweight dropdown menu
st.markdown(
    """
    <div class="hero-title">
        <h1>Theo</h1>
        <p class="hero-subtitle">Ask questions about religious and philosophical texts from our PDF collection.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("##### Quick question templates")
template_cols = st.columns(2)
for idx, template in enumerate(QUICK_QUESTION_TEMPLATES):
    col = template_cols[idx % 2]
    with col:
        if st.button(f"Try: {template}", key=f"quick_template_{idx}"):
            st.session_state.pending_prefill = template

if st.session_state.pending_prefill:
    st.session_state.question_input = st.session_state.pending_prefill
    st.session_state.pending_prefill = ""

# Source filter panel
with st.expander("Focus Sources (Optional)", expanded=False):
    st.write("Select specific texts to focus your search:")
    available_sources = list(load_available_sources())

    if st.session_state.clear_sources_pending:
        st.session_state.selected_sources = []
        for tradition_texts in TEXT_COLLECTIONS.values():
            for text in tradition_texts:
                state_key = f"source_{text}"
                if state_key in st.session_state:
                    st.session_state[state_key] = False
        st.session_state.clear_sources_pending = False

    # Group sources by tradition for easier selection
    col1, col2 = st.columns(2)

    with col1:
        for tradition in list(TEXT_COLLECTIONS.keys())[:3]:
            st.write(f"**{tradition}**")
            for text in TEXT_COLLECTIONS[tradition]:
                if text in available_sources:
                    if st.checkbox(text, key=f"source_{text}"):
                        if text not in st.session_state.selected_sources:
                            st.session_state.selected_sources.append(text)
                    elif text in st.session_state.selected_sources:
                        st.session_state.selected_sources.remove(text)

    with col2:
        for tradition in list(TEXT_COLLECTIONS.keys())[3:]:
            st.write(f"**{tradition}**")
            for text in TEXT_COLLECTIONS[tradition]:
                if text in available_sources:
                    if st.checkbox(text, key=f"source_{text}"):
                        if text not in st.session_state.selected_sources:
                            st.session_state.selected_sources.append(text)
                    elif text in st.session_state.selected_sources:
                        st.session_state.selected_sources.remove(text)

    if st.session_state.selected_sources:
        st.write(f"**Selected sources:** {', '.join(st.session_state.selected_sources)}")
        if st.button("Clear All", key="clear_sources"):
            st.session_state.clear_sources_pending = True
            st.rerun()

    st.caption(f"{len(st.session_state.selected_sources)} selected - {len(available_sources)} available")

# Main chat interface
main_col = st.container()

with main_col:
    conversation = st.session_state.current_conversation
    settings = st.session_state.user_settings
    st.markdown("#### Conversation")
    chat_stream = st.container()
    answer_counter = 0

    with chat_stream:
        st.markdown("<div class='chat-stream'>", unsafe_allow_html=True)
        for idx, msg in enumerate(conversation):
            msg_id = msg.get("id") or f"{msg['type']}-{idx}"
            role = "You" if msg["type"] == "question" else "Theo"
            timestamp = format_timestamp(msg.get("timestamp")) if settings.get("show_timestamps", True) else ""
            bubble_class = "user" if msg["type"] == "question" else "assistant"
            body_html = sanitize_html(msg["content"])

            if msg["type"] == "answer":
                answer_counter += 1
                body_html = make_citations_clickable(body_html, answer_counter)

            meta = f"<span>{role}</span>"
            if timestamp:
                meta += f"<span>{timestamp}</span>"

            st.markdown(
                f"""
                <div class='message-bubble {bubble_class}' id='{msg_id}'>
                    <div class='message-meta'>{meta}</div>
                    <div class='message-body'>{body_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if msg["type"] == "answer":
                actions = st.columns([1, 1])
                with actions[0]:
                    copy_payload = encode_copy_payload(msg['content'])
                    st.markdown(
                        f"<button class='action-btn copy' data-copy-b64='{copy_payload}'>Copy</button>",
                        unsafe_allow_html=True,
                    )
                with actions[1]:
                    is_latest_answer = idx == len(conversation) - 1
                    if is_latest_answer:
                        if st.button("Regenerate", key=f"regen_{msg_id}", use_container_width=True):
                            conversation.pop()
                            regen_placeholder = st.empty()
                            regen_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)
                            try:
                                prior_question = msg.get("source_question")
                                if not prior_question and idx > 0 and conversation[idx - 1]["type"] == "question":
                                    prior_question = conversation[idx - 1]["content"]
                                regen_ok = run_answer_flow(prior_question or "", add_question=False)
                            finally:
                                regen_placeholder.empty()
                            if regen_ok:
                                st.rerun()
                    else:
                        st.caption("Regenerate latest answer only")

                if msg.get("citations"):
                    exp_label = f"Citations - Answer {answer_counter}"
                    with st.expander(exp_label, expanded=False):
                        for citation in msg["citations"]:
                            label = citation["index"]
                            display_name = format_title(citation["pdf"])
                            citation_id = f"citation-{answer_counter}-{label}"
                            citation_text_raw = f"{display_name}, Page {citation['page']}"
                            citation_payload = encode_copy_payload(citation_text_raw)
                            st.markdown(
                                f"""
                                <div class='citation-card' id='{sanitize_html(citation_id)}'>
                                    <span class='citation-badge'>#{sanitize_html(str(label))}</span>
                                    <div>
                                        <p class='citation-title'>{sanitize_html(display_name)} - Page {sanitize_html(str(citation['page']))}</p>
                                        <p class='citation-snippet'>{sanitize_html(citation['snippet'])}</p>
                                        <div class='copy-buttons'>
                                            <button class='copy-btn' data-copy-b64='{citation_payload}'>Copy Citation</button>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        st.markdown("</div>", unsafe_allow_html=True)

    typing_placeholder = st.empty()

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    # Input form
    if st.session_state.get("question_needs_reset"):
        st.session_state.question_input = ""
        st.session_state.question_needs_reset = False

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_area(
            "Continue the conversation...",
            key="question_input",
            placeholder="Ask a follow-up question or start a new topic",
            height=90,
        )
        submitted = st.form_submit_button("Send")

    # Process form submission
    if submitted and question:
        typing_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)
        with st.spinner("Theo is composing a reply..."):
            try:
                submit_ok = run_answer_flow(question)
            finally:
                typing_placeholder.empty()
        if submit_ok:
            st.session_state.question_needs_reset = True
            st.rerun()

auto_scroll_flag = "true" if st.session_state.user_settings.get("auto_scroll", True) else "false"
st.markdown(
    f"""
    <div id="chat-bottom"></div>
    <script>
        const autoScrollEnabled = {auto_scroll_flag};
        if (autoScrollEnabled) {{
            setTimeout(() => {{
                const bottom = document.getElementById('chat-bottom');
                if (bottom) bottom.scrollIntoView({{behavior: 'smooth', block: 'end'}});
            }}, 120);
        }}
        document.querySelectorAll('button[data-copy-b64]').forEach(btn => {{
            btn.addEventListener('click', () => {{
                const payload = btn.getAttribute('data-copy-b64');
                if (payload) {{
                    try {{
                        const text = atob(payload);
                        navigator.clipboard.writeText(text);
                        btn.classList.add('copied');
                        setTimeout(() => btn.classList.remove('copied'), 800);
                    }} catch (err) {{
                        console.error('Copy failed', err);
                    }}
                }}
            }});
        }});
    </script>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown("---")
st.caption("Theo - Powered by OpenAI API and FAISS")

# Sidebar conversations
with st.sidebar:
    st.markdown("### Conversations")
    search_query = st.text_input(
        "Search conversations",
        key="conversation_search",
        placeholder="Search by title or content",
    ).strip().lower()

    def matches_query(conv):
        if not search_query:
            return True
        title_match = search_query in conv.get("title", "").lower()
        if title_match:
            return True
        for entry in conv.get("messages", []):
            if search_query in entry.get("content", "").lower():
                return True
        return False

    matching_conversations = [conv for conv in st.session_state.conversations if matches_query(conv)]

    if matching_conversations:
        for idx, conv in enumerate(reversed(matching_conversations[-MAX_CONVERSATIONS:])):
            base_id = str(conv.get('id'))
            widget_suffix = f"{base_id}_{idx}"
            conv_key = widget_suffix
            message_total = len(conv.get("messages", []))
            timestamp_label = format_timestamp(conv.get("timestamp")) or "Unsaved draft"

            select_key = f"select_{widget_suffix}"
            checked = st.checkbox(
                "Select",
                key=select_key,
                value=base_id in st.session_state.selected_conversations,
                help="Select for batch delete",
            )
            if checked and base_id not in st.session_state.selected_conversations:
                st.session_state.selected_conversations.append(base_id)
            elif not checked and base_id in st.session_state.selected_conversations:
                st.session_state.selected_conversations.remove(base_id)

            title_text = conv.get('title', 'Conversation') or 'Conversation'
            st.markdown(f"**{sanitize_html(title_text)}**")
            st.caption(f"{timestamp_label} - {message_total} messages")

            if st.session_state.rename_mode == base_id:
                new_title = st.text_input(
                    "Rename conversation",
                    value=conv['title'],
                    key=f"rename_input_{widget_suffix}"
                )
                if st.button("Save name", key=f"save_{conv_key}", use_container_width=True):
                    conv['title'] = new_title
                    st.session_state.rename_mode = None
                    save_conversations_to_disk(st.session_state.conversations)
                if st.button("Cancel rename", key=f"cancel_{conv_key}", use_container_width=True):
                    st.session_state.rename_mode = None
            else:
                if st.button(
                    "Open conversation",
                    key=f"open_{conv_key}",
                    use_container_width=True,
                ):
                    st.session_state.current_conversation = conv["messages"].copy()
                    st.session_state.active_conversation_id = conv.get('id')
                    st.rerun()

                if st.button("Rename", key=f"rename_{conv_key}", use_container_width=True):
                    st.session_state.rename_mode = base_id

                if st.button("Delete", key=f"delete_{conv_key}", use_container_width=True):
                    removed = remove_conversation(base_id)
                    if base_id in st.session_state.selected_conversations:
                        st.session_state.selected_conversations.remove(base_id)
                    if removed and st.session_state.active_conversation_id is not None and str(st.session_state.active_conversation_id) == base_id:
                        st.session_state.active_conversation_id = None
                        st.session_state.current_conversation = []
                    save_conversations_to_disk(st.session_state.conversations)
                    st.rerun()

            st.markdown("---")
    else:
        st.markdown("*No conversations yet*")

    if st.session_state.selected_conversations:
        if st.button("Delete selected conversations", key="delete_selected"):
            st.session_state.pending_delete_confirmation = True

    if st.session_state.pending_delete_confirmation:
        st.warning("This action permanently deletes the selected conversations.")
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Confirm delete", key="confirm_batch_delete"):
                ids_to_remove = set(st.session_state.selected_conversations)
                st.session_state.conversations = [
                    conv for conv in st.session_state.conversations
                    if str(conv.get('id')) not in ids_to_remove
                ]
                if st.session_state.active_conversation_id and str(st.session_state.active_conversation_id) in ids_to_remove:
                    st.session_state.active_conversation_id = None
                    st.session_state.current_conversation = []
                st.session_state.selected_conversations = []
                st.session_state.pending_delete_confirmation = False
                save_conversations_to_disk(st.session_state.conversations)
                st.rerun()
        with cancel_col:
            if st.button("Cancel", key="cancel_batch_delete"):
                st.session_state.pending_delete_confirmation = False

    if st.button("Start New Conversation", key="sidebar_start_new", use_container_width=True):
        if st.session_state.active_conversation_id is not None:
            idx = find_conversation_index(st.session_state.active_conversation_id)
            if idx is not None and not st.session_state.conversations[idx].get("messages"):
                st.session_state.conversations.pop(idx)
                save_conversations_to_disk(st.session_state.conversations)
        st.session_state.current_conversation = []
        st.session_state.active_conversation_id = None
        st.session_state.rename_mode = None
        st.session_state.pending_prefill = ""
        st.session_state.last_error = None
        st.session_state.question_needs_reset = True
        st.rerun()

    with st.expander("Settings & Preferences"):
        settings = st.session_state.user_settings
        settings["show_timestamps"] = st.checkbox("Show message timestamps", value=settings.get("show_timestamps", True))
        settings["auto_scroll"] = st.checkbox("Auto-scroll to newest message", value=settings.get("auto_scroll", True))
        settings["compact_mode"] = st.checkbox("Compact layout", value=settings.get("compact_mode", False))
