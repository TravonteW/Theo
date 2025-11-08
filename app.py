#!/usr/bin/env python3
"""
Streamlit web interface for the PDF Q&A system.
"""

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

from ask import answer

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

def format_title(name: str) -> str:
    return name.removesuffix(".pdf")

def normalize_title(filename: str) -> str:
    return Path(filename).stem.replace("_", " ")

def sanitize_html(text: str) -> str:
    """Sanitize HTML content to prevent XSS attacks"""
    return html.escape(text)

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

def save_conversations_to_disk(conversations, max_conversations=50):
    """Save conversations to disk with size cap and rotation"""
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
        "rename_mode": None,  # stores the ID of conversation being renamed
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@st.cache_data
def load_available_sources():
    with Path("citations.json").open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    lookup = {}
    for info in data.values():
        original = info["source_pdf"]
        normalized = normalize_title(original)
        lookup[normalized] = original
    return lookup

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
        --bg-color: #050505;
        --panel-color: #101010;
        --panel-border: #2b2b2b;
        --accent: #4cc3ff;
        --accent-soft: #ffb347;
        --text-color: #f5f5f5;
        --text-muted: #c1c1c1;
    }

    .stApp {
        background: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter','Segoe UI',sans-serif;
    }

    body, .stMarkdown, .stText, .stTextInput label, .stCaption, .sidebar-card, .answer-card, .citation-card {
        color: var(--text-color) !important;
    }

    .stApp > main .block-container {
        max-width: 1100px;
        padding: 2.5rem 2rem 3.5rem;
        gap: 1.5rem;
        background: transparent;
    }

    .hero-title h1 {
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }

    .hero-title .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        color: var(--text-muted);
    }

    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 8px !important;
        border: 1px solid var(--panel-border) !important;
        padding: 0.5rem !important;
        margin-bottom: 0.25rem !important;
        color: var(--text-color) !important;
    }

    div[data-testid="stExpander"] > div[data-testid="stExpanderContent"] {
        background: var(--panel-color);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 10px 35px rgba(0,0,0,0.45);
    }

    .stTextInput > div > div,
    textarea {
        background: rgba(255,255,255,0.03);
        color: var(--text-color);
        border-radius: 12px;
        border: 1px solid var(--panel-border);
        padding: 0.6rem 0.85rem;
    }

    .stTextInput > div > div input {
        color: var(--text-color);
    }

    .stButton button {
        padding: 0.7rem 1rem;
        min-height: 42px;
        border-radius: 12px;
        background: linear-gradient(90deg, #1a1a1a, #252525);
        color: var(--text-color);
        border: 1px solid #3a3a3a;
        transition: border-color 0.2s ease, color 0.2s ease;
    }

    .stButton button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    .chat-message {
        background: var(--panel-color);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid var(--panel-border);
        box-shadow: 0 6px 20px rgba(0,0,0,0.45);
    }

    .chat-question {
        border: 1px solid var(--accent);
        background: #0f1a1e;
    }

    .answer-card {
        background: var(--panel-color);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--panel-border);
        box-shadow: 0 6px 24px rgba(0,0,0,0.6);
    }

    .citation-card {
        background: #0b0b0b;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #272727;
        margin: 0.6rem 0;
        display: flex;
        gap: 0.85rem;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
    }

    .citation-badge {
        background: var(--accent);
        color: #041016;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        flex-shrink: 0;
    }

    .citation-title {
        font-weight: 600;
        color: var(--text-color);
        margin: 0 0 0.35rem 0;
        font-size: 0.95rem;
    }

    .citation-snippet {
        color: var(--text-muted);
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .copy-buttons {
        margin-top: 0.6rem;
        display: flex;
        gap: 0.5rem;
    }

    .copy-btn {
        background: transparent;
        border: 1px solid var(--accent);
        color: var(--accent);
        border-radius: 6px;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: background 0.2s ease;
    }

    .copy-btn:hover {
        background: rgba(76,195,255,0.15);
    }

    .stSidebar,
    .stSidebar > div {
        background: #050505;
        color: var(--text-color);
    }

    .stSidebar .stButton button {
        width: 100%;
    }

    .stForm {
        margin-bottom: 1.25rem;
    }
</style>

<script>
// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+K to focus input
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        const input = document.querySelector('input[type="text"]');
        if (input) input.focus();
    }

    // Enter in form to submit (handled by browser by default, but ensuring it works)
    if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
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
source_lookup = load_available_sources()


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

# Lightweight dropdown for available texts
with st.expander("Browse Available Texts", expanded=False):
    available = source_lookup
    for tradition, titles in TEXT_COLLECTIONS.items():
        present = sorted(t for t in titles if t in available)
        if present:
            st.write(f"**{tradition}** ({len(present)} texts)")
            for title in present:
                st.write(f"- {title}")
    st.caption("Select texts in the Focus Sources section below to narrow your search.")

# Source filter panel
with st.expander("Focus Sources (Optional)", expanded=False):
    st.write("Select specific texts to focus your search:")
    available_sources = list(source_lookup.keys())

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
            st.session_state.selected_sources = []
            st.rerun()

# Main chat interface
main_col = st.container()

with main_col:
    # Display current conversation
    if st.session_state.current_conversation:
        st.markdown("#### Current Conversation")
        answer_counter = 0
        for msg in st.session_state.current_conversation:
            if msg["type"] == "question":
                st.markdown(
                    f"<div class='chat-message chat-question'><strong>You:</strong> {sanitize_html(msg['content'])}</div>",
                    unsafe_allow_html=True
                )
            else:
                answer_counter += 1
                # Make citations clickable in the answer
                clickable_content = make_citations_clickable(sanitize_html(msg['content']), answer_counter)
                st.markdown(
                    f"<div class='chat-message'><strong>Theo:</strong> {clickable_content}</div>",
                    unsafe_allow_html=True
                )

                # Show citations if available
                if msg.get("citations"):
                    expander_label = "Citations" + (" " * answer_counter)
                    with st.expander(expander_label, expanded=False):
                        for citation in msg["citations"]:
                            label = citation["index"]
                            display_name = format_title(citation["pdf"])
                            citation_id = f"citation-{answer_counter}-{label}"
                            # Prepare citation text for copying - fully sanitized
                            citation_text_raw = f"{display_name}, Page {citation['page']}"
                            safe_citation_js = sanitize_html(citation_text_raw).replace('"', '&quot;').replace("'", "&#39;")
                            st.markdown(
                                f"""
                                <div class='citation-card' id='{sanitize_html(citation_id)}'>
                                    <span class='citation-badge'>#{sanitize_html(str(label))}</span>
                                    <div>
                                        <p class='citation-title'>{sanitize_html(display_name)} - Page {sanitize_html(str(citation['page']))}</p>
                                        <p class='citation-snippet'>{sanitize_html(citation['snippet'])}</p>
                                        <div class='copy-buttons'>
                                            <button class='copy-btn' onclick='navigator.clipboard.writeText("{safe_citation_js}")'>Copy Citation</button>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Continue the conversation...", placeholder="Ask a follow-up question or start a new topic")
        submitted = st.form_submit_button("Send")

    # Process form submission
    if submitted and question:
        with st.spinner("Thinking..."):
            try:
                # Derive the history that matches the new schema
                history_messages = []
                for msg in st.session_state.current_conversation[-6:]:
                    if msg["type"] == "question":
                        history_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["type"] == "answer":
                        history_messages.append({"role": "assistant", "content": msg["content"]})

                # Pass selected sources for focused retrieval
                mapped_focus = []
                for name in st.session_state.selected_sources:
                    actual = source_lookup.get(name, name)
                    if actual not in mapped_focus:
                        mapped_focus.append(actual)
                focus_sources = mapped_focus if mapped_focus else None
                result = answer(question, conversation=history_messages, focus_sources=focus_sources)

                # Add question to current conversation
                st.session_state.current_conversation.append({
                    "type": "question",
                    "content": question
                })

                # Add answer to current conversation
                st.session_state.current_conversation.append({
                    "type": "answer",
                    "content": result["answer"],
                    "citations": result.get("citations", [])
                })

                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "not found" in str(e).lower():
                    st.info("Please make sure to run ingest.py and embed.py before asking questions.")

# Footer
st.markdown("---")
st.caption("Theo - Powered by OpenAI API and FAISS")

# Sidebar conversations
with st.sidebar:
    st.markdown("### Conversations")
    if st.session_state.conversations:
        for conv in reversed(st.session_state.conversations[-5:]):
            conv_id = conv['id']

            # Check if this conversation is in rename mode
            if st.session_state.rename_mode == conv_id:
                # Show rename input
                new_title = st.text_input(
                    "Rename:",
                    value=conv['title'],
                    key=f"rename_input_{conv_id}"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save", key=f"save_rename_{conv_id}", help="Save"):
                        # Update conversation title
                        for i, c in enumerate(st.session_state.conversations):
                            if c['id'] == conv_id:
                                st.session_state.conversations[i]['title'] = new_title
                                break
                        st.session_state.rename_mode = None
                        st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_rename_{conv_id}", help="Cancel"):
                        st.session_state.rename_mode = None
                        st.rerun()
            else:
                # Show normal conversation with controls
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    if st.button(f"Open {conv['title']}", key=f"conv_{conv_id}"):
                        st.session_state.current_conversation = conv["messages"].copy()
                        st.rerun()
                with col2:
                    if st.button("Rename", key=f"edit_{conv_id}", help="Rename"):
                        st.session_state.rename_mode = conv_id
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"delete_{conv_id}", help="Delete"):
                        st.session_state.conversations = [
                            c for c in st.session_state.conversations if c['id'] != conv_id
                        ]
                        st.rerun()
    else:
        st.markdown("*No previous conversations*")

    if st.button("Start New Conversation", key="sidebar_start_new"):
        if st.session_state.current_conversation:
            from datetime import datetime
            title = generate_conversation_title(st.session_state.current_conversation)
            new_conversation = {
                "id": st.session_state.conversation_counter,
                "title": title,
                "messages": st.session_state.current_conversation.copy(),
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.conversations.append(new_conversation)
            st.session_state.conversation_counter += 1
            # Save to disk with rotation
            save_conversations_to_disk(st.session_state.conversations)
        st.session_state.current_conversation = []
        st.rerun()
