import os
import re
import time
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --surface-3: #20202e;
    --border: #2a2a3a;
    --border-soft: #1e1e2a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --text-dim: #4d4d68;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.025) 1px, transparent 1px);
    background-size: 42px 42px;
    pointer-events: none;
    z-index: 0;
}

.block-container { padding-top: 2.5rem !important; max-width: 1180px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .block-container { padding-top: 2rem !important; }

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.card:hover { border-color: var(--accent); }
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-content { font-size: 0.875rem; line-height: 1.7; color: var(--text); white-space: pre-wrap; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important; }

/* ── Radio (segmented control look) ── */
[data-testid="stRadio"] > div { gap: 0.4rem; }
[data-testid="stRadio"] label {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.8rem !important;
    margin-right: 0.3rem;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--accent) !important;
    background: rgba(124,58,237,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover:not(:disabled) { transform: translateY(-1px) !important; box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important; }
.stButton > button:disabled { opacity: 0.35 !important; }
.stButton > button[kind="secondary"] { background: var(--surface-2) !important; border: 1px solid var(--border) !important; box-shadow: none !important; }
.stDownloadButton > button {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    text-transform: none !important;
}
.stDownloadButton > button:hover { border-color: var(--accent-2) !important; color: var(--accent-2) !important; }

/* ── Status / Expander (pipeline progress) ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.3rem;
    border-bottom: 1px solid var(--border);
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    background: transparent;
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-glow) !important;
    background: var(--surface) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.7rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; color: var(--accent-glow) !important; }

/* ── Chat (native st.chat_message) ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    padding: 0.4rem 0.6rem !important;
}
[data-testid="stChatMessage"] p { font-size: 0.87rem; line-height: 1.65; }
[data-testid="stChatInput"] textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
[data-testid="stChatInput"] { border-top: 1px solid var(--border) !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.4rem 0 !important; }

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 480px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Misc Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }
[data-testid="stCaptionContainer"] { color: var(--text-dim) !important; }

/* ── Process steps (sidebar, static) ── */
.step-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.35rem 0; font-size: 0.78rem; color: var(--text-muted); }
.step-num {
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--surface-2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.62rem; font-weight: 700; color: var(--accent-glow); flex-shrink: 0;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "analysed_at": None,
    "analysed_language": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Helpers ────────────────────────────────────────────────────────────────────
def count_list_items(text: str) -> int:
    """Roughly count numbered list entries in an LLM-generated list block."""
    if not text:
        return 0
    matches = re.findall(r"(?m)^\s*\d+[\.\)]\s", text)
    return len(matches)


def reset_session():
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.analysed_at = None
    st.session_state.analysed_language = None


def timed_step(status, label: str, fn, *args, **kwargs):
    """Run one pipeline stage, reporting how long it took inside the status widget."""
    status.write(f"{label}…")
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    status.write(f"↳ done in {elapsed:.1f}s")
    return result


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<span class="badge badge-purple">Source</span>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Source type", ["YouTube URL", "Upload File"],
        horizontal=True, label_visibility="collapsed",
    )

    source = None
    if input_mode == "YouTube URL":
        url = st.text_input(
            "YouTube URL", placeholder="https://youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        source = url.strip() if url and url.strip() else None
    else:
        uploaded = st.file_uploader(
            "Upload audio or video",
            type=["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "m4a", "ogg"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            tmp_dir = tempfile.mkdtemp(prefix="avi_upload_")
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source = tmp_path

    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    st.caption("Hinglish audio is transcribed *and* translated to English via Sarvam AI.")

    run_btn = st.button(
        "⚡  Analyse", use_container_width=True,
        disabled=source is None,
    )

    st.divider()

    if st.session_state.result:
        st.markdown('<span class="badge badge-green">Current Session</span>', unsafe_allow_html=True)
        st.markdown(f"**{st.session_state.result['title']}**")
        st.caption(f"{st.session_state.analysed_language} · analysed {st.session_state.analysed_at}")
        if st.button("↺  New Analysis", use_container_width=True, type="secondary"):
            reset_session()
            st.rerun()
    else:
        st.markdown('<span class="badge badge-cyan">How it works</span>', unsafe_allow_html=True)
        for i, label in enumerate([
            "Extract & chunk audio",
            "Transcribe speech",
            "Summarise & extract insights",
            "Chat over the transcript",
        ], start=1):
            st.markdown(
                f'<div class="step-row"><div class="step-num">{i}</div>{label}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption("Powered by Whisper · Mistral AI · ChromaDB · Sarvam AI")

# ─── Main Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Run Pipeline ───────────────────────────────────────────────────────────────
if run_btn and source:
    st.session_state.chat_history = []
    st.session_state.result = None

    with st.status("Running analysis pipeline...", expanded=True) as status:
        try:
            pipeline_start = time.perf_counter()

            chunks = timed_step(status, "🔊 Processing audio", process_input, source)
            transcript = timed_step(status, "📝 Transcribing", transcribe_all, chunks, language)
            title = timed_step(status, "🏷️ Generating title", generate_title, transcript)
            summary = timed_step(status, "📋 Summarising", summarize, transcript)
            extracted = timed_step(
                status, "🔍 Extracting action items, decisions & questions (parallel)",
                extract_all, transcript,
            )
            rag_chain = timed_step(status, "🧠 Building chat engine", build_rag_chain, transcript)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": extracted["action_items"],
                "key_decisions": extracted["key_decisions"],
                "open_questions": extracted["open_questions"],
                "rag_chain": rag_chain,
            }
            st.session_state.analysed_at = datetime.now().strftime("%b %d, %I:%M %p")
            st.session_state.analysed_language = language

            total = time.perf_counter() - pipeline_start
            status.update(label=f"✅ Analysis complete in {total:.1f}s", state="complete", expanded=False)

        except Exception as e:
            status.update(label=f"❌ Failed — {e}", state="error", expanded=True)

# ─── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words Transcribed", f"{len(r['transcript'].split()):,}")
    m2.metric("Action Items", count_list_items(r["action_items"]))
    m3.metric("Key Decisions", count_list_items(r["key_decisions"]))
    m4.metric("Open Questions", count_list_items(r["open_questions"]))

    st.markdown("<br>", unsafe_allow_html=True)

    tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions, tab_chat = st.tabs(
        ["📋 Summary", "📝 Transcript", "✅ Action Items", "🔑 Decisions", "❓ Questions", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(f'<div class="card-content">{r["summary"]}</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download Summary", r["summary"], file_name="summary.txt", use_container_width=False)

    with tab_transcript:
        st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download Transcript", r["transcript"], file_name="transcript.txt", use_container_width=False)

    with tab_actions:
        st.markdown(f'<div class="card-content">{r["action_items"]}</div>', unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(f'<div class="card-content">{r["key_decisions"]}</div>', unsafe_allow_html=True)

    with tab_questions:
        st.markdown(f'<div class="card-content">{r["open_questions"]}</div>', unsafe_allow_html=True)

    with tab_chat:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <div style="font-size:1.8rem;margin-bottom:0.5rem">💬</div>
                <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
            </div>""", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        if prompt := st.chat_input("What were the main decisions made?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking…"):
                    answer = ask_question(r["rag_chain"], prompt)
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:400px;line-height:1.7">
            Paste a YouTube URL or upload a file in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)
