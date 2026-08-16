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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #08090b;
    --surface: #111318;
    --surface-2: #181a20;
    --surface-3: #20232b;
    --border: #292d36;
    --text: #f5f5f5;
    --muted: #a1a1aa;
    --red: #ff0033;
    --red-dark: #cc0029;
    --red-soft: rgba(255, 0, 51, .13);
    --green: #22c55e;
    --blue: #38bdf8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background:
        radial-gradient(circle at 75% -10%, rgba(255,0,51,.10), transparent 30%),
        radial-gradient(circle at 5% 30%, rgba(56,189,248,.045), transparent 25%),
        var(--bg) !important;
}

.block-container {
    max-width: 1250px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

[data-testid="stSidebar"] {
    background: #0d0f13 !important;
    border-right: 1px solid #22252c !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.15rem !important;
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

h1,h2,h3,h4,h5,h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

/* Brand */
.brand {
    display:flex;
    align-items:center;
    gap:.75rem;
    margin-bottom:.25rem;
}
.brand-icon {
    width:42px;
    height:30px;
    border-radius:9px;
    background:var(--red);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:17px;
    box-shadow:0 8px 25px rgba(255,0,51,.25);
}
.brand-name {
    font-family:'Space Grotesk',sans-serif;
    font-size:1.25rem;
    font-weight:700;
}
.brand-sub {
    margin-left:54px;
    color:var(--muted);
    font-size:.72rem;
    margin-bottom:1.5rem;
}

/* Header */
.topbar {
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:1.4rem;
}
.hero-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(2rem,4vw,3rem);
    font-weight:700;
    letter-spacing:-.04em;
    margin:0;
}
.hero-title span { color:var(--red); }
.hero-sub {
    color:var(--muted);
    font-size:.88rem;
    margin-top:.35rem;
}
.live-pill {
    display:inline-flex;
    align-items:center;
    gap:.4rem;
    padding:.4rem .7rem;
    border:1px solid #30343d;
    background:#111318;
    border-radius:999px;
    font-size:.7rem;
    color:#d4d4d8;
}
.live-dot {
    width:7px;height:7px;border-radius:50%;
    background:var(--green);
    box-shadow:0 0 10px rgba(34,197,94,.7);
}

/* Cards */
.card {
    background:linear-gradient(145deg,#14161b,#101216);
    border:1px solid var(--border);
    border-radius:14px;
    padding:1.25rem 1.35rem;
    margin-bottom:1rem;
    box-shadow:0 12px 35px rgba(0,0,0,.18);
}
.card:hover { border-color:#3a3e48; }
.card-title {
    display:flex;
    align-items:center;
    gap:.5rem;
    color:#a1a1aa;
    font-size:.72rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.08em;
    margin-bottom:.7rem;
}
.card-content {
    color:#e4e4e7;
    font-size:.92rem;
    line-height:1.8;
}

/* YouTube source panel */
.source-card {
    background:linear-gradient(135deg,rgba(255,0,51,.08),#111318 55%);
    border:1px solid #3a2027;
    border-radius:16px;
    padding:1.25rem;
    margin-bottom:1.2rem;
}
.source-label {
    color:#fca5b5;
    font-size:.7rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.1em;
    margin-bottom:.55rem;
}
.url-preview {
    display:flex;
    align-items:center;
    gap:.65rem;
    background:#0b0c0f;
    border:1px solid #292d36;
    border-radius:10px;
    padding:.75rem .9rem;
    color:#d4d4d8;
    font-size:.82rem;
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {
    background:#15171c !important;
    border:1px solid #30343d !important;
    border-radius:9px !important;
    color:#fff !important;
}
.stTextInput > div > div > input:focus {
    border-color:var(--red) !important;
    box-shadow:0 0 0 2px var(--red-soft) !important;
}

/* Buttons */
.stButton > button {
    background:var(--red) !important;
    color:white !important;
    border:0 !important;
    border-radius:9px !important;
    font-weight:700 !important;
    min-height:42px !important;
    transition:.18s ease !important;
}
.stButton > button:hover {
    background:var(--red-dark) !important;
    transform:translateY(-1px);
    box-shadow:0 8px 25px rgba(255,0,51,.25) !important;
}
.stButton > button[kind="secondary"] {
    background:#181a20 !important;
    border:1px solid #30343d !important;
    color:#e4e4e7 !important;
}
.stDownloadButton > button {
    background:#181a20 !important;
    border:1px solid #30343d !important;
    color:#e4e4e7 !important;
    border-radius:9px !important;
}

/* Radio */
[data-testid="stRadio"] > div { gap:.35rem; }
[data-testid="stRadio"] label {
    background:#15171c;
    border:1px solid #30343d;
    border-radius:8px;
    padding:.35rem .65rem !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background:var(--red-soft) !important;
    border-color:var(--red) !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background:#15171c !important;
    border:1px dashed #3a3e48 !important;
    border-radius:10px !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap:.25rem;
    border-bottom:1px solid #292d36;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family:'Inter',sans-serif !important;
    font-weight:600;
    color:#8f939c;
    border-radius:8px 8px 0 0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color:#fff !important;
    border-bottom:2px solid var(--red) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background:#111318;
    border:1px solid #292d36;
    border-radius:12px;
    padding:.9rem 1rem;
}
[data-testid="stMetricLabel"] {
    color:#8f939c !important;
    font-size:.7rem !important;
}
[data-testid="stMetricValue"] {
    color:#fff !important;
    font-family:'Space Grotesk',sans-serif !important;
}

/* Transcript */
.transcript-box {
    background:#0d0f13;
    border:1px solid #292d36;
    border-radius:10px;
    padding:1.2rem;
    font-size:.85rem;
    line-height:1.85;
    max-height:520px;
    overflow-y:auto;
    color:#c8cbd1;
    white-space:pre-wrap;
}

/* Chat */
[data-testid="stChatMessage"] {
    background:#111318 !important;
    border:1px solid #292d36 !important;
    border-radius:12px !important;
}
[data-testid="stChatInput"] textarea {
    background:#15171c !important;
    color:#fff !important;
    border:1px solid #30343d !important;
}

/* Status */
[data-testid="stExpander"] {
    background:#111318 !important;
    border:1px solid #292d36 !important;
    border-radius:12px !important;
}
.stProgress > div > div > div { background:var(--red) !important; }
.stSpinner > div { border-top-color:var(--red) !important; }

.badge {
    display:inline-flex;
    padding:.28rem .6rem;
    border-radius:999px;
    font-size:.65rem;
    font-weight:700;
    border:1px solid #30343d;
    background:#181a20;
    color:#bfc2c9;
}
.badge-red { color:#ff7d96; border-color:#572431; background:var(--red-soft); }
.badge-green { color:#86efac; border-color:#245c39; background:rgba(34,197,94,.10); }
.badge-blue { color:#7dd3fc; border-color:#21495c; background:rgba(56,189,248,.08); }

.step-row {
    display:flex;
    align-items:center;
    gap:.65rem;
    padding:.45rem 0;
    color:#a1a1aa;
    font-size:.76rem;
}
.step-num {
    width:22px;height:22px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    background:#181a20;border:1px solid #30343d;
    color:#ff7189;font-size:.65rem;font-weight:700;
}

.empty-state {
    text-align:center;
    padding:5rem 1rem;
}
.empty-icon {
    width:74px;height:74px;
    margin:0 auto 1.2rem;
    border-radius:18px;
    background:var(--red-soft);
    border:1px solid #572431;
    display:flex;align-items:center;justify-content:center;
    font-size:2rem;
}
.feature-row {
    display:flex;
    justify-content:center;
    gap:.5rem;
    flex-wrap:wrap;
    margin-top:1.4rem;
}

hr { border-top:1px solid #252830 !important; }

label { color:#a1a1aa !important; }
[data-testid="stCaptionContainer"] { color:#71717a !important; }
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
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">▶</div>
        <div class="brand-name">Video<span style="color:#ff0033">AI</span></div>
    </div>
    <div class="brand-sub">YOUTUBE VIDEO ASSISTANT</div>
    """, unsafe_allow_html=True)
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
st.markdown("""
<div class="topbar">
    <div>
        <div class="hero-title">YouTube <span>Video Assistant</span></div>
        <div class="hero-sub">Turn long videos into summaries, insights and an AI-powered conversation.</div>
    </div>
    <div class="live-pill"><span class="live-dot"></span> AI READY</div>
</div>
""", unsafe_allow_html=True)

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
    <div class="empty-state">
        <div class="empty-icon">▶</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.65rem;font-weight:700;margin-bottom:.5rem">
            Your video workspace is ready
        </div>
        <div style="color:#8f939c;font-size:.88rem;max-width:520px;margin:auto;line-height:1.7">
            Paste a YouTube link in the sidebar and let VideoAI turn the content into a searchable knowledge base.
        </div>
        <div class="feature-row">
            <span class="badge badge-red">● Transcription</span>
            <span class="badge badge-blue">✦ AI Summary</span>
            <span class="badge badge-green">⌁ RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)