import os
import re
import html
import time
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # must run before the core/utils imports below

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables ──────────────────────────────────────────────────────
   Warm studio palette, grounded in analogue recording gear rather than
   a generic dark-SaaS purple/cyan pairing: a toasted-charcoal surface,
   one amber "signal" accent (VU-meter / on-air lamp), one muted sage
   secondary for the conversational (RAG chat) parts of the product. */
:root {
    --bg: #14130f;
    --surface: #1b1912;
    --surface-2: #221f16;
    --surface-3: #29251a;
    --border: #362f1f;
    --border-soft: #262114;
    --accent: #d99a3f;
    --accent-strong: #f0b25c;
    --accent-ink: #241705;
    --signal: #7fa08f;
    --text: #ece5d6;
    --text-muted: #9c9484;
    --text-dim: #635c4a;
    --success: #7fae82;
    --warning: #d9b23f;
    --danger: #c96a54;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

/* Faint film-grain / tape-hiss texture instead of a neon grid — quiet
   enough to read as material, not decoration. */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: radial-gradient(rgba(217, 154, 63, 0.055) 1px, transparent 1px);
    background-size: 3px 3px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.5;
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
h1, h2, h3, h4, h5, h6 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: var(--text) !important; }

/* ── Accessibility: visible keyboard focus everywhere ── */
a:focus-visible, button:focus-visible, input:focus-visible, textarea:focus-visible,
[tabindex]:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}
@media (prefers-reduced-motion: reduce) {
    .waveform span { animation: none !important; }
    .stButton > button, .card { transition: none !important; }
}

/* ── Brand mark (sidebar) ── */
.brand-row { display: flex; align-items: center; gap: 0.7rem; }
.brand-mark {
    width: 34px; height: 34px; flex-shrink: 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface-2);
    display: flex; align-items: flex-end; justify-content: center;
    gap: 2.5px;
    padding: 8px 7px 6px;
}
.brand-mark span { display: block; width: 2.5px; border-radius: 1px; background: var(--accent); }
.brand-wordmark { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.05rem; line-height: 1.15; color: var(--text); }
.brand-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-dim); margin-top: 0.15rem; }

/* ── Waveform signature element ── */
.waveform { display: flex; align-items: flex-end; gap: 3px; }
.waveform span {
    display: block; width: 3px; border-radius: 2px;
    background: var(--accent); opacity: 0.6;
    transform-origin: bottom;
    animation: wave-pulse 2.6s ease-in-out infinite;
    animation-delay: calc(var(--i, 0) * 90ms);
}
@keyframes wave-pulse {
    0%, 100% { transform: scaleY(0.72); opacity: 0.4; }
    50% { transform: scaleY(1); opacity: 0.85; }
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(2rem, 5vw, 3.1rem);
    font-weight: 600;
    font-style: italic;
    line-height: 1.08;
    margin: 0;
    color: var(--text);
    letter-spacing: -0.01em;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: var(--text-muted);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-top: 0.55rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, transform 0.2s;
}
.card:hover { border-color: var(--border-soft); }
.card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-content { font-size: 0.9rem; line-height: 1.75; color: var(--text); white-space: pre-wrap; }

/* ── Eyebrow labels (replace colored pill badges) ── */
.eyebrow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}
.eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.eyebrow-accent { color: var(--accent-strong); }
.eyebrow-signal { color: var(--signal); }

/* ── Developer credit (sidebar footer) ── */
.dev-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 0.92rem; color: var(--text); margin-top: 0.15rem; }
.dev-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-dim); margin-top: 0.2rem; line-height: 1.5; }
.skill-tags { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.65rem; }
.skill-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.03em;
    color: var(--text-muted);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.18rem 0.6rem;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-soft, rgba(217,154,63,0.18)) !important; }

/* ── Radio (segmented control look) ── */
[data-testid="stRadio"] > div { gap: 0.4rem; }
[data-testid="stRadio"] label {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.35rem 0.8rem !important;
    margin-right: 0.3rem;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--accent) !important;
    background: rgba(217,154,63,0.12) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 9px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: var(--accent-ink) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.55rem 1.4rem !important;
    transition: background-color 0.15s, transform 0.15s !important;
}
.stButton > button:hover:not(:disabled) { background: var(--accent-strong) !important; border-color: var(--accent-strong) !important; transform: translateY(-1px) !important; }
.stButton > button:disabled { opacity: 0.35 !important; }
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover:not(:disabled) { background: var(--surface-2) !important; border-color: var(--text-dim) !important; }
.stDownloadButton > button {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
}
.stDownloadButton > button:hover { border-color: var(--accent) !important; color: var(--accent-strong) !important; }

/* ── Status / Expander (pipeline progress) ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
}
[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 1.4rem;
    border-bottom: 1px solid var(--border);
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.85rem;
    color: var(--text-muted);
    background: transparent;
    padding: 0.6rem 0.1rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text) !important;
    font-weight: 600 !important;
    background: transparent !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.9rem 1.1rem;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.68rem !important; letter-spacing: 0.1em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricValue"] { font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: var(--text) !important; }

/* ── Chat (native st.chat_message) ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 11px !important;
    padding: 0.4rem 0.6rem !important;
}
[data-testid="stChatMessage"] p { font-size: 0.88rem; line-height: 1.65; }
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
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.85;
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
[data-testid="stCaptionContainer"] { color: var(--text-dim) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.72rem !important; }

/* ── Process steps (sidebar) — styled as timecodes, since the content
   really is a timed sequence of pipeline stages ── */
.step-row { display: flex; align-items: center; gap: 0.65rem; padding: 0.4rem 0; font-size: 0.82rem; color: var(--text-muted); }
.step-num {
    min-width: 34px; height: 20px; border-radius: 4px;
    background: var(--surface-2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; font-weight: 500; color: var(--accent-strong); flex-shrink: 0;
    letter-spacing: 0.02em;
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
    st.markdown("""
    <div class="brand-row">
        <div class="brand-mark">
            <span style="height:8px"></span><span style="height:16px"></span>
            <span style="height:11px"></span><span style="height:20px"></span>
        </div>
        <div>
            <div class="brand-wordmark">AI Video Assistant</div>
            <div class="brand-eyebrow">Meeting Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="eyebrow eyebrow-accent"><span class="dot"></span>Source</div>', unsafe_allow_html=True)
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
        "Analyse", use_container_width=True,
        disabled=source is None,
    )

    st.divider()

    if st.session_state.result:
        st.markdown('<div class="eyebrow eyebrow-signal"><span class="dot"></span>Current session</div>', unsafe_allow_html=True)
        st.markdown(f"**{html.escape(st.session_state.result['title'])}**")
        st.caption(f"{st.session_state.analysed_language} · analysed {st.session_state.analysed_at}")
        if st.button("New analysis", use_container_width=True, type="secondary"):
            reset_session()
            st.rerun()
    else:
        st.markdown('<div class="eyebrow"><span class="dot"></span>How it works</div>', unsafe_allow_html=True)
        for i, label in enumerate([
            "Extract & chunk audio",
            "Transcribe speech",
            "Summarise & extract insights",
            "Chat over the transcript",
        ]):
            st.markdown(
                f'<div class="step-row"><div class="step-num">{i:02d}</div>{label}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption("Powered by Whisper · Mistral AI · ChromaDB · Sarvam AI")

    st.divider()
    st.markdown("""
    <div class="eyebrow" style="margin-bottom:0.2rem"><span class="dot"></span>Built by</div>
    <div class="dev-name">Md Salman Ali</div>
    <div class="dev-meta">B.Tech CSE · Data Science<br>Jamia Millia Islamia</div>
    <div class="skill-tags">
        <span class="skill-tag">Python</span>
        <span class="skill-tag">ML/AI</span>
        <span class="skill-tag">GenAI</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">AI Video Assistant</div>
<div class="waveform" aria-hidden="true" style="margin:0.9rem 0 0.7rem">
    <span style="height:6px;--i:0"></span><span style="height:13px;--i:1"></span>
    <span style="height:20px;--i:2"></span><span style="height:10px;--i:3"></span>
    <span style="height:24px;--i:4"></span><span style="height:16px;--i:5"></span>
    <span style="height:8px;--i:6"></span><span style="height:19px;--i:7"></span>
    <span style="height:22px;--i:8"></span><span style="height:11px;--i:9"></span>
    <span style="height:6px;--i:10"></span><span style="height:15px;--i:11"></span>
    <span style="height:23px;--i:12"></span><span style="height:9px;--i:13"></span>
    <span style="height:17px;--i:14"></span><span style="height:12px;--i:15"></span>
</div>
<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Run Pipeline ───────────────────────────────────────────────────────────────
if run_btn and source:
    st.session_state.chat_history = []
    st.session_state.result = None

    with st.status("Running analysis pipeline...", expanded=True) as status:
        try:
            pipeline_start = time.perf_counter()

            chunks = timed_step(status, "Processing audio", process_input, source)
            transcript = timed_step(status, "Transcribing", transcribe_all, chunks, language)
            title = timed_step(status, "Generating title", generate_title, transcript)
            summary = timed_step(status, "Summarising", summarize, transcript)
            extracted = timed_step(
                status, "Extracting action items, decisions & questions",
                extract_all, transcript,
            )
            rag_chain = timed_step(status, "Building chat engine", build_rag_chain, transcript)

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
            status.update(label=f"Analysis complete — {total:.1f}s", state="complete", expanded=False)

        except Exception as e:
            status.update(label=f"Failed — {e}", state="error", expanded=True)

# ─── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Session title</div>
        <div style="font-family:'Fraunces',serif;font-size:1.4rem;font-weight:600;color:var(--text)">
            {html.escape(r['title'])}
        </div>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words Transcribed", f"{len(r['transcript'].split()):,}")
    m2.metric("Action Items", count_list_items(r["action_items"]))
    m3.metric("Key Decisions", count_list_items(r["key_decisions"]))
    m4.metric("Open Questions", count_list_items(r["open_questions"]))

    st.markdown("<br>", unsafe_allow_html=True)

    tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions, tab_chat = st.tabs(
        ["Summary", "Transcript", "Action Items", "Decisions", "Questions", "Chat"]
    )

    with tab_summary:
        st.markdown(f'<div class="card-content">{html.escape(r["summary"])}</div>', unsafe_allow_html=True)
        st.download_button("Download summary", r["summary"], file_name="summary.txt", use_container_width=False)

    with tab_transcript:
        st.markdown(f'<div class="transcript-box">{html.escape(r["transcript"])}</div>', unsafe_allow_html=True)
        st.download_button("Download transcript", r["transcript"], file_name="transcript.txt", use_container_width=False)

    with tab_actions:
        st.markdown(f'<div class="card-content">{html.escape(r["action_items"])}</div>', unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(f'<div class="card-content">{html.escape(r["key_decisions"])}</div>', unsafe_allow_html=True)

    with tab_questions:
        st.markdown(f'<div class="card-content">{html.escape(r["open_questions"])}</div>', unsafe_allow_html=True)

    with tab_chat:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <div class="eyebrow" style="justify-content:center;margin-bottom:0.3rem"><span class="dot"></span>Chat</div>
                <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript.</div>
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
            if st.button("Clear chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div class="waveform" aria-hidden="true" style="height:34px;margin-bottom:1.6rem">
            <span style="height:10px;--i:0"></span><span style="height:22px;--i:1"></span>
            <span style="height:14px;--i:2"></span><span style="height:32px;--i:3"></span>
            <span style="height:18px;--i:4"></span><span style="height:26px;--i:5"></span>
            <span style="height:12px;--i:6"></span><span style="height:20px;--i:7"></span>
        </div>
        <div style="font-family:'Fraunces',serif;font-style:italic;font-size:1.5rem;font-weight:600;color:var(--text);margin-bottom:0.6rem">
            Add a source to begin
        </div>
        <div style="color:var(--text-muted);font-size:0.88rem;max-width:400px;line-height:1.7">
            Paste a YouTube link or drop an audio or video file in the sidebar, pick a language, then press <strong>Analyse</strong>.
        </div>
        <div class="brand-eyebrow" style="margin-top:2rem;letter-spacing:0.14em">
            Transcript · Summary · Action items · Decisions · Chat
        </div>
    </div>""", unsafe_allow_html=True)