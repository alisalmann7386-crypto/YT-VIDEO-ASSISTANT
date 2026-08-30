import os
import streamlit as st
from dotenv import load_dotenv

# 1. Page Configuration
st.set_page_config(
    page_title="YouTube AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS - Dark & Modern Theme Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0D1117;
        color: #E6EDF3;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0D1117;
        border-bottom: 1px solid #30363D;
        padding-bottom: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #161B22;
        border-radius: 8px 8px 0px 0px;
        color: #8B949E;
        border: 1px solid #30363D;
        padding-left: 18px;
        padding-right: 18px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FF0000 !important;
        color: #FFFFFF !important;
        font-weight: 700;
        border: 1px solid #FF0000 !important;
        box-shadow: 0 4px 12px rgba(255, 0, 0, 0.3);
    }

    .stButton > button {
        background-color: #FF0000;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(255, 0, 0, 0.25);
    }

    .stButton > button:hover {
        background-color: #CC0000;
        color: #FFFFFF;
        box-shadow: 0 4px 14px rgba(255, 0, 0, 0.4);
        transform: translateY(-1px);
    }

    .yt-red {
        color: #FF0000;
        font-weight: 800;
    }
    
    .glass-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .stChatInput textarea {
        color: #111111 !important;
        background-color: #FFFFFF !important;
    }
    
    .stChatInput textarea::placeholder {
        color: #666666 !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #1E2631 !important;
        color: #FFFFFF !important;
        border-radius: 8px;
    }

    [data-testid="stVerticalBlock"] > div > div > ul {
        padding-left: 20px;
        margin-top: 10px;
    }
    [data-testid="stVerticalBlock"] > div > div > ul > li {
        margin-bottom: 10px;
        line-height: 1.5;
        color: #E6EDF3;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Environment Variables & Backend Modules
load_dotenv()

try:
    if hasattr(st, "secrets"):
        for key, val in st.secrets.items():
            if isinstance(val, str) and key not in os.environ:
                os.environ[key] = val
except Exception:
    pass

from utils.audio_processor import process_input, process_text_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_insights
from core.rag_engine import build_rag_chain, ask_question

# 4. Initialize Session State Variables
if "processed" not in st.session_state:
    st.session_state.processed = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Helper function to execute pipeline
def execute_pipeline(input_mode, url_source="", raw_text="", file_source="", lang="english"):
    with st.status("⚙️ Executing AI Processing Pipeline...", expanded=True) as status:
        try:
            if input_mode == "text":
                st.write("📥 Step 1/4: Structuring transcript text...")
                transcript_data, metadata = process_text_input(raw_text)
            elif input_mode == "file":
                st.write("📥 Step 1/4: Processing local audio/video file...")
                proc_type, proc_data, metadata = process_input(file_source)
                chunks = proc_data
                st.write(f"🎙️ Step 2/4: Transcribing {len(chunks)} chunk(s) via {lang.upper()} engine...")
                transcript_data = transcribe_all(chunks, language=lang)
            else: # YouTube URL
                st.write("📥 Step 1/4: Extracting YouTube video transcript & metadata...")
                proc_type, proc_data, metadata = process_input(url_source)
                if proc_type == "FAST_TRANSCRIPT":
                    st.write("⚡ Step 2/4: Subtitles extracted instantly via YouTube API!")
                    transcript_data = proc_data
                else:
                    chunks = proc_data
                    st.write(f"🎙️ Step 2/4: Transcribing {len(chunks)} audio chunk(s) via {lang.upper()} engine...")
                    transcript_data = transcribe_all(chunks, language=lang)

            if isinstance(transcript_data, dict):
                full_transcript = transcript_data.get("full_text", "")
                segments = transcript_data.get("segments", [])
            else:
                full_transcript = transcript_data
                segments = []

            st.write("🧠 Step 3/4: Generating title, summary, and insights via Mistral AI...")
            title = generate_title(full_transcript)
            summary = summarize(full_transcript)
            insights = extract_insights(full_transcript)

            st.write("⚡ Step 4/4: Indexing transcript into FAISS Vector Database...")
            rag_chain = build_rag_chain(segments)

            st.session_state.analysis_data = {
                "title": title,
                "metadata": metadata,
                "full_transcript": full_transcript,
                "segments": segments,
                "summary": summary,
                "action_items": insights["action_items"],
                "decisions": insights["key_decisions"],
                "questions": insights["open_questions"],
                "rag_chain": rag_chain,
            }
            st.session_state.processed = True
            st.session_state.chat_history = []
            status.update(label="✅ Processing Complete!", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ Pipeline Execution Issue", state="error", expanded=True)
            st.error(f"Error encountered: {str(e)}")

# 5. Sidebar Layout & Controls
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <svg height="35" viewBox="0 0 28 20" width="45" xmlns="http://www.w3.org/2000/svg">
                <path d="M27.9727 3.12324C27.6435 1.89323 26.6768 0.926623 25.4468 0.597366C23.2197 0 14 0 14 0C14 0 4.78027 0 2.55317 0.597366C1.32316 0.926623 0.356555 1.89323 0.0272986 3.12324C0 5.35034 0 10 0 10C0 10 0 14.6497 0.0272986 16.8768C0.356555 18.1068 1.32316 19.0734 2.55317 19.4026C4.78027 20 14 20 14 20C4.78027 20 23.2197 20 25.4468 19.4026C26.6768 19.0734 27.9727 18.1068 27.9727 16.8768C28 14.6497 28 10 28 10C28 10 28 5.35034 27.9727 3.12324Z" fill="#FF0000"/>
                <path d="M11.2 14.2857L18.4 10L11.2 5.71429V14.2857Z" fill="white"/>
            </svg>
            <span style="font-size: 20px; font-weight: bold; color: white;">Control Panel</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    language = st.selectbox(
        "Audio Language Engine",
        options=["english", "hinglish"],
        index=0,
        help="Select 'english' for Whisper or 'hinglish' for Sarvam AI."
    )

    if st.session_state.processed:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.processed = False
            st.session_state.analysis_data = {}
            st.session_state.chat_history = []
            st.rerun()

# 6. Main Dashboard Header
st.markdown("""
    <div style="background: linear-gradient(135deg, #161B22 0%, #0D1117 100%); padding: 25px; border-radius: 12px; border: 1px solid #30363D; margin-bottom: 25px;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <svg height="45" viewBox="0 0 28 20" width="60" xmlns="http://www.w3.org/2000/svg">
                <path d="M27.9727 3.12324C27.6435 1.89323 26.6768 0.926623 25.4468 0.597366C23.2197 0 14 0 14 0C14 0 4.78027 0 2.55317 0.597366C1.32316 0.926623 0.356555 1.89323 0.0272986 3.12324C0 5.35034 0 10 0 10C0 10 0 14.6497 0.0272986 16.8768C0.356555 18.1068 1.32316 19.0734 2.55317 19.4026C4.78027 20 14 20 14 20C4.78027 20 23.2197 20 25.4468 19.4026C26.6768 19.0734 27.9727 18.1068 27.9727 16.8768C28 14.6497 28 10 28 10C28 10 28 5.35034 27.9727 3.12324Z" fill="#FF0000"/>
                <path d="M11.2 14.2857L18.4 10L11.2 5.71429V14.2857Z" fill="white"/>
            </svg>
            <div>
                <h1 style="margin: 0; padding: 0; font-size: 32px; font-weight: 800; color: #FFFFFF;">
                    YouTube <span class="yt-red">AI Assistant</span>
                </h1>
                <p style="margin: 5px 0 0 0; color: #8B949E; font-size: 14px;">
                    Turn YouTube videos, transcripts, and meeting audio into executive summaries, key insights, and interactive RAG chats.
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 7. Input Screen UI (Rendered when no analysis is active)
if not st.session_state.processed:
    st.markdown("### 🎯 Select Input Source")
    
    main_tab_url, main_tab_text, main_tab_file = st.tabs([
        "🔗 YouTube Video Link",
        "📝 Paste Transcript / Text",
        "📁 Upload Local File"
    ])
    
    # Tab 1: YouTube Link
    with main_tab_url:
        st.markdown("##### Process YouTube URL")
        yt_input = st.text_input(
            "YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            key="main_yt_url"
        )
        if st.button("🚀 Analyze YouTube Video", key="btn_run_yt", use_container_width=True):
            if yt_input.strip():
                execute_pipeline("url", url_source=yt_input.strip(), lang=language)
            else:
                st.warning("Please enter a valid YouTube Video URL.")

    # Tab 2: Paste Transcript / Text
    with main_tab_text:
        st.markdown("##### Paste Video Transcript or Meeting Notes")
        text_input = st.text_area(
            "Paste Transcript Text:",
            height=200,
            placeholder="Paste YouTube transcript, video text, or meeting notes here...",
            key="main_text_area"
        )
        if st.button("🧠 Analyze Transcript Text", key="btn_run_text", use_container_width=True):
            if text_input.strip():
                execute_pipeline("text", raw_text=text_input.strip(), lang=language)
            else:
                st.warning("Please paste transcript text before analyzing.")

    # Tab 3: Upload Local File
    with main_tab_file:
        st.markdown("##### Upload Local Audio or Video File")
        uploaded_media = st.file_uploader(
            "Select Audio/Video File:",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "webm"],
            key="main_file_uploader"
        )
        if st.button("🎙️ Process Uploaded Media", key="btn_run_file", use_container_width=True):
            if uploaded_media is not None:
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                saved_path = os.path.join(temp_dir, uploaded_media.name)
                with open(saved_path, "wb") as f:
                    f.write(uploaded_media.getbuffer())
                execute_pipeline("file", file_source=saved_path, lang=language)
            else:
                st.warning("Please select a file to upload.")

# 8. Content Dashboard Render (Rendered when analysis is complete)
else:
    data = st.session_state.analysis_data
    meta = data.get("metadata", {})

    with st.container():
        col_thumb, col_info = st.columns([1, 2])
        
        with col_thumb:
            if meta.get("thumbnail"):
                st.image(meta["thumbnail"], use_container_width=True)
            else:
                st.markdown("🎥 **Local File / Text Input**")
                
        with col_info:
            st.markdown(f"## {data['title']}")
            st.markdown(f"**Channel / Author:** `{meta.get('channel', 'N/A')}`")
            st.markdown(f"⏱️ **Duration:** `{meta.get('duration', 'N/A')}`")
            if meta.get("url"):
                st.markdown(f"🔗 [Watch on YouTube]({meta['url']})")

    st.markdown("---")

    tab_summary, tab_insights, tab_transcript, tab_chat = st.tabs([
        "📋 Executive Summary", 
        "💡 Key Insights", 
        "📝 Transcript", 
        "💬 Chat with Video (RAG)"
    ])
    
    with tab_summary:
        st.markdown("### Executive Summary")
        st.markdown(data["summary"])
        st.download_button(
            label="💾 Export Summary (.txt)",
            data=f"TITLE: {data['title']}\n\nSUMMARY:\n{data['summary']}",
            file_name="video_summary.txt",
            mime="text/plain"
        )

    with tab_insights:
        col1, col2, col3 = st.columns(3)
        def parse_to_list(raw_input):
            if isinstance(raw_input, list):
                return raw_input
            if isinstance(raw_input, str):
                return [item.strip("• ").strip() for item in raw_input.split("\n") if item.strip()]
            return []

        with col1:
            with st.container(border=True):
                st.markdown("### ✅ Action Items")
                items = parse_to_list(data.get("action_items", []))
                for item in items:
                    st.markdown(f"- {item}")
            
        with col2:
            with st.container(border=True):
                st.markdown("### 🔑 Key Decisions")
                decisions = parse_to_list(data.get("decisions", []))
                for decision in decisions:
                    st.markdown(f"- {decision}")
            
        with col3:
            with st.container(border=True):
                st.markdown("### ❓ Open Questions")
                questions = parse_to_list(data.get("questions", []))
                for question in questions:
                    st.markdown(f"- {question}")

    with tab_transcript:
        st.markdown("### Video Transcript")
        segments = data.get("segments", [])
        if segments:
            view_mode = st.radio("Display Mode:", ["Timestamped Sentences", "Full Text Paragraph"], horizontal=True)
            st.markdown("---")
            if view_mode == "Timestamped Sentences":
                for seg in segments:
                    col_time, col_text = st.columns([1, 5])
                    with col_time:
                        st.caption(f"⏱️ `{seg.get('start', '00:00')} - {seg.get('end', '00:00')}`")
                    with col_text:
                        st.write(seg.get("text", ""))
            else:
                st.text_area("Full Text", data.get("full_transcript", ""), height=400)
        else:
            st.text_area("Full Transcript", data.get("full_transcript", ""), height=400)
            
        st.download_button(
            label="💾 Export Transcript (.txt)",
            data=data.get("full_transcript", ""),
            file_name="transcript.txt",
            mime="text/plain"
        )

    with tab_chat:
        st.markdown("### 💬 Chat with Video")
        st.caption("Ask questions about the content. Answers are drawn directly from the transcript.")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        if user_query := st.chat_input("Ask anything about this video..."):
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
                
            with st.chat_message("assistant"):
                with st.spinner("Searching video context..."):
                    response = ask_question(data["rag_chain"], user_query)
                    st.write(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})