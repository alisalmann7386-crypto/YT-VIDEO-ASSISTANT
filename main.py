from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()

def run_pipeline(source :str, language :str = "english") -> dict:
    print("starting AI Video Assistant")

    proc_type, proc_data, metadata = process_input(source)
    if proc_type == "FAST_TRANSCRIPT":
        transcript_data = proc_data
    else:
        chunks = proc_data
        transcript_data = transcribe_all(chunks, language)

    if isinstance(transcript_data, dict):
        full_transcript = transcript_data.get("full_text", "")
        segments = transcript_data.get("segments", [])
    else:
        full_transcript = transcript_data
        segments = []

    print(f"raw transcription (first 300 characters): {full_transcript[:300]}")

    title = generate_title(full_transcript)

    summary = summarize(full_transcript)

    action_item = extract_action_items(full_transcript)

    decisions = extract_key_decisions(full_transcript)
    questions = extract_questions(full_transcript)
    
    rag_chain = build_rag_chain(segments)

    return {
        "title": title,
        "transcript": full_transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")