import os
import sys
from dotenv import load_dotenv

# Load .env first so FFMPEG_LOCATION is available
load_dotenv()

# Inject FFmpeg into PATH before any ffmpeg-dependent imports (mirrors temp.py)
ffmpeg_dir = os.getenv("FFMPEG_LOCATION")
if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

# Fix stdout encoding for emoji/Unicode output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from utils.AudioProcessor import process_input, chunk_audio
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Meeting Assistant")

    # process_input returns a dict with ok/audio_path/error info
    result = process_input(source)

    if not result["ok"]:
        print(f"Error processing source: {result['message']}")
        print(f"Suggestion: {result.get('suggestion', '')}")
        return {}

    audio_path = result["whisper_audio_path"]

    # Chunk the audio into 10-minute segments
    chunks = chunk_audio(audio_path, chunk_minutes=10)

    transcript = transcribe_all(chunks, language)
    print(f"Raw transcription (first 300 chars): {transcript[:300]}")

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_item = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
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

    if not result:
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

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
