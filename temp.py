import os
import sys
from dotenv import load_dotenv

load_dotenv()

ffmpeg_dir = os.getenv("FFMPEG_LOCATION")

if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
    os.environ["PATH"] = (
        ffmpeg_dir
        + os.pathsep
        + os.environ["PATH"]
    )

from utils.AudioProcessor import process_input, chunk_audio
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

source = "https://youtu.be/syFZfO_wfMQ?si=SkXVO04snHDnNXv7"

result = process_input(source)

if result["ok"]:
    audio_path = result["whisper_audio_path"]
    chunks = chunk_audio(audio_path, chunk_minutes=10)
    
    # Choose language: "english" (Whisper) or "hindi" / "hinglish" (Sarvam AI)
    language = "hinglish"
    # language = "english"
    transcript = transcribe_all(chunks, language=language)
    print("\n--- FINAL TRANSCRIPTION ---")
    print(transcript)

    title = generate_title(transcript)
    summary = summarize(transcript)

    print("\n" + "=" * 60)
    print(f"📌 TITLE: {title}")
    print("=" * 60)
    print("\n📋 SUMMARY")
    print("-" * 60)
    print(summary)


    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)


    print("\n" + "=" * 60)
    print("✅ ACTION ITEMS")
    print("=" * 60)
    print(action_items)

    print("\n" + "=" * 60)
    print("🔑 KEY DECISIONS")
    print("=" * 60)
    print(decisions)

    print("\n" + "=" * 60)
    print("❓ OPEN QUESTIONS")
    print("=" * 60)
    print(questions)
else:
    print(f"Error processing audio: {result['message']}")

