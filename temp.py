import sys
from dotenv import load_dotenv
from utils.AudioProcessor import process_input, chunk_audio
from core.transcriber import transcribe_all

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

source = "https://youtu.be/O4GorB4N2Kg?si=WGACDZqf20VRcLRX"

result = process_input(source)

if result["ok"]:
    audio_path = result["whisper_audio_path"]
    chunks = chunk_audio(audio_path, chunk_minutes=10)
    
    # Choose language: "english" (Whisper) or "hindi" / "hinglish" (Sarvam AI)
    language = "hinglish"
    transcript = transcribe_all(chunks, language=language)
    print("\n--- FINAL TRANSCRIPTION ---")
    print(transcript)
else:
    print(f"Error processing audio: {result['message']}")

