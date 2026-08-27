import os
import wave
from typing import Optional

import requests
import whisper
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Load variables from .env file
load_dotenv()

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
SARVAM_PIECE_SECONDS = int(os.getenv("SARVAM_PIECE_SECONDS", "25"))

_whisper_model = None
_sarvam_client = None


def load_whisper_model():
    """Load the Whisper model lazily if not already loaded."""
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model '{WHISPER_MODEL}'...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully.")
    return _whisper_model


def get_sarvam_client() -> SarvamAI:
    """Initialize Sarvam AI client using the environment API key."""
    global _sarvam_client
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment or .env file.")
    if _sarvam_client is None:
        _sarvam_client = SarvamAI(api_subscription_key=api_key)
    return _sarvam_client


def slice_audio_wave(wav_path: str, piece_seconds: int = SARVAM_PIECE_SECONDS) -> list[str]:
    """
    Slices a WAV file into smaller segments of `piece_seconds` duration using
    Python's standard wave module.
    """
    if piece_seconds <= 0:
        return [wav_path]

    try:
        with wave.open(wav_path, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            total_frames = wav_file.getnframes()
            total_duration = total_frames / framerate

            if total_duration <= piece_seconds:
                return [wav_path]

            frames_per_piece = int(piece_seconds * framerate)
            num_pieces = int(total_duration // piece_seconds) + (
                1 if total_duration % piece_seconds > 0 else 0
            )

            base_name, _ = os.path.splitext(wav_path)
            piece_paths = []

            for i in range(num_pieces):
                start_frame = i * frames_per_piece
                end_frame = min((i + 1) * frames_per_piece, total_frames)
                frames_to_read = end_frame - start_frame

                wav_file.setpos(start_frame)
                frames = wav_file.readframes(frames_to_read)

                piece_path = f"{base_name}_sarvam_piece_{i}.wav"
                with wave.open(piece_path, "wb") as piece_file:
                    piece_file.setnchannels(n_channels)
                    piece_file.setsampwidth(sampwidth)
                    piece_file.setframerate(framerate)
                    piece_file.writeframes(frames)

                piece_paths.append(piece_path)

            return piece_paths
    except Exception as exc:
        print(f"Warning: Could not slice audio using wave ({exc}). Using full file.")
        return [wav_path]


def transcribe_chunk_whisper(chunk_path: str, translate: bool = False) -> str:
    """Transcribe an audio chunk using local Whisper model."""
    model = load_whisper_model()
    task = "translate" if translate else "transcribe"
    print(f"Transcribing {chunk_path} with Whisper...")

    result = model.transcribe(
        chunk_path,
        task=task,
        beam_size=5,
    )
    return result.get("text", "").strip()


def _send_to_sarvam(
    piece_path: str,
    language_code: str = "hi-IN",
    mode: str = "transcribe",
) -> str:
    """Send one piece (<=30s) to Sarvam AI and return transcript."""
    client = get_sarvam_client()
    with open(piece_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model=SARVAM_STT_MODEL,
            language_code=language_code,
            mode=mode,
        )
    return getattr(response, "transcript", "") or ""


def transcribe_chunk_sarvam(
    chunk_path: str,
    language_code: str = "hi-IN",
    mode: str = "transcribe",
) -> str:
    """
    Transcribes an audio chunk using Sarvam AI.
    Slices large chunks into <=25s pieces to comply with Sarvam sync API limits.
    """
    pieces = slice_audio_wave(chunk_path, piece_seconds=SARVAM_PIECE_SECONDS)
    full_text = []

    for i, piece_path in enumerate(pieces):
        created_temp = piece_path != chunk_path
        try:
            if len(pieces) > 1:
                print(f"  -> Sarvam piece {i + 1}/{len(pieces)}...")
            transcript = _send_to_sarvam(piece_path, language_code=language_code, mode=mode)
            if transcript:
                full_text.append(transcript.strip())
        finally:
            if created_temp and os.path.exists(piece_path):
                try:
                    os.remove(piece_path)
                except OSError:
                    pass

    return " ".join(full_text).strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
    translate: bool = False,
) -> str:
    """
    Route an audio chunk to Whisper or Sarvam AI based on the specified language:
    - 'hinglish' -> Sarvam AI (mode='translate', translates directly to English)
    - 'hindi' / 'hi' -> Sarvam AI (translates to English if translate=True, else native Hindi script)
    - 'english' / 'en' -> Whisper
    """
    lang_lower = language.lower().strip()

    if lang_lower in {"hindi", "hi", "hinglish", "hi-in"}:
        mode = "translate" if (translate or lang_lower == "hinglish") else "transcribe"
        lang_code = "unknown" if lang_lower == "hinglish" else "hi-IN"
        return transcribe_chunk_sarvam(chunk_path, language_code=lang_code, mode=mode)

    return transcribe_chunk_whisper(chunk_path, translate=translate)


def transcribe_all(
    chunks: list[str],
    language: str = "english",
    translate: bool = False,
) -> str:
    """
    Transcribes a list of audio chunk file paths.
    Routes to Sarvam AI for Hindi/Hinglish and Whisper for English.
    """
    lang_lower = language.lower().strip()
    engine = "Sarvam AI" if lang_lower in {"hindi", "hi", "hinglish", "hi-in"} else "Whisper"
    print(f"Using {engine} for transcription (language: '{language}').")

    full_transcript = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language, translate=translate)
        if text:
            full_transcript.append(text)

    print("Transcription complete.")
    return "\n".join(full_transcript)