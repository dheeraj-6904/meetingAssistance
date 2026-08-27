import os
import shutil
import sys
from typing import Any

from dotenv import load_dotenv
import ffmpeg
import yt_dlp
from yt_dlp.utils import DownloadError

load_dotenv()

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def is_youtube_block_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "sign in to confirm" in lowered
        or "not a bot" in lowered
        or "too many requests" in lowered
        or "cookies-from-browser" in lowered
        or "429" in lowered
        or "blocked" in lowered
        or "truncated" in lowered
        or "incomplete youtube id" in lowered
    )


def resolve_ffmpeg_bin_dir() -> str | None:
    """
    Resolves the directory containing the FFmpeg binaries dynamically.

    Resolution order:
    1. Environment variable `FFMPEG_LOCATION` (can be a directory or binary path).
    2. System PATH via shutil.which('ffmpeg').
    """
    env_path = os.getenv("FFMPEG_LOCATION")
    if env_path:
        if os.path.isfile(env_path):
            return os.path.dirname(os.path.abspath(env_path))
        if os.path.isdir(env_path):
            return os.path.abspath(env_path)

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return os.path.dirname(os.path.abspath(ffmpeg_exe))

    return None


def download_youtube_audio(url: str) -> str:
    """
    Downloads a YouTube video and returns the path to the downloaded file.

    Args:
        url (str): The URL of the YouTube video.
        output_dir (str): The directory to save the downloaded file.

    Returns:
        str: The path to the downloaded file.
    """

    output_path = os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')
    ffmpeg_dir = resolve_ffmpeg_bin_dir()
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    if ffmpeg_dir:
        ydl_opts['ffmpeg_location'] = ffmpeg_dir
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return filename
    except DownloadError as exc:
        message = str(exc)
        if is_youtube_block_error(message):
            raise RuntimeError(
                "YouTube blocked this download. This is usually caused by anti-bot checks on the video or server IP. "
                "Use a different public URL or upload the audio/video directly."
            ) from exc
        raise


def process_input(source: str) -> dict[str, Any]:
    """
    Attempts YouTube audio extraction or local-file conversion and returns
    structured status instead of crashing so caller can fall back to upload
    flow in the UI.
    """

    try:
        if source.startswith("http://") or source.startswith("https://"):
            downloaded_file = download_youtube_audio(source)
            processed_file = convert_to_wav(downloaded_file)
            source_type = "youtube"
        else:
            processed_file = convert_to_wav(source)
            source_type = "local"

        return {
            "ok": True,
            "source": source_type,
            "audio_path": processed_file,
            "whisper_audio_path": processed_file,
            "error_code": None,
            "message": None,
        }
    except RuntimeError as exc:
        return {
            "ok": False,
            "source": "youtube" if source.startswith("http://") or source.startswith("https://") else "local",
            "audio_path": None,
            "whisper_audio_path": None,
            "error_code": "blocked_by_youtube_auth",
            "message": str(exc),
            "suggestion": "Please upload the meeting audio/video file directly.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "source": "youtube" if source.startswith("http://") or source.startswith("https://") else "local",
            "audio_path": None,
            "whisper_audio_path": None,
            "error_code": "youtube_download_failed",
            "message": str(exc),
            "suggestion": "Please upload the meeting audio/video file directly.",
        }


def convert_to_wav(input_path: str, output_path: str | None = None) -> str:
    """
    Converts audio into Whisper-friendly WAV (16kHz, mono, PCM 16-bit).

    Args:
        input_path (str): Source audio path.
        output_path (str | None): Optional destination path.

    Returns:
        str: Converted WAV file path.
    """

    if output_path is None:
        base_name, _ = os.path.splitext(input_path)
        output_path = f"{base_name}_16k_mono.wav"

    ffmpeg_dir = resolve_ffmpeg_bin_dir()
    binary_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffmpeg_path = os.path.join(ffmpeg_dir, binary_name) if ffmpeg_dir else shutil.which("ffmpeg")

    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        raise RuntimeError("ffmpeg not found. Install FFmpeg and make sure ffmpeg/ffprobe are on PATH or set FFMPEG_LOCATION.")

    (
        ffmpeg
        .input(input_path)
        .output(output_path, ac=1, ar=16000, format="wav", acodec="pcm_s16le")
        .overwrite_output()
        .run(cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)
    )
    return output_path

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    """
    Chunks a WAV file into smaller segments of a fixed duration in minutes.

    Args:
        wav_path (str): Path to the input WAV file.
        chunk_minutes (int): Length of each chunk in minutes.

    Returns:
        list[str]: List of paths to the chunked WAV files.
    """
    import wave

    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than 0")

    seconds_per_chunk = chunk_minutes * 60
    chunk_paths = []
    with wave.open(wav_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        total_frames = wav_file.getnframes()
        total_duration_seconds = total_frames / frame_rate
        num_chunks = int(total_duration_seconds // seconds_per_chunk) + (
            1 if total_duration_seconds % seconds_per_chunk > 0 else 0
        )

        for i in range(num_chunks):
            start_second = i * seconds_per_chunk
            end_second = min((i + 1) * seconds_per_chunk, total_duration_seconds)
            start_frame = int(start_second * frame_rate)
            end_frame = int(end_second * frame_rate)
            wav_file.setpos(start_frame)
            frames = wav_file.readframes(end_frame - start_frame)

            chunk_path = f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"
            with wave.open(chunk_path, 'wb') as chunk_file:
                chunk_file.setnchannels(wav_file.getnchannels())
                chunk_file.setsampwidth(wav_file.getsampwidth())
                chunk_file.setframerate(frame_rate)
                chunk_file.writeframes(frames)

            chunk_paths.append(chunk_path)

    return chunk_paths


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/Ty8gcCKuwNI?si=8u_QZzrrhOAZ3Shm"
    result = process_input(source)

    if result["ok"]:
        print(f"Processed file: {result['audio_path']}")
        files_in_downloads = os.listdir(DOWNLOAD_DIR)
        print(f"Files in downloads folder: {files_in_downloads}")
        chunked_files = chunk_audio(result["whisper_audio_path"], chunk_minutes=10)
        print(f"Chunked audio files: {chunked_files}")
    else:
        print(f"Processing failed: {result['error_code']}")
        print(result["message"])
        print(result["suggestion"])