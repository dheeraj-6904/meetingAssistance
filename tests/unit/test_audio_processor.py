import os
import wave

import pytest

from utils.AudioProcessor import (
    chunk_audio,
    is_youtube_block_error,
    process_input,
    resolve_ffmpeg_bin_dir,
)


class TestIsYoutubeBlockError:
    def testDetectsBotCheckMessage(self):
        Message = "Sign in to confirm you’re not a bot. Use --cookies-from-browser"
        assert is_youtube_block_error(Message) is True

    def testDetectsRateLimitMessage(self):
        Message = "HTTP Error 429: Too Many Requests"
        assert is_youtube_block_error(Message) is True

    def testIgnoresGenericOperationMessage(self):
        Message = "ERROR: Postprocessing: ffprobe and ffmpeg not found"
        assert is_youtube_block_error(Message) is False


class TestChunkAudio:
    def testCreatesChunkFilesForWaveFile(self, tmp_path):
        AudioPath = tmp_path / "sample.wav"

        with wave.open(str(AudioPath), "wb") as AudioFile:
            AudioFile.setnchannels(1)
            AudioFile.setsampwidth(2)
            AudioFile.setframerate(8000)
            frames = b"\x00\x00" * (8000 * 61)
            AudioFile.writeframes(frames)

        Chunks = chunk_audio(str(AudioPath), chunk_minutes=1)

        assert len(Chunks) == 2
        assert all(os.path.exists(Path) for Path in Chunks)
        assert all(Path.endswith(".wav") for Path in Chunks)

    def testRejectsNonPositiveChunkMinutes(self, tmp_path):
        AudioPath = tmp_path / "sample.wav"

        with wave.open(str(AudioPath), "wb") as AudioFile:
            AudioFile.setnchannels(1)
            AudioFile.setsampwidth(2)
            AudioFile.setframerate(8000)
            AudioFile.writeframes(b"\x00\x00" * 8000)

        with pytest.raises(ValueError):
            chunk_audio(str(AudioPath), chunk_minutes=0)


class TestProcessInput:
    def testReturnsSuccessForLocalAudio(self, tmp_path):
        AudioPath = tmp_path / "meeting.wav"

        with wave.open(str(AudioPath), "wb") as AudioFile:
            AudioFile.setnchannels(1)
            AudioFile.setsampwidth(2)
            AudioFile.setframerate(8000)
            AudioFile.writeframes(b"\x00\x00" * 8000)

        Result = process_input(str(AudioPath))

        assert Result["ok"] is True
        assert Result["source"] == "local"
        assert Result["audio_path"].endswith("_16k_mono.wav")
        assert os.path.exists(Result["audio_path"]) is True

    def testReturnsFailureForBlockedYoutubeUrl(self):
        Result = process_input("https://youtube.com/watch?v=blocked")

        assert Result["ok"] is False
        assert Result["source"] == "youtube"
        assert Result["error_code"] == "blocked_by_youtube_auth"
        assert "Please upload" in Result["suggestion"]


class TestResolveFfmpegBinDir:
    def testReturnsEnvPathIfDirectory(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "custom_bin"
        bin_dir.mkdir()
        monkeypatch.setenv("FFMPEG_LOCATION", str(bin_dir))
        assert resolve_ffmpeg_bin_dir() == str(bin_dir)

    def testReturnsEnvPathDirIfFile(self, monkeypatch, tmp_path):
        bin_file = tmp_path / "ffmpeg.exe"
        bin_file.write_text("")
        monkeypatch.setenv("FFMPEG_LOCATION", str(bin_file))
        assert resolve_ffmpeg_bin_dir() == str(tmp_path)

    def testReturnsPathFromShutilWhich(self, monkeypatch):
        monkeypatch.delenv("FFMPEG_LOCATION", raising=False)
        fake_ffmpeg = os.path.abspath(os.path.join("dummy", "bin", "ffmpeg"))
        monkeypatch.setattr("shutil.which", lambda cmd: fake_ffmpeg if cmd == "ffmpeg" else None)
        assert resolve_ffmpeg_bin_dir() == os.path.dirname(fake_ffmpeg)

    def testReturnsNoneWhenNotFound(self, monkeypatch):
        monkeypatch.delenv("FFMPEG_LOCATION", raising=False)
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        assert resolve_ffmpeg_bin_dir() is None

