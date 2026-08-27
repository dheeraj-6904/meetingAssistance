import os
import wave
from unittest.mock import MagicMock, mock_open, patch

import pytest

import core.transcriber as transcriber_module
from core.transcriber import (
    _send_to_sarvam,
    get_sarvam_client,
    load_whisper_model,
    slice_audio_wave,
    transcribe_all,
    transcribe_chunk,
    transcribe_chunk_sarvam,
    transcribe_chunk_whisper,
)


def create_dummy_wav(file_path: str, duration_seconds: int = 1, framerate: int = 16000):
    """Helper to generate a valid PCM 16-bit WAV file for testing."""
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * (framerate * duration_seconds))


class TestLoadWhisperModel:
    def testLazyLoadsAndCachesModel(self, monkeypatch):
        mock_model = MagicMock()
        monkeypatch.setattr(transcriber_module, "_whisper_model", None)
        with patch("whisper.load_model", return_value=mock_model) as mock_load:
            model1 = load_whisper_model()
            model2 = load_whisper_model()

            assert model1 is mock_model
            assert model2 is mock_model
            mock_load.assert_called_once_with(transcriber_module.WHISPER_MODEL)


class TestGetSarvamClient:
    def testReturnsCachedClientWhenApiKeyPresent(self, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "test_key_123")
        monkeypatch.setattr(transcriber_module, "_sarvam_client", None)

        with patch("core.transcriber.SarvamAI") as mock_sarvam_cls:
            client1 = get_sarvam_client()
            client2 = get_sarvam_client()

            assert client1 == client2
            mock_sarvam_cls.assert_called_once_with(api_subscription_key="test_key_123")

    def testRaisesErrorWhenApiKeyMissing(self, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        monkeypatch.setattr(transcriber_module, "_sarvam_client", None)

        with pytest.raises(RuntimeError, match="SARVAM_API_KEY is not set"):
            get_sarvam_client()


class TestSliceAudioWave:
    def testReturnsOriginalWhenPieceSecondsNonPositive(self, tmp_path):
        wav_file = str(tmp_path / "audio.wav")
        create_dummy_wav(wav_file, duration_seconds=5)

        assert slice_audio_wave(wav_file, piece_seconds=0) == [wav_file]
        assert slice_audio_wave(wav_file, piece_seconds=-1) == [wav_file]

    def testReturnsOriginalWhenShorterThanPiece(self, tmp_path):
        wav_file = str(tmp_path / "short.wav")
        create_dummy_wav(wav_file, duration_seconds=5)

        pieces = slice_audio_wave(wav_file, piece_seconds=10)
        assert pieces == [wav_file]

    def testSlicesIntoMultiplePieces(self, tmp_path):
        wav_file = str(tmp_path / "long.wav")
        create_dummy_wav(wav_file, duration_seconds=55)

        pieces = slice_audio_wave(wav_file, piece_seconds=25)
        assert len(pieces) == 3
        assert all(os.path.exists(p) for p in pieces)

        # Cleanup generated pieces
        for p in pieces:
            if p != wav_file and os.path.exists(p):
                os.remove(p)

    def testFallbackOnNonWaveFile(self, tmp_path):
        bad_file = str(tmp_path / "not_a_wav.txt")
        with open(bad_file, "w") as f:
            f.write("invalid wave data")

        pieces = slice_audio_wave(bad_file, piece_seconds=10)
        assert pieces == [bad_file]


class TestTranscribeChunkWhisper:
    def testTranscribeTaskTranscribe(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello World  "}

        with patch("core.transcriber.load_whisper_model", return_value=mock_model):
            result = transcribe_chunk_whisper("test.wav", translate=False)
            assert result == "Hello World"
            mock_model.transcribe.assert_called_once_with("test.wav", task="transcribe", beam_size=5)

    def testTranscribeTaskTranslate(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Translated English text"}

        with patch("core.transcriber.load_whisper_model", return_value=mock_model):
            result = transcribe_chunk_whisper("test.wav", translate=True)
            assert result == "Translated English text"
            mock_model.transcribe.assert_called_once_with("test.wav", task="translate", beam_size=5)


class TestSendToSarvam:
    def testSendsPieceAndReturnsTranscript(self, tmp_path):
        wav_file = str(tmp_path / "piece.wav")
        create_dummy_wav(wav_file, duration_seconds=1)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.transcript = "Namaste dosto"
        mock_client.speech_to_text.transcribe.return_value = mock_response

        with patch("core.transcriber.get_sarvam_client", return_value=mock_client):
            transcript = _send_to_sarvam(wav_file, language_code="hi-IN", mode="transcribe")
            assert transcript == "Namaste dosto"
            mock_client.speech_to_text.transcribe.assert_called_once()


class TestTranscribeChunkSarvam:
    def testSlicesAndCleansUpTempFiles(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SARVAM_API_KEY", "test_key")
        wav_file = str(tmp_path / "meeting_audio.wav")
        create_dummy_wav(wav_file, duration_seconds=60)

        mock_send = MagicMock(side_effect=["Part one.", "Part two.", "Part three."])
        with patch("core.transcriber._send_to_sarvam", mock_send):
            result = transcribe_chunk_sarvam(wav_file, language_code="hi-IN", mode="translate")
            assert result == "Part one. Part two. Part three."
            assert mock_send.call_count == 3

        # Verify temporary sliced files were cleaned up
        remaining_files = os.listdir(tmp_path)
        assert remaining_files == ["meeting_audio.wav"]


class TestTranscriberRouting:
    @patch("core.transcriber.transcribe_chunk_whisper")
    def testRoutesEnglishToWhisper(self, mock_whisper):
        mock_whisper.return_value = "Hello world"
        result = transcribe_chunk("dummy.wav", language="english")

        assert result == "Hello world"
        mock_whisper.assert_called_once_with("dummy.wav", translate=False)

    @patch("core.transcriber.transcribe_chunk_sarvam")
    def testRoutesHindiToSarvamTranscribe(self, mock_sarvam):
        mock_sarvam.return_value = "नमस्ते"
        result = transcribe_chunk("dummy.wav", language="hindi", translate=False)

        assert result == "नमस्ते"
        mock_sarvam.assert_called_once_with("dummy.wav", language_code="hi-IN", mode="transcribe")

    @patch("core.transcriber.transcribe_chunk_sarvam")
    def testRoutesHindiWithTranslateFlag(self, mock_sarvam):
        mock_sarvam.return_value = "Hello in English"
        result = transcribe_chunk("dummy.wav", language="hindi", translate=True)

        assert result == "Hello in English"
        mock_sarvam.assert_called_once_with("dummy.wav", language_code="hi-IN", mode="translate")

    @patch("core.transcriber.transcribe_chunk_sarvam")
    def testRoutesHinglishToSarvamTranslate(self, mock_sarvam):
        mock_sarvam.return_value = "Hello brother how are you"
        result = transcribe_chunk("dummy.wav", language="hinglish")

        assert result == "Hello brother how are you"
        mock_sarvam.assert_called_once_with("dummy.wav", language_code="unknown", mode="translate")


class TestTranscribeAll:
    @patch("core.transcriber.transcribe_chunk")
    def testAggregatesMultipleChunksWithNewlines(self, mock_chunk):
        mock_chunk.side_effect = ["First section text.", "Second section text."]
        result = transcribe_all(["chunk1.wav", "chunk2.wav"], language="english")

        assert result == "First section text.\nSecond section text."
        assert mock_chunk.call_count == 2

    def testHandlesEmptyChunkList(self):
        result = transcribe_all([], language="english")
        assert result == ""
