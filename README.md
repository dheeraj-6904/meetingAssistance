# Meeting Assistance

Meeting Assistance turns a YouTube video or uploaded recording into structured, searchable meeting notes. It transcribes the audio, creates a concise summary, extracts action items and decisions, highlights open questions, and lets you ask follow-up questions grounded in the transcript.

The project includes a polished Streamlit interface as well as a command-line entry point.

## What it does

- Accepts YouTube links and local audio/video uploads.
- Converts recordings to Whisper-friendly 16 kHz mono WAV audio.
- Transcribes English with local OpenAI Whisper.
- Transcribes Hindi and Hinglish with Sarvam AI.
- Generates:
  - A meeting title
  - Summary
  - Action items
  - Key decisions
  - Open questions and follow-up topics
- Builds a local Chroma vector store for transcript-based question answering.
- Provides a chat interface for asking questions about the meeting.
- Exports the generated notes as a Markdown file.
- Clears generated downloads and vector-store contents when a session ends.

## Screenshots

### Start a session

Choose the spoken language, paste a YouTube link, or upload a recording. The app shows progress while it downloads, converts, transcribes, and analyzes the meeting.

![Meeting intake screen](ss/Screenshot%20%281888%29.png)

### Review notes and ask questions

The results view separates the transcript from the summary, action items, decisions, and open questions. The right-hand chat panel answers questions using the transcript as context.

![Meeting results and transcript chat](ss/Screenshot%20%281889%29.png)

## Requirements

- Python 3.10 or newer
- FFmpeg
- A Mistral API key
- A Sarvam AI API key when using Hindi or Hinglish transcription
- Internet access for YouTube downloads and the first-time model downloads

Whisper runs locally, so its model files and PyTorch installation may require significant disk space.

## Installation

Clone the repository and create an isolated environment:

```powershell
git clone https://github.com/dheeraj-6904/meetingAssistance.git
cd meetingAssistance

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install FFmpeg separately and make sure `ffmpeg` is available on your `PATH`. On Windows, you can alternatively point the application to the FFmpeg directory or executable with `FFMPEG_LOCATION`.

## Environment variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key

# Optional settings
WHISPER_MODEL=tiny
SARVAM_STT_MODEL=saaras:v4
SARVAM_PIECE_SECONDS=25
FFMPEG_LOCATION=C:\path\to\ffmpeg\bin
```

Do not commit `.env` or API keys. The repository's `.gitignore` already excludes `.env`.

## Running the application

### Streamlit interface

```powershell
streamlit run app.py
```

Open the local URL displayed by Streamlit, usually `http://localhost:8501`.

The interface supports:

1. Selecting English or Hinglish.
2. Processing a YouTube URL.
3. Uploading `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.webm`, `.ogg`, or `.flac` files.
4. Downloading notes as `meeting-notes.md`.
5. Ending the session to clear the current transcript, chat, downloads, and vector data.

### Command-line interface

```powershell
python main.py
```

Enter a YouTube URL or local file path, select the language, and then ask questions in the terminal. Type `exit`, `quit`, or `q` to end the session.

## Processing pipeline

```text
YouTube URL or uploaded file
              |
              v
      Audio download/conversion
              |
              v
       10-minute audio chunks
              |
              v
        Speech transcription
              |
              v
   Title, summary, actions, decisions
              |
              v
       Chroma vector store + RAG chat
```

English uses Whisper. Hindi and Hinglish use Sarvam AI. The transcript is split into chunks and embedded locally with a sentence-transformer model before it is used by the retrieval-augmented chat flow.

## Project structure

```text
meetingAssistance/
├── app.py                    # Streamlit application
├── main.py                   # Command-line application
├── requirements.txt           # Python dependencies
├── core/
│   ├── extractor.py           # Actions, decisions, and questions
│   ├── rag_engine.py          # Retrieval-augmented question answering
│   ├── summarize.py           # Title and summary generation
│   ├── transcriber.py         # Whisper and Sarvam transcription
│   └── vector_store.py        # Chroma vector-store management
├── utils/
│   └── AudioProcessor.py      # Downloading, conversion, and chunking
├── ss/                        # Application screenshots
└── tests/                     # Unit tests
```

The `downloads/` and `vector_db/` directories are runtime locations. Their contents are temporary and are cleared when the session is ended.

## Troubleshooting

### FFmpeg is not found

Install FFmpeg and add its `bin` directory to `PATH`, or set:

```env
FFMPEG_LOCATION=C:\path\to\ffmpeg\bin
```

### Hindi or Hinglish transcription fails

Confirm that `SARVAM_API_KEY` is present and valid. English transcription does not require Sarvam AI.

### Processing takes a while

Long recordings are split into ten-minute chunks. The first run may also download the Whisper and embedding models, so it can take longer than subsequent runs.