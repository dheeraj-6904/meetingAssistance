import os
import gc
import shutil
import sys

import streamlit as st
from dotenv import load_dotenv

# load .env first so FFMPEG_LOCATION is available
load_dotenv()

# inject ffmpeg into PATH before any ffmpeg-dependent imports
ffmpeg_dir = os.getenv("FFMPEG_LOCATION")
if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

from utils.AudioProcessor import process_input, chunk_audio
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question, close_vector_store

# page config has to be the first streamlit call
st.set_page_config(page_title="Meeting notes", layout="wide", initial_sidebar_state="collapsed")

# ---------- styling ----------
# one place for all the css so the python below stays readable
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
  --paper: #F1F4F1;
  --card: #FFFFFF;
  --ink: #14201E;
  --muted: #5B6663;
  --line: #D9DFDB;
  --teal: #0F6B63;
  --teal-soft: #DCEBE8;
  --amber: #D9A406;
  --slate: #5B7C99;
}

.stApp { font-family: 'Schibsted Grotesk', sans-serif; color: var(--ink); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 3.2rem; padding-bottom: 4rem; }

/* headings */
.hero { font-size: 2.9rem; line-height: 1.08; font-weight: 700; letter-spacing: -0.03em; margin: 0 0 1rem 0; max-width: 18ch; }
.lede { font-size: 1.08rem; line-height: 1.55; color: var(--muted); max-width: 46ch; margin: 0 0 2rem 0; }
.steps { padding-left: 1.2rem; margin: 0; color: var(--ink); font-size: 0.98rem; line-height: 1.5; }
.steps li { margin-bottom: 0.6rem; padding-left: 0.3rem; }
.steps li::marker { color: var(--teal); font-weight: 600; }
.note { font-size: 0.88rem; color: var(--muted); margin-top: 1.4rem; max-width: 46ch; }
.doc-title { font-size: 2.1rem; line-height: 1.15; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 0.3rem 0; }
.panel-title { font-size: 1.15rem; font-weight: 600; margin: 0 0 0.15rem 0; }

/* buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 8px; font-weight: 500; padding: 0.55rem 1.1rem;
  border: 1px solid var(--line); background: var(--card); color: var(--ink);
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--teal); color: var(--teal); }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: var(--teal); border-color: var(--teal); color: #fff;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: #0B544E; color: #fff; }

/* inputs */
.stTextInput input { background: var(--card); border-radius: 8px; }
[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 1.5px dashed #B9C4BE; border-radius: 10px; }

/* intake card */
.st-key-intake { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.4rem 1.4rem 1.2rem 1.4rem; }

/* tabs */
button[data-baseweb="tab"] { font-weight: 500; }

/* note sections: left rule colour says what kind of content it is */
.st-key-sec-summary, .st-key-sec-actions, .st-key-sec-decisions, .st-key-sec-questions {
  border-left: 3px solid var(--teal); padding-left: 1.2rem; margin-top: 0.4rem;
}
.st-key-sec-actions { border-left-color: var(--amber); }
.st-key-sec-questions { border-left-color: var(--slate); }
.st-key-sec-summary p, .st-key-sec-summary li, .st-key-sec-actions p, .st-key-sec-actions li,
.st-key-sec-decisions p, .st-key-sec-decisions li, .st-key-sec-questions p, .st-key-sec-questions li {
  font-family: 'Source Serif 4', Georgia, serif; font-size: 1.05rem; line-height: 1.7;
}

/* chat */
.st-key-chat-panel { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.2rem; }
[data-testid="stChatMessage"] { background: transparent; border-radius: 12px; padding: 0.6rem 0.8rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--teal-soft); }
[data-testid="stChatMessage"] p { font-size: 0.98rem; line-height: 1.6; }
.st-key-chips button { border-radius: 999px; font-size: 0.86rem; text-align: left; justify-content: flex-start; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- helpers ----------
def clear_directory_contents(directory: str) -> None:
    if not os.path.isdir(directory):
        return

    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def end_session() -> None:
    # same cleanup the cli does on exit, we drop the chain first so the vector store can close
    result = st.session_state.pop("result", None)
    st.session_state.pop("messages", None)
    if result:
        result["rag_chain"] = None
    del result
    gc.collect()
    close_vector_store()
    clear_directory_contents("vector_db")
    clear_directory_contents("downloads")


def queue_question(q: str) -> None:
    # chips set this and the next rerun picks it up like a typed question
    st.session_state.pending_q = q


def notes_as_markdown(r: dict) -> str:
    return (
        f"# {r['title']}\n\n"
        f"## Summary\n{r['summary']}\n\n"
        f"## Action items\n{r['action_items']}\n\n"
        f"## Key decisions\n{r['key_decisions']}\n\n"
        f"## Open questions\n{r['open_questions']}\n"
    )


def section(key: str, caption: str, text: str) -> None:
    st.caption(caption)
    with st.container(key=key):
        st.markdown(str(text).strip() or "Nothing was flagged here.")


def process(source: str, language: str) -> None:
    # same steps as run_pipeline, split out so we can show progress between them
    with st.status("Starting", expanded=True) as status:
        try:
            status.update(label="Getting the audio")
            res = process_input(source)
            if not res["ok"]:
                status.update(label="Couldn't read that source", state="error")
                st.error(res["message"])
                if res.get("suggestion"):
                    st.caption(res["suggestion"])
                return

            status.update(label="Transcribing, long meetings take a few minutes")
            chunks = chunk_audio(res["whisper_audio_path"], chunk_minutes=10)
            transcript = transcribe_all(chunks, language)

            status.update(label="Writing the summary")
            title = generate_title(transcript)
            summary = summarize(transcript)

            status.update(label="Finding actions, decisions and open questions")
            action_item = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)

            status.update(label="Getting chat ready")
            rag_chain = build_rag_chain(transcript)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(str(e))
            return

        status.update(label="Done", state="complete", expanded=False)

    st.session_state.result = {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }
    st.session_state.messages = []
    st.rerun()


# ---------- intake screen ----------
def intake_view() -> None:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<h1 class="hero">Drop in a meeting. Get the notes.</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="lede">Paste a YouTube link or upload a recording. You get a summary, '
            "who needs to do what, what was decided and what is still open. Then you can ask it questions.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<ol class="steps">'
            "<li>We pull the audio and transcribe it.</li>"
            "<li>We write the summary, actions, decisions and open questions.</li>"
            "<li>You ask follow-ups in a chat that reads the transcript.</li>"
            "</ol>"
            '<p class="note">Audio is split into 10 minute parts, so long meetings take a few minutes. '
            "Everything is cleared when you end the session.</p>",
            unsafe_allow_html=True,
        )

    with right:
        with st.container(key="intake"):
            language = st.radio("Spoken language", ["english", "hinglish"], horizontal=True, format_func=str.capitalize)
            link_tab, file_tab = st.tabs(["YouTube link", "Upload a recording"])

            with link_tab:
                url = st.text_input("Link", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
                # button stays off until there is something to process, so no empty-submit errors
                if st.button("Create notes", type="primary", width="stretch", disabled=not url.strip(), key="go_link"):
                    process(url.strip(), language)

            with file_tab:
                up = st.file_uploader(
                    "Recording",
                    type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg", "flac"],
                    label_visibility="collapsed",
                )
                if st.button("Create notes", type="primary", width="stretch", disabled=up is None, key="go_file"):
                    # we save the upload where process_input can read it as a normal local path
                    os.makedirs("downloads", exist_ok=True)
                    path = os.path.join("downloads", up.name)
                    with open(path, "wb") as out:
                        out.write(up.getbuffer())
                    process(path, language)


# ---------- results screen ----------
def results_view(r: dict) -> None:
    head, dl, new = st.columns([6, 1.5, 1.4], vertical_alignment="bottom")
    with head:
        st.markdown(f'<h1 class="doc-title">{r["title"]}</h1>', unsafe_allow_html=True)
    with dl:
        st.download_button("Download notes", notes_as_markdown(r), file_name="meeting-notes.md", mime="text/markdown", width="stretch")
    with new:
        st.button("End session", on_click=end_session, width="stretch", help="Clears the transcript and chat, then starts fresh")

    st.write("")
    notes, chat = st.columns([1.3, 1], gap="large")

    with notes:
        t_sum, t_act, t_dec, t_q, t_tr = st.tabs(["Summary", "Action items", "Decisions", "Open questions", "Transcript"])
        with t_sum:
            section("sec-summary", "The short version.", r["summary"])
        with t_act:
            section("sec-actions", "Things someone needs to do.", r["action_items"])
        with t_dec:
            section("sec-decisions", "What the group settled on.", r["key_decisions"])
        with t_q:
            section("sec-questions", "Still unanswered when the meeting ended.", r["open_questions"])
        with t_tr:
            with st.container(height=520, border=False):
                st.markdown(r["transcript"])

    with chat:
        with st.container(key="chat-panel"):
            st.markdown('<p class="panel-title">Ask this meeting</p>', unsafe_allow_html=True)
            st.caption("Answers are drawn from the transcript.")

            box = st.container(height=420, border=False)
            prompt = st.chat_input("Ask about anything that was said")

            # a typed question wins, otherwise we take whatever a chip queued
            queued = st.session_state.pop("pending_q", None)
            q = prompt or queued

            with box:
                if not st.session_state.messages and not q:
                    with st.container(key="chips"):
                        for s in ["What were the main takeaways?", "Who owns what next?", "What is still unresolved?"]:
                            st.button(s, on_click=queue_question, args=(s,), width="stretch", key=f"chip_{s}")

                for m in st.session_state.messages:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])

                if q:
                    with st.chat_message("user"):
                        st.markdown(q)
                    with st.chat_message("assistant"):
                        with st.spinner("Looking through the transcript"):
                            answer = ask_question(r["rag_chain"], q)
                        st.markdown(answer)
                    st.session_state.messages += [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": answer},
                    ]


# ---------- router ----------
if st.session_state.get("result"):
    results_view(st.session_state.result)
else:
    intake_view()