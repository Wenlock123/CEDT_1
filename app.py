import streamlit as st
import whisper

from utils.rag_utils import retrieve_context, ask_llm
from utils.audio_utils import text_to_speech

st.title("AI Socratic Tutor")

# โหลด whisper
model = whisper.load_model("medium")

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["m4a", "wav", "mp3"]
)

if uploaded_file:

    with open("temp_audio.m4a", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Transcribing...")

    result = model.transcribe(
        "temp_audio.m4a",
        language="th"
    )

    question = result["text"]

    st.write("User Question:")
    st.write(question)

    # RAG
    context = retrieve_context(question)

    # LLM
    answer = ask_llm(question, context)

    st.write("AI Tutor:")
    st.write(answer)

    # TTS
    audio_path = text_to_speech(answer)

    st.audio(audio_path)
