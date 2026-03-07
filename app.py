import streamlit as st
from utils.whisper_utils import speech_to_text
from utils.rag_utils import ask_llm
from utils.tts_utils import text_to_speech

st.set_page_config(
    page_title="Voice RAG Assistant",
    layout="centered"
)

st.title("🎤 Voice RAG AI Assistant")

uploaded_audio = st.file_uploader(
    "Upload audio file",
    type=["wav","mp3","m4a"]
)

if uploaded_audio is not None:

    with open("input_audio.m4a","wb") as f:
        f.write(uploaded_audio.read())

    st.info("Transcribing audio...")

    question = speech_to_text("input_audio.m4a")

    st.write("### User Question")
    st.write(question)

    st.info("Searching knowledge base...")

    answer = ask_llm(question)

    st.write("### AI Answer")
    st.write(answer)

    st.info("Generating voice response...")

    audio_file = text_to_speech(answer)

    st.audio(audio_file)
