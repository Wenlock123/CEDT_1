import streamlit as st
import os

from utils.db_utils import extract_chromadb
from utils.whisper_utils import speech_to_text
from utils.rag_utils import retrieve_context, ask_llm
from utils.tts_utils import text_to_speech

# -----------------------------
# Page Title
# -----------------------------

st.title("AI Socratic Tutor")
st.write("อัปโหลดไฟล์เสียงเพื่อถามคำถาม")

# -----------------------------
# Extract ChromaDB (ครั้งแรก)
# -----------------------------

extract_chromadb()

# -----------------------------
# Upload Audio
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["m4a", "wav", "mp3"]
)

if uploaded_file:

    # save temp file
    audio_path = "temp_audio.m4a"

    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("กำลังแปลงเสียงเป็นข้อความ...")

    # -----------------------------
    # Speech → Text (Whisper)
    # -----------------------------

    question = speech_to_text(audio_path)

    st.subheader("คำถามของผู้ใช้")
    st.write(question)

    # -----------------------------
    # Retrieve Context (RAG)
    # -----------------------------

    context = retrieve_context(question)

    # -----------------------------
    # LLM Response
    # -----------------------------

    answer = ask_llm(question, context)

    st.subheader("AI Tutor")
    st.write(answer)

    # -----------------------------
    # Text → Speech
    # -----------------------------

    audio_file = text_to_speech(answer)

    st.audio(audio_file)

    # cleanup temp file
    if os.path.exists(audio_path):
        os.remove(audio_path)
