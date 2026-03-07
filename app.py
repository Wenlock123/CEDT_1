import streamlit as st
import whisper
import os

from utils.rag_utils import retrieve_context, ask_llm
from utils.audio_utils import text_to_speech

st.title("AI Socratic Tutor")

st.write("อัปโหลดไฟล์เสียงเพื่อถามคำถาม")

# -----------------------------
# Load Whisper Model (Cache)
# -----------------------------

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")  # เปลี่ยนจาก medium → base

model = load_whisper()

# -----------------------------
# Upload Audio
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["m4a", "wav", "mp3"]
)

if uploaded_file:

    with open("temp_audio.m4a", "wb") as f:
        f.write(uploaded_file.read())

    st.write("กำลังแปลงเสียงเป็นข้อความ...")

    # -----------------------------
    # Speech → Text
    # -----------------------------

    result = model.transcribe(
        "temp_audio.m4a",
        language="th"
    )

    question = result["text"]

    st.subheader("คำถามของผู้ใช้")
    st.write(question)

    # -----------------------------
    # RAG Retrieval
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

    audio_path = text_to_speech(answer)

    st.audio(audio_path)
