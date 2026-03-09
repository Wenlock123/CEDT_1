import streamlit as st
from streamlit_mic_recorder import mic_recorder

from utils.rag_utils import retrieve_context, ask_llm
from utils.whisper_utils import speech_to_text
from utils.tts_utils import text_to_speech

st.title("🎓 AI Socratic Tutor")

# -------------------------
# Session state
# -------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "topic_started" not in st.session_state:
    st.session_state.topic_started = False

# ใช้ reset mic
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0


# -------------------------
# Topic input
# -------------------------

topic = st.text_input("Topic ที่อยากเรียน")

if topic and not st.session_state.topic_started:

    context = retrieve_context(topic)

    first_question = ask_llm(
        topic,
        f"เริ่มสอนหัวข้อนี้: {topic}",
        context,
        []
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": first_question}
    )

    st.session_state.topic_started = True


# -------------------------
# Display chat
# -------------------------

for msg in st.session_state.chat_history:

    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])

    if msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])


# -------------------------
# Voice input
# -------------------------

audio = mic_recorder(
    start_prompt="🎤 กดเพื่อพูด",
    stop_prompt="⏹️ หยุด",
    key=f"mic_{st.session_state.mic_key}"
)

if audio and "bytes" in audio:

    user_text = speech_to_text(audio["bytes"])

    st.session_state.chat_history.append(
        {"role": "user", "content": user_text}
    )

    context = retrieve_context(topic)

    answer = ask_llm(
        topic,
        user_text,
        context,
        st.session_state.chat_history
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

    # เล่นเสียงตอบ
    audio_response = text_to_speech(answer)
    st.audio(audio_response, format="audio/mp3")

    # reset mic
    st.session_state.mic_key += 1

    st.rerun()
