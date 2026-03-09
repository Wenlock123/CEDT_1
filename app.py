import streamlit as st
from streamlit_mic_recorder import mic_recorder

from utils.rag_utils import retrieve_context, ask_llm
from utils.whisper_utils import speech_to_text
from utils.tts_utils import text_to_speech

st.title("🎙️ Just Talk AI Tutor")

st.write("ใส่หัวข้อที่อยากเรียน แล้วคุยกับ AI ด้วยเสียง")


# -------------------------
# Session memory
# -------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "topic" not in st.session_state:
    st.session_state.topic = None

if "context" not in st.session_state:
    st.session_state.context = None


# -------------------------
# Topic input
# -------------------------

topic = st.text_input("Topic ที่อยากเรียน")

if topic and st.session_state.topic is None:

    st.session_state.topic = topic

    # RAG
    context = retrieve_context(topic)
    st.session_state.context = context

    answer = ask_llm(
        topic,
        "เริ่มต้นบทสนทนา",
        context,
        st.session_state.chat_history
    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })


# -------------------------
# Display chat
# -------------------------

for msg in st.session_state.chat_history:

    if msg["role"] == "assistant":

        st.chat_message("assistant").write(msg["content"])

        # AI พูด
        audio = text_to_speech(msg["content"])
        st.audio(audio)

    else:

        st.chat_message("user").write(msg["content"])


# -------------------------
# Voice input
# -------------------------

audio = mic_recorder(
    start_prompt="🎤 กดเพื่อพูด",
    stop_prompt="⏹️ หยุด",
    just_once=True
)


if audio:

    with open("input.wav", "wb") as f:
        f.write(audio["bytes"])

    user_text = speech_to_text("input.wav")

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_text
    })

    answer = ask_llm(
        st.session_state.topic,
        user_text,
        st.session_state.context,
        st.session_state.chat_history
    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()
