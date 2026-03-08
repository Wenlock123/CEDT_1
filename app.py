import streamlit as st
from utils.rag_utils import retrieve_context, ask_llm

st.title("Just Talk AI Tutor")

st.write("ใส่หัวข้อที่อยากเรียน แล้วคุยกับ AI")

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
        topic,
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
    else:
        st.chat_message("user").write(msg["content"])


# -------------------------
# User chat input
# -------------------------

user_input = st.chat_input("ตอบคำถาม หรือพิมพ์ 'สิ้นสุด'")

if user_input:

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    answer = ask_llm(
        st.session_state.topic,
        user_input,
        st.session_state.context,
        st.session_state.chat_history
    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()
