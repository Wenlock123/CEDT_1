import streamlit as st
from streamlit_mic_recorder import mic_recorder

from utils.whisper_utils import speech_to_text
from utils.tts_utils import text_to_speech

st.title("🎤 Just Talk")
st.caption("Talk and think together")

# -------------------------
# Session state
# -------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "step" not in st.session_state:
    st.session_state.step = 0

if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# -------------------------
# Script conversation
# -------------------------

script = [

"ถ้าท่องจำอย่างเดียวมันลืมง่าย ลองคิดจากวิถีชีวิตของมันดีกว่า ปกติเวลาสัตว์หิวหรือร้อนมากๆ มันทำยังไง",

"ใช่เลย สัตว์เคลื่อนที่ได้ แต่พืชขยับหนีไปไหนไม่ได้เลย งั้นคิดว่าพืชต้องมีอะไรในเซลล์ เพื่อสร้างอาหารเองจากที่ที่มันยืนอยู่",

"ถูกต้อง ทีนี้ลองคิดเรื่องโครงสร้างบ้าง สัตว์มีกระดูกช่วยพยุงตัว แต่พืชไม่มี แล้วมันใช้อะไรทำให้ลำต้นแข็งแรงตั้งตรงได้",

"ใช่ เพราะงั้นเซลล์พืชจึงมีทั้งคลอโรพลาสต์สำหรับสร้างอาหาร และผนังเซลล์สำหรับค้ำจุนโครงสร้าง",

"เป๊ะเลย และเพราะมีผนังเซลล์ เซลล์พืชจึงมักเป็นเหลี่ยมๆ ส่วนเซลล์สัตว์จะดูมนและยืดหยุ่นมากกว่า"

]

# -------------------------
# Topic input
# -------------------------

topic = st.text_input("Topic ที่อยากคุย")

if topic and st.session_state.step == 0:

    st.session_state.chat_history.append(
        {"role": "user", "content": topic}
    )

    first_answer = script[0]

    st.session_state.chat_history.append(
        {"role": "assistant", "content": first_answer}
    )

    st.session_state.last_audio = text_to_speech(first_answer)

    st.session_state.step = 1

# -------------------------
# Display chat
# -------------------------

for msg in st.session_state.chat_history:

    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])

    if msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

# -------------------------
# Play voice
# -------------------------

if st.session_state.last_audio:
    st.audio(st.session_state.last_audio, format="audio/mp3")
    st.session_state.last_audio = None

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

    if st.session_state.step < len(script):

        answer = script[st.session_state.step]

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

        st.session_state.last_audio = text_to_speech(answer)

        st.session_state.step += 1

    st.session_state.mic_key += 1

    st.rerun()
