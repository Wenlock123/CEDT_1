import streamlit as st
from streamlit_mic_recorder import mic_recorder
from utils.whisper_utils import speech_to_text
from utils.tts_utils import text_to_speech

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Just Talk",
    page_icon="🎤",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

.stApp{
    background-color:#f4f6fb;
}

/* title */
.title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#1f3b73;
    margin-bottom:20px;
}

/* chat container */
.chat-card{
    background:white;
    border-radius:20px;
    padding:30px;
    padding-bottom:120px;
    box-shadow:0 10px 25px rgba(0,0,0,0.08);
}

/* user bubble */
.user-bubble{
    background:#2f63b5;
    color:white;
    padding:14px 18px;
    border-radius:14px;
    margin:10px 0;
    width:70%;
    margin-left:auto;
    font-size:16px;
}

/* bot bubble */
.bot-bubble{
    background:#d9d9de;
    color:black;
    padding:14px 18px;
    border-radius:14px;
    margin:10px 0;
    width:70%;
    font-size:16px;
}

/* mic wrapper bottom center */
.mic-wrapper{
    position:fixed;
    bottom:30px;
    left:50%;
    transform:translateX(-50%);
    z-index:999;
}

/* mic circle */
.mic-circle{
    background:#f5c233;
    width:70px;
    height:70px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    box-shadow:0 6px 15px rgba(0,0,0,0.2);
    font-size:28px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.markdown('<div class="title">Just Talk</div>', unsafe_allow_html=True)

st.markdown('<div class="chat-card">', unsafe_allow_html=True)

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
script_user = [
"เราพยายามจำความต่างเซลล์พืชกับเซลล์สัตว์ แต่ลืมตลอดเลย",
"แผงโซลาร์ ถ้าในเซลล์ก็คือ คลอโรพลาสต์ ใช่ไหม แต่ถ้าสัตว์ไม่มีคลอโรพลาสต์ แล้วสัตว์เอาพลังงานจากไหนมาขยายขนาดร่างกายล่ะ",
"ต้องเป็นผนังเซลล์แน่เลย แล้วถ้าสัตว์มีผนังเซลล์บ้างล่ะ มันจะช่วยให้เราแข็งแกร่งขึ้นไหม",
"ไม่ไหวแน่ ขยับตัวลำบาก วิ่งไปหาทรัพยากรใหม่ๆ ไม่ได้ สรุปคือพืชยอมแข็งเพื่อผลิตเอง ส่วนสัตว์เลือกยืดหยุ่นเพื่อออกไปหาของกิน แบบนี้ถูกมั้ย",
"ก็คงตั้งตรงไม่ได้ แล้วถล่มลงมาเพราะรับน้ำหนักตัวเองไม่ไหว",
"เราว่ามันคงแปลกมาก เพราะถ้ามันมีคลอโรพลาสต์แต่วิ่งเร็วด้วย มันน่าจะเป็นระบบที่ใช้พลังงานสูงมาก"
]

script_bot = [
"เพราะมัวแต่จำชื่อหรือเปล่าอี้ ลองใช้แนวคิดแบบการสร้างธุรกิจดูนะ ถ้าพืชเป็นโรงงานที่ย้ายไปไหนไม่ได้เลย มันต้องมีอุปกรณ์ตัวไหนไว้ดึงพลังงานจากแดดมาสร้างอาหารเอง",
"เป็นคำถามที่ดี ในเมื่อสัตว์ไม่มีเครื่องผลิตพลังงานในตัว สัตว์เลยต้องใช้ระบบนำเข้า หรือการกินสิ่งมีชีวิตอื่นเข้าไปแทน",
"แข็งแกร่งขึ้นแน่ แต่มันจะติดปัญหาใหญ่เรื่องความคล่องตัว",
"ถูกแล้วอี้ เพราะมีผนังเซลล์ เซลล์พืชเลยเป็นเหลี่ยมแข็งแรง ส่วนเซลล์สัตว์จะมนและยืดหยุ่นกว่า",
"ดีมากอี้ ถ้างั้นก่อนแยกย้ายกันไป ลองทดสอบความเข้าใจสั้นๆ หน่อยนะ",
"วิเคราะห์ได้ขาดมากอี้! สรุปสั้นๆ วันนี้ที่เราคุยกันคือ พืชเน้นระบบพึ่งพาตัวเองด้วยคลอโรพลาสต์ และผนังเซลล์ ส่วนสัตว์เน้นความยืดหยุ่นและการเคลื่อนที่"
]

# -------------------------
# Display chat
# -------------------------
for msg in st.session_state.chat_history:

    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="bot-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

# -------------------------
# Play voice
# -------------------------
if st.session_state.last_audio:
    st.audio(st.session_state.last_audio, format="audio/mp3")
    st.session_state.last_audio = None

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Mic button
# -------------------------
st.markdown('<div class="mic-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="mic-circle">', unsafe_allow_html=True)

audio = mic_recorder(
    start_prompt="🎤",
    stop_prompt="⏹️",
    key=f"mic_{st.session_state.mic_key}"
)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Logic
# -------------------------
if audio and "bytes" in audio:

    user_text = speech_to_text(audio["bytes"])

    if st.session_state.step < len(script_user):

        user_script = script_user[st.session_state.step]

        st.session_state.chat_history.append(
            {"role": "user", "content": user_script}
        )

        bot_script = script_bot[st.session_state.step]

        st.session_state.chat_history.append(
            {"role": "assistant", "content": bot_script}
        )

        st.session_state.last_audio = text_to_speech(bot_script)

        st.session_state.step += 1
        st.session_state.mic_key += 1

        st.rerun()
