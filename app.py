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
    padding-bottom:140px;
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

/* mic area bottom center */
.mic-wrapper{
    position:fixed;
    bottom:35px;
    left:50%;
    transform:translateX(-50%);
    z-index:999;
}

/* hide default mic button */
button[kind="secondary"]{
    background:transparent !important;
    border:none !important;
    font-size:32px !important;
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
    "เราอยากรู้ว่าความเเตกต่างระหว่างเซลล์พืชกับเซลล์สัตว์คืออะไร แพรช่วยบอกเราหน่อยสิ",
    "เซลล์พืชมีผนังเซลล์แต่เซลล์สัตว์ไม่มีรึเปล่าแพร",
    "เพราะพืชไม่สามารถเคลื่อนที่ได้ ต้องยืนต้นตลอดเวลา แต่สัตว์ไม่ต้องมีเพราะสัตว์เคลื่อนที่ได้ไงแพร",
    "ใช่แล้ว แล้วนอกจากนี้เซลล์พืชมีคลอโรพลาสเอาไว้สร้างอาหารแต่เซลล์สัตว์ต้องกินสิ่งมีชีวิตอื่นเป็นอาหารใช่มั้ย",
    "โอเคได้สิ ที่เซลล์พืชสร้างผนังเซลล์เป็นกำแพงแข็งเพื่อค้ำจุนโครงสร้างให้ยืนต้นรับแสงแดดได้โดยไม่ต้องขยับ ส่วนสัตว์ตัดกำแพงนี้ทิ้งเพื่อให้เซลล์ยืดหยุ่นและเคลื่อนที่ไปหาแหล่งอาหารได้อย่างอิสระ"
]

script_bot = [
    "ได้เลยอี้ รอบที่แล้วที่เราคุยกันเรื่องยีนสนุกมากเลย แล้วอี้คิดว่าเซลล์พืชกับเซลล์สัตว์ต่างกันยังไงอ่ะ",
    "แล้วอี้ลองวิเคราะห์ดูสิว่า การที่พืชมีผนังเซลล์แต่สัตว์ไม่มี มันตอบโจทย์วิถีชีวิตที่ต่างกันของพวกมันยังไง",
    "อืม... ในเมื่อความแข็งของผนังเซลล์ทำให้พืชยืนต้นนิ่งๆ ได้ แล้วสัตว์ที่ต้องเคลื่อนที่ตลอดเวลา ธรรมชาติมีวิธีออกแบบโครงสร้างมาค้ำจุนร่างกายแทนผนังเซลล์ยังไงบ้างล่ะ",
    "วิเคราะห์ได้เฉียบมากอี้! งั้นก่อนจะไปเรื่องคลอโรพลาสต์ อี้ช่วยสรุปภาพรวมของกลยุทธ์ 'กำแพงแข็ง' vs 'ความยืดหยุ่น' ของพืชกับสัตว์ในมุมของอี้ให้ฟังหน่อยสิ",
    "โห้ เหมือนแนวคิดแบบ first principles thinking ของ Elon Musk ที่อี้ชอบเลย เราคุยเรื่องการสร้างอาหารของพืชและสัตว์กันต่อเลยมั้ยอี้"
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
# Mic button bottom
# -------------------------
st.markdown('<div class="mic-wrapper">', unsafe_allow_html=True)

audio = mic_recorder(
    start_prompt="🎤",
    stop_prompt="⏹️",
    key=f"mic_{st.session_state.mic_key}"
)

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
