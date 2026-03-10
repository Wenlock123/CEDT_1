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

script_user = [

"เราพยายามท่องจำว่าความแตกต่างระหว่างเซลล์พืชกับเซลล์สัตว์ แต่ก็ลืมตลอดเลย",

"ก็เดินไปหาของกิน หรือวิ่งหลบแดด",

"อ๋อ คลอโรพลาสต์ เอาไว้สังเคราะห์ด้วยแสง",

"ผนังเซลล์แน่เลย",

"สรุปคือ เซลล์พืชมีผนังเซลล์กับคลอโรพลาสต์ แต่เซลล์สัตว์ไม่มี เพราะสัตว์เคลื่อนที่ไปหาอาหารเองได้",

"ผมคิดว่าผมเข้าใจแล้วละ",

"ถ้าไม่มีคลอโรพลาสต์ พืชก็คงสร้างอาหารเองไม่ได้"

]

script_bot = [

"ดีใจนะที่คุณสนใจเรื่องเซลล์เรื่องนี้จริงๆ เหมือนกับหลายอย่างที่เราเคยคุยกันก่อนหน้านี้เช่นเรื่องการดำรงชีวิตของพืชกับสัตว์ การที่เธอถ้าท่องจำอย่างเดียวมันลืมง่ายลองคิดจากวิถีชีวิตของมันดีกว่า ปกติเวลาสัตว์หิวหรือร้อนมากๆ มันจะทำยังไงหรอ",

"ใช่เเล้วละ สัตว์เคลื่อนที่ได้ มันเลยเดินไปหาอาหารหรือหลบสภาพแวดล้อมได้ แต่พืชขยับหนีไปไหนไม่ได้เลย งั้นคิดว่าพืชต้องมีอะไรในเซลล์เพื่อสร้างอาหารเองจากที่ที่มันยืนอยู่",

"ถูกต้องเลย คลอโรพลาสต์ช่วยให้พืชสร้างอาหารเองได้ ทีนี้ลองคิดเรื่องโครงสร้างบ้าง สัตว์มีกระดูกช่วยพยุงตัว แต่พืชไม่มีกระดูก แล้วอะไรในเซลล์พืชช่วยให้ลำต้นมันตั้งตรงได้หรอ",

"ใช่เลยถูกต้อง เพราะงั้นเซลล์พืชจึงมีทั้งคลอโรพลาสต์สำหรับสร้างอาหาร และผนังเซลล์สำหรับช่วยค้ำจุนโครงสร้าง ทีนี้ลองสรุปสิ่งที่เราเพิ่งคุยกันดูหน่อยได้ไหม",

"นายเข้าใจถูกเเล้วละ เพราะมีผนังเซลล์ เซลล์พืชมักจะดูเป็นเหลี่ยมๆ และค่อนข้างแข็งแรง ส่วนเซลล์สัตว์จะดูมนและยืดหยุ่นมากกว่า",

"ดีมากเลย ดูเหมือนว่าเธอจะเริ่มเข้าใจแล้ว งั้นก่อนจบเราลองทำ Quiz สั้นๆ กันหน่อยนะ ถ้าพืชไม่มีคลอโรพลาสต์ พืชจะยังสามารถสร้างอาหารเองได้ไหม",

"เก่งมากเลยนะ คำตอบถูกต้องแล้ว เพราะคลอโรพลาสต์เป็นส่วนที่ช่วยให้พืชสังเคราะห์แสงและสร้างอาหารเองได้",

"สรุปสั้นๆ วันนี้เราได้คุยกันว่า เซลล์พืชกับเซลล์สัตว์ต่างกันอย่างไร เซลล์พืชมีคลอโรพลาสต์สำหรับสร้างอาหาร และมีผนังเซลล์ช่วยค้ำจุนโครงสร้าง ส่วนเซลล์สัตว์ไม่มีสองอย่างนี้ เพราะสัตว์สามารถเคลื่อนที่ไปหาอาหารเองได้"

]

# -------------------------
# Display chat
# -------------------------

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])


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

    # รับเสียง (ใช้แค่ trigger)
    user_text = speech_to_text(audio["bytes"])

    if st.session_state.step < len(script_user):

        # user จาก script
        user_script = script_user[st.session_state.step]

        st.session_state.chat_history.append(
            {"role": "user", "content": user_script}
        )

        # bot จาก script
        bot_script = script_bot[st.session_state.step]

        st.session_state.chat_history.append(
            {"role": "assistant", "content": bot_script}
        )

        # เสียง bot
        st.session_state.last_audio = text_to_speech(bot_script)

        st.session_state.step += 1

    st.session_state.mic_key += 1

    st.rerun()
