import whisper
import tempfile

# -------------------------
# โหลด Whisper model
# -------------------------

model = whisper.load_model("small")


# -------------------------
# แก้คำที่ Whisper ฟังผิดบ่อย
# -------------------------

def fix_common_errors(text):

    corrections = {
        "เซว": "เซลล์",
        "พืด": "พืช",
        "สัด": "สัตว์"
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    return text


# -------------------------
# Speech to Text
# -------------------------

def speech_to_text(audio_bytes):

    # สร้างไฟล์เสียงชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        temp_audio_path = tmp.name

    # ให้ Whisper ถอดเสียง
    result = model.transcribe(
        temp_audio_path,
        language="th",

        # เพิ่ม prompt ช่วยให้เดาคำชีวะได้ถูก
        initial_prompt="เซลล์พืช เซลล์สัตว์ โครงสร้างเซลล์ ผนังเซลล์ คลอโรพลาสต์ ไมโทคอนเดรีย",

        # ลดการเดาคำมั่ว
        temperature=0
    )

    text = result["text"]

    # แก้คำที่ฟังผิด
    text = fix_common_errors(text)

    return text
