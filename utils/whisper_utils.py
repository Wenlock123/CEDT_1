import whisper
import tempfile

# โหลดโมเดลครั้งเดียว
model = whisper.load_model("small")


def speech_to_text(audio_bytes):

    # สร้างไฟล์ชั่วคราวสำหรับเก็บเสียง
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        temp_audio_path = tmp.name

    # ให้ whisper ถอดเสียง
    result = model.transcribe(
        temp_audio_path,
        language="th"
    )

    return result["text"]
