import whisper

model = whisper.load_model("base")

def speech_to_text(audio_path):

    result = model.transcribe(
        audio_path,
        language="th"
    )

    return result["text"]
