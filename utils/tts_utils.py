import edge_tts
import asyncio

async def generate_audio(text):

    communicate = edge_tts.Communicate(
        text,
        voice="th-TH-PremwadeeNeural"
    )

    await communicate.save("response.mp3")


def text_to_speech(text):

    asyncio.run(generate_audio(text))

    return "response.mp3"
