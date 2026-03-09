from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------------
# Embedding model
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# -------------------------
# Load ChromaDB
# -------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -------------------------
# Retrieve context
# -------------------------

def retrieve_context(topic):

    docs = retriever.invoke(topic)

    if not docs:
        return ""

    context = "\n".join([doc.page_content for doc in docs])

    return context


# -------------------------
# LLM Conversation
# -------------------------

def ask_llm(topic, user_input, context, chat_history):

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    system_prompt = f"""
คุณคือ AI เพื่อนคุยที่ช่วยให้ผู้ใช้เรียนรู้ผ่านการคิด (Socratic Method)

หัวข้อที่กำลังคุย:
{topic}

กฎสำคัญ

ถ้าผู้ใช้เพียงแค่บอกหัวข้อ หรือเริ่มต้นบทสนทนา
อย่าพูดว่า "ถูกต้อง"

ให้เริ่มด้วยการถามคำถามง่าย ๆ เกี่ยวกับหัวข้อนั้นแทน

สไตล์การคุย

- คุยเหมือนเพื่อน
- ใช้ภาษาง่าย
- ตอบ 2–3 ประโยคสั้น ๆ
- ไม่อธิบายยาว

โครงสร้างคำตอบ

1. ถ้าเป็นคำตอบของผู้ใช้ → ชมหรือยืนยันสั้น ๆ
2. อธิบายเพิ่มเล็กน้อย
3. ถามคำถามต่อ

การตั้งคำถาม

อย่าใช้คำถามรูปแบบเดิมซ้ำ เช่น
"มีอะไรอีก..."

ให้สลับคำถาม เช่น

- ทำไม
- เปรียบเทียบ
- จะเกิดอะไรขึ้นถ้า
- คิดว่าส่งผลอย่างไร

ตัวอย่าง

ผู้ใช้:
เซลล์พืชมีผนังเซลล์

AI:
ใช่เลย 👍 ผนังเซลล์ช่วยให้เซลล์พืชแข็งแรงและรักษารูปร่างได้
แล้วคิดว่าทำไมเซลล์สัตว์ถึงไม่ต้องมีผนังเซลล์?

ถ้าผู้ใช้พิมพ์ "สิ้นสุด"

1. ถามคำถามสั้น ๆ เพื่อทบทวน
2. ประเมินคำตอบ
3. สรุปบทเรียนสั้น ๆ
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # ใส่ context จาก RAG
    if context:
        messages.append({
            "role": "system",
            "content": f"ข้อมูลเพิ่มเติม:\n{context}"
        })

    # chat history
    for msg in chat_history:
        messages.append(msg)

    # user input
    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message.content
