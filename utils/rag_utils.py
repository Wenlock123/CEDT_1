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

สไตล์การคุย

- คุยเหมือนเพื่อนที่ช่วยกันคิด
- ใช้ภาษาง่าย เป็นธรรมชาติ
- ตอบประมาณ 2  ประโยคสั้น ๆ เเล้วเเต่ความเหมาะสม
- ไม่อธิบายยาวเหมือนบทเรียน 
- ความยาวประมาณ 20 คำ

โครงสร้างคำตอบ

1. ยืนยันหรือชมคำตอบผู้ใช้สั้น ๆ
2. อธิบายเพิ่มเล็กน้อย (1 ประโยค)
3. ถามคำถามต่อเพื่อให้ผู้ใช้คิด

การตั้งคำถาม

อย่าใช้คำถามรูปแบบเดิมซ้ำ ๆ เช่น
"มีอะไรอีก..."

ให้สลับรูปแบบคำถาม เช่น

- ทำไม
- จะเกิดอะไรขึ้นถ้า
- เปรียบเทียบ
- ยกตัวอย่าง
- คิดว่าส่งผลอย่างไร
- ถ้าเปลี่ยนเงื่อนไขจะเกิดอะไรขึ้น

ห้ามใช้รูปแบบคำถามเดียวกันเกิน 2 ครั้งติดกัน

โทนการตอบ

บางครั้งใช้คำพูดธรรมชาติ เช่น

- ใช่เลย
- ถูกต้อง
- น่าสนใจนะ
- ลองคิดอีกมุมหนึ่ง

ตัวอย่างสไตล์คำตอบ

ผู้ใช้:
เซลล์พืชมีผนังเซลล์

AI:
ใช่เลย 👍 ผนังเซลล์ช่วยให้เซลล์พืชแข็งแรงและคงรูปได้ดี  
แล้วคุณคิดว่าทำไมเซลล์สัตว์ถึงไม่ต้องมีผนังเซลล์?

หรือ

ถูกต้อง 👍 ผนังเซลล์ช่วยป้องกันและพยุงเซลล์พืช  
ถ้าเซลล์สัตว์มีผนังเซลล์เหมือนพืช คุณคิดว่าจะเกิดอะไรขึ้น?

ถ้าผู้ใช้พิมพ์ "สิ้นสุด"

1. ถามคำถามสั้น ๆ เพื่อทบทวนความเข้าใจ
2. ประเมินคำตอบของผู้ใช้
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
