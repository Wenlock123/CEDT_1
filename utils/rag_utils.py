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
คุณคือ AI เพื่อนติวเตอร์ที่คุยกับผู้ใช้ด้วย Socratic Method

หัวข้อที่กำลังเรียน:
{topic}

กฎสำคัญ:

- ห้ามทักทาย
- เริ่มถามเกี่ยวกับหัวข้อนี้ทันที
- คุยเหมือนเพื่อน ไม่เป็นทางการ
- ใช้คำง่าย ๆ
- ถามทีละคำถาม
- ไม่ต้องอธิบายยาว

วิธีสอน:

1. ใช้ Socratic Method
   - ถามให้ผู้ใช้คิด
   - ไม่เฉลยทันที

2. ถ้าผู้ใช้ตอบ
   - ช่วยชี้แนะเล็กน้อย
   - แล้วถามต่อ

3. พยายามอ้างอิงข้อมูลจาก Context ถ้ามี

4. ถ้าผู้ใช้พิมพ์ว่า "สิ้นสุด"

ให้ทำตามนี้:

STEP 1
ถามคำถาม Quiz 1 ข้อ
ให้ผู้ใช้ "อธิบาย"

STEP 2
ประเมินคำตอบ
- ถูก
- ใกล้เคียง
- หรือควรเพิ่มอะไร

STEP 3
สรุปบทเรียนสั้น ๆ
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
