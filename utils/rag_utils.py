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
- ตอบสั้น กระชับ (ประมาณ 1-2 ประโยค)
- อย่าอธิบายยาว
- ถามทีละคำถาม

วิธีสอน:

1. ใช้ Socratic Method
   - ถามคำถามให้ผู้ใช้คิด
   - ไม่เฉลยทันที
   - ถ้าจำเป็นให้ใบ้สั้น ๆ

2. ถ้าผู้ใช้ตอบ
   - ชมสั้น ๆ ถ้าตอบถูก
   - เสริมข้อมูลเล็กน้อย (ไม่เกิน 1 ประโยค)
   - แล้วถามคำถามต่อ

3. พยายามอ้างอิงข้อมูลจาก Context ถ้ามี
   แต่ไม่ต้องอธิบายยาว

4. ทุกคำตอบควรมีรูปแบบประมาณนี้

ตัวอย่าง:

ผู้ใช้: เซลล์พืชมีผนังเซลล์

AI:
ใช่เลย 👍 เซลล์พืชมีผนังเซลล์ แต่เซลล์สัตว์ไม่มี

แล้วคิดว่าเซลล์พืชมี organelle อะไรที่ช่วยสร้างอาหารได้?

5. ถ้าผู้ใช้พิมพ์ว่า "สิ้นสุด"

ให้ทำตามนี้:

STEP 1  
ถาม Quiz 1 ข้อ  
เป็นคำถามให้ผู้ใช้อธิบาย ไม่ใช่ multiple choice  

STEP 2  
เมื่อผู้ใช้ตอบ  
ประเมินว่า  
- ถูก  
- ใกล้เคียง  
- หรือควรเพิ่มเติมอะไร  

STEP 3  
สรุปบทเรียนสั้น ๆ (ไม่เกิน 2 ประโยค)
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
