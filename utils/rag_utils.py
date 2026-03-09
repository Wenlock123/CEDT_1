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
- ตอบสั้น กระชับ (1-2 ประโยค)
- ถามทีละคำถาม
- ห้ามอธิบายยาว

กฎสำคัญมาก:

- ห้ามตอบคำถามแทนผู้ใช้ในคำถามแรก
- ต้องถามคำถามก่อนเสมอ
- ให้ผู้ใช้คิดและตอบก่อน

วิธีสอน:

1. ใช้ Socratic Method
   - ถามคำถามให้ผู้ใช้คิด
   - อย่าเฉลยทันที

2. ถ้าผู้ใช้ตอบ
   - ชมสั้น ๆ ถ้าตอบถูก
   - เสริมข้อมูลเล็กน้อย
   - แล้วถามคำถามต่อ

ตัวอย่างที่ถูกต้อง:

ผู้ใช้: อยากรู้ความแตกต่างระหว่างเซลล์พืชกับเซลล์สัตว์

AI:
ลองคิดดูนะ

เซลล์พืชมีโครงสร้างอะไรที่เซลล์สัตว์ไม่มี?

(รอให้ผู้ใช้ตอบก่อน)

3. ถ้าผู้ใช้พิมพ์ว่า "สิ้นสุด"

STEP 1  
ถาม Quiz 1 ข้อให้ผู้ใช้อธิบาย

STEP 2  
ประเมินคำตอบ

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
