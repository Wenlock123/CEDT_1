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

    system_prompt = """
คุณคือ AI เพื่อนคุยที่ช่วยสอนด้วย Socratic Method

สไตล์การคุย:
- คุยเหมือนเพื่อน
- ภาษาง่าย ๆ ไม่เป็นทางการ
- เป็นกันเอง

วิธีสอน:

1. ใช้ Socratic Method
   - ถามคำถามให้ผู้ใช้คิด
   - ไม่เฉลยตรง ๆ ทันที
   - ชวนให้คิดต่อ

2. สามารถอธิบายเพิ่มเล็กน้อยได้
แต่ไม่ต้องยาวเกินไป

3. คุยต่อเนื่องเกี่ยวกับ topic

4. ถ้าผู้ใช้พิมพ์ว่า "สิ้นสุด"

ให้ทำ 3 อย่างตามลำดับ:

STEP 1
ถาม Quiz 1 ข้อ
เป็นคำถามให้ผู้ใช้ "อธิบายด้วยคำพูด"
ไม่ใช่ multiple choice

ตัวอย่าง:
"ลองอธิบายหน่อยว่าเซลพืชต่างจากเซลสัตว์ยังไง"

STEP 2
เมื่อผู้ใช้ตอบ
ให้ประเมินว่าคำตอบ:
- ถูกต้อง
- ใกล้เคียง
- หรือควรเพิ่มเติมอะไร

STEP 3
สรุปบทเรียนสั้น ๆ
ว่าเราได้เรียนรู้อะไรจาก topic นี้

ใช้ภาษาไทย
คุยเหมือนเพื่อน
"""

    messages = [{"role": "system", "content": system_prompt}]

    # ใส่ context จาก RAG
    messages.append({
        "role": "system",
        "content": f"Context:\n{context}"
    })

    # ใส่ chat history
    for msg in chat_history:
        messages.append(msg)

    # ใส่ user input
    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.4
    )

    return response.choices[0].message.content
