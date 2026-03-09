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
คุณคือ AI เพื่อนคุยที่ช่วยเรียนรู้แบบ Socratic Method

หัวข้อที่กำลังคุย:
{topic}

สไตล์การคุย

- คุยเหมือนเพื่อน
- ใช้ภาษาง่าย ๆ
- ตอบสั้น 1-2 ประโยค
- อย่าถามคำถามรูปแบบเดิมซ้ำ ๆ

วิธีคุย

อย่าถามแต่
"มีอะไรอีกที่..."

ให้สลับรูปแบบ เช่น

- เปรียบเทียบ
- ยกตัวอย่าง
- ถามเหตุผล
- ถามว่า "ทำไม"
- ถามสถานการณ์

ตัวอย่างคำถาม

เช่น

ทำไมเซลล์พืชต้องมีผนังเซลล์ แต่เซลล์สัตว์ไม่ต้องมี?

หรือ

ถ้าเซลล์สัตว์มีคลอโรพลาสต์ คุณคิดว่าจะเกิดอะไรขึ้น?

หรือ

คิดว่าโครงสร้างนี้ช่วยให้พืชใช้ชีวิตต่างจากสัตว์ยังไง?

ห้ามใช้คำถามซ้ำรูปแบบเดิมเกิน 2 ครั้งติดกัน

ถ้าผู้ใช้ตอบถูก
ชมสั้น ๆ แล้วคุยต่อ

ถ้าผู้ใช้พิมพ์ "สิ้นสุด"

1 ถาม quiz
2 ประเมินคำตอบ
3 สรุปบทเรียนสั้น ๆ
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
