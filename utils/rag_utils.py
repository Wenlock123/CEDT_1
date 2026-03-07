from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

# -------------------------
# Load environment variables
# -------------------------

load_dotenv()

# -------------------------
# Embedding model
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# -------------------------
# Load ChromaDB (โหลดครั้งเดียว)
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

def retrieve_context(question):

    docs = retriever.get_relevant_documents(question)

    if not docs:
        return ""

    context = "\n".join([doc.page_content for doc in docs])

    return context


# -------------------------
# LLM (Groq)
# -------------------------

def ask_llm(question, context):

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    system_prompt = """
คุณคือ AI Tutor ที่ใช้ Socratic Method

กฎการตอบ:

1. ใช้ Socratic method
- ตั้งคำถามชวนคิด
- ไม่เฉลยตรง ๆ
- ให้ผู้เรียนคิดต่อ

2. ถ้ามีข้อมูลจาก Context
- ใช้ข้อมูลนั้นเป็นพื้นฐาน

3. จำกัดคำตอบให้สั้น
ไม่เกิน 1-2 ประโยค

4. รูปแบบคำตอบ
- คำถามชวนคิด
หรือ
- hint สั้น ๆ

5. ห้ามอธิบายยาว

6. ใช้ภาษาไทย
"""

    user_prompt = f"""
Context จากฐานข้อมูล:

{context}

คำถามของผู้เรียน:
{question}

ตอบโดยใช้ Socratic method และจำกัดคำตอบไม่เกิน 1-2 ประโยค
"""

    response = client.chat.completions.create(

        model="meta-llama/llama-4-scout-17b-16e-instruct",

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],

        temperature=0.4
    )

    return response.choices[0].message.content
