from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os

# -------------------------
# Embedding model
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# -------------------------
# Load ChromaDB
# -------------------------

def load_vectorstore():

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    return vectorstore


# -------------------------
# Retrieve context
# -------------------------

def retrieve_context(question):

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.get_relevant_documents(question)

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

กฎสำคัญ:

1. ใช้วิธี Socratic
- ตั้งคำถามชวนคิด
- ไม่เฉลยทันที
- ช่วยให้ผู้เรียนคิดเอง

2. ถ้ามีข้อมูลจาก Context
- ใช้ข้อมูลนั้นตอบ
- จำกัดคำตอบ 1-2 ประโยค

3. รูปแบบคำตอบที่ต้องการ
- ตั้งคำถามชวนคิด
หรือ
- ให้ Hint สั้น ๆ

4. ห้ามอธิบายยาว

5. ใช้ภาษาไทย

ตัวอย่างคำตอบที่ดี:

ถ้าเซลล์พืชสร้างอาหารเอง คุณคิดว่าโครงสร้างใดในเซลล์ที่เกี่ยวข้องกับการใช้พลังงานจากแสง?

หรือ

ใน context มีโครงสร้างที่เกี่ยวกับการสังเคราะห์แสง คุณจำได้ไหมว่ามันชื่ออะไร?
"""

    user_prompt = f"""
Context จากฐานข้อมูล:

{context}

คำถามของผู้เรียน:
{question}

ตอบโดยใช้ Socratic method และจำกัด 1-2 ประโยค
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
