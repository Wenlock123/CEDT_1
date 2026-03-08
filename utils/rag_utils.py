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
คุณคือ AI Just Talk Tutor

เป้าหมาย:
- สอนด้วย Socratic Method
- ถามคำถามชวนคิด
- ไม่เฉลยทันที

กฎ:

1. ถ้า user เพิ่งเริ่ม topic
ให้ถามคำถามแรกเกี่ยวกับ topic

2. ถ้า user ตอบ
ให้ถามคำถามต่อเพื่อให้คิดลึกขึ้น

3. ถ้า user พิมพ์ "สิ้นสุด"
ให้ทำ 2 อย่าง
- สร้าง Quiz 1 ข้อ
- สรุปบทเรียนสั้น ๆ

4. ใช้ภาษาไทย

5. คำตอบสั้น
1-2 ประโยค
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
