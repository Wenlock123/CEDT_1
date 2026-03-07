import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# embedding model
embedding_model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2"
)

# load chromadb
chroma_client = chromadb.PersistentClient(
    path="chromadb_database"
)

collection = chroma_client.get_collection(
    name="knowledge"
)

# groq client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_llm(question):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
You are a helpful AI assistant.

Your task is to answer the user's question using the information from the provided context.

Guidelines:
- Use the context as the primary source of information.
- Summarize relevant information.
- Keep the answer concise and clear.
- Answer in 1-2 sentences.
- Focus only on information relevant to the question.
- Avoid unnecessary details.

Context:
{context}

User Question:
{question}

Answer in Thai:
"""

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content
