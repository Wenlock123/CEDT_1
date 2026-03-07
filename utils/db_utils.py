import zipfile
import os

def extract_chromadb():

    if not os.path.exists("chroma_db"):

        with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
            zip_ref.extractall()

        print("ChromaDB extracted")
