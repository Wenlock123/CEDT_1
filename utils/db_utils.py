import zipfile
import os

def extract_chromadb():

    if not os.path.exists("chromadb_database"):

        with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
            zip_ref.extractall()

        print("ChromaDB extracted")
