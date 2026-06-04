# =====================================================
# COMPLETE RAG SYSTEM (GPU SAFE)
# =====================================================

# ----------- FORCE CPU ------------
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
API = os.getenv("GOOGLE_API_KEY")

import torch
torch.cuda.is_available = lambda: False

# ----------- IMPORTS ------------
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =====================================================
# 1. CONFIGURE GEMINI
# =====================================================

genai.configure(api_key=API)

# Change to gemini-2.5-flash if available
model = genai.GenerativeModel("gemini-2.5-flash")


# =====================================================
# 2. PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


# =====================================================
# 3. CHUNKING WITH OVERLAP
# =====================================================

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def chunk_by_questions(text):
    processed = text.split("\n\n")
    return processed

# =====================================================
# 4. LOAD EMBEDDING MODEL (CPU ONLY)
# =====================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# =====================================================
# 5. CREATE CHROMADB COLLECTION
# =====================================================

chroma_client = chromadb.Client()

DB1 = chroma_client.create_collection(
    name="answer_keys_collection"
)
DB2 = chroma_client.create_collection(
    name="answer_paper_collection"
)


# =====================================================
# 6. STORE EMBEDDINGS IN CHROMADB
# =====================================================

def store_chunks(DB,chunks):
    for index,chunk in enumerate(chunks):
        embedding = embedding_model.encode(
            chunk,
            convert_to_numpy=True
        ).tolist()

        DB.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"q_{index}"]
        )

# =====================================================
# 9. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    answer_key_raw=input("Answer Key File Path:")
    answer_key_text = extract_text_from_pdf(answer_key_raw)
    chunks = chunk_by_questions(answer_key_text)
    store_chunks(DB1,chunks)
    
    answer_paper_raw=input("Answer Key File Path:")
    answer_paper_text = extract_text_from_pdf(answer_paper_raw)
    chunks = chunk_by_questions(answer_paper_text)
    store_chunks(DB2,chunks)

    print("AI evaluation id started using GEMINI:-")

    for i in range(len(chunks)):
        query = f"""
        Question: {chunks[i]}
        Answer Key: {DB1.get(ids=[f"q_{i}"])['documents'][0]}
        Answer Paper: {DB2.get(ids=[f"q_{i}"])['documents'][0]}
        Evaluate the answer paper against the answer key and give a score out of 10.
        """

        response = model.generate_content(query)
        print(f"Question {i+1} Score: {response.text}\n")