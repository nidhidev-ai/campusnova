# rag.py — Lightweight RAG for CampusNova
# Uses TF-IDF instead of sentence-transformers
# Much lighter — works on Render free tier!

import os
import pickle
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: Extract text from PDF
# ─────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num + 1}]\n{page_text}"
    return text

# ─────────────────────────────────────────
# STEP 2: Split text into chunks
# ─────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-10:]
            current_size = sum(len(w) + 1 for w in current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# ─────────────────────────────────────────
# STEP 3: Store chunks
# ─────────────────────────────────────────
def process_and_store_pdf(pdf_path: str, doc_name: str, category: str):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    chunks_path = os.path.join(VECTOR_STORE_PATH, "chunks.pkl")

    if os.path.exists(chunks_path):
        with open(chunks_path, 'rb') as f:
            stored_chunks = pickle.load(f)
    else:
        stored_chunks = []

    for chunk in chunks:
        stored_chunks.append({
            'text': str(chunk),
            'doc_name': str(doc_name),
            'category': str(category)
        })

    with open(chunks_path, 'wb') as f:
        pickle.dump(stored_chunks, f)

    return len(chunks)

# ─────────────────────────────────────────
# STEP 4: Search relevant chunks
# Uses TF-IDF — lightweight and fast!
# ─────────────────────────────────────────
def search_relevant_chunks(question: str, top_k: int = 3) -> list:
    chunks_path = os.path.join(VECTOR_STORE_PATH, "chunks.pkl")

    if not os.path.exists(chunks_path):
        return []

    with open(chunks_path, 'rb') as f:
        stored_chunks = pickle.load(f)

    if not stored_chunks:
        return []

    # Get all chunk texts
    texts = [chunk['text'] for chunk in stored_chunks]

    # Add question to texts for TF-IDF
    all_texts = texts + [question]

    # Calculate TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Compare question vector with all chunks
    question_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(question_vector, chunk_vectors)[0]

    # Get top_k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append(stored_chunks[idx])

    return results
