# rag.py — The brain of CampusNova
# This file handles all RAG operations

import os
import pickle
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# Load the sentence transformer model
# This converts text into numbers (vectors)
# so we can search through them
# ─────────────────────────────────────────
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Where we store processed chunks
VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: Extract text from PDF
# ─────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    """Read PDF and return all text"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Add page number marker
                text += f"\n[Page {page_num + 1}]\n{page_text}"
    return text

# ─────────────────────────────────────────
# STEP 2: Split text into chunks
# ─────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Break text into small overlapping chunks
    chunk_size = how many characters per chunk
    overlap = how many characters shared between chunks
    (overlap helps so answers don't get cut off)
    """
    chunks = []
    words = text.split()
    
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            # Keep last few words for overlap
            overlap_words = current_chunk[-10:]
            current_chunk = overlap_words
            current_size = sum(len(w) + 1 for w in overlap_words)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# ─────────────────────────────────────────
# STEP 3: Store chunks in vector database
# ─────────────────────────────────────────
def process_and_store_pdf(pdf_path: str, doc_name: str, category: str):
    """
    Full pipeline:
    PDF → Extract text → Chunk → Convert to vectors → Store
    """
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # Split into chunks
    chunks = chunk_text(text)
    
    # Convert chunks to vectors (numbers AI can search)
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    # Load or create FAISS index
    index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_PATH, "chunks.pkl")
    
    if os.path.exists(index_path):
        # Load existing index and add to it
        index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            stored_chunks = pickle.load(f)
    else:
        # Create new index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        stored_chunks = []
    
    # Add new chunks
    index.add(embeddings)
    for chunk in chunks:
        stored_chunks.append({
        'text': str(chunk),
        'doc_name': str(doc_name),
        'category': str(category)
    })
    
    # Save everything
    faiss.write_index(index, index_path)
    with open(chunks_path, 'wb') as f:
        pickle.dump(stored_chunks, f)
    
    return len(chunks)

# ─────────────────────────────────────────
# STEP 4: Search for relevant chunks
# ─────────────────────────────────────────
def search_relevant_chunks(question: str, top_k: int = 3) -> list:
    """
    Convert question to vector
    Search FAISS for most similar chunks
    Return top_k most relevant chunks
    """
    index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    chunks_path = os.path.join(VECTOR_STORE_PATH, "chunks.pkl")
    
    if not os.path.exists(index_path):
        return []
    
    # Load index and chunks
    index = faiss.read_index(index_path)
    with open(chunks_path, 'rb') as f:
        stored_chunks = pickle.load(f)
    
    # Convert question to vector
    question_vector = model.encode([question])
    question_vector = np.array(question_vector).astype('float32')
    
    # Search for similar chunks
    distances, indices = index.search(question_vector, top_k)
    
    # Return relevant chunks
    results = []
    for idx in indices[0]:
        if idx < len(stored_chunks):
            results.append(stored_chunks[idx])
    
    return results
