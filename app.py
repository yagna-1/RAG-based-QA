import os
import re
import tempfile
import faiss
import numpy as np
import streamlit as st
from io import StringIO
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pdfplumber
from PyPDF2 import PdfReader
from docx import Document
from nltk.translate.bleu_score import sentence_bleu

# ========================
# Configuration Defaults
# ========================
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

DEFAULT_GENERATOR_MODEL = "EleutherAI/gpt-neo-125M"
# Increased max length to allow more detailed answers.
DEFAULT_GENERATOR_MAX_LENGTH = 300  
DEFAULT_GENERATOR_TEMPERATURE = 0.8  # Adjusted temperature for slightly more creativity

# ========================
# Session State Initialization
# ========================
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "gen_model" not in st.session_state:
    st.session_state.gen_model = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunk_texts" not in st.session_state:
    st.session_state.chunk_texts = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# ===============================
# Caching Model Loading Functions
# ===============================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def load_gen_model():
    return pipeline("text-generation", model=DEFAULT_GENERATOR_MODEL, tokenizer=DEFAULT_GENERATOR_MODEL)

# ===============================
# Document Processing Functions
# ===============================
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.write("pdfplumber error:", e, "; trying PyPDF2...")
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e2:
            st.write("PyPDF2 error:", e2)
    return text

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.write("DOCX extraction error:", e)
        return ""

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.write("TXT extraction error:", e)
        return ""

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def load_document(file_path: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".doc", ".docx"]:
        text = extract_text_from_docx(file_path)
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        st.error("Unsupported file format!")
        return ""
    return clean_text(text)

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# ===============================
# FAISS Index & Retrieval Functions
# ===============================
def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def retrieve_chunks(query, embedder, index, chunks, top_k: int):
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ===============================
# Generation & Evaluation Functions
# ===============================
def generate_answer(query, context, gen_pipeline):
    # Modified prompt to instruct the model to generate a detailed answer.
    prompt = (
        f"Using the context provided below, give a detailed, complete answer in full sentences to the question. "
        f"If there is insufficient information, say 'Insufficient information to provide an answer.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Detailed Answer:"
    )
    generated = gen_pipeline(
        prompt,
        max_length=DEFAULT_GENERATOR_MAX_LENGTH,
        do_sample=True,
        temperature=DEFAULT_GENERATOR_TEMPERATURE
    )
    answer = generated[0]["generated_text"]
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    return answer

def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    return sentence_bleu([ref_tokens], hyp_tokens)

# ===============================
# Helper: Save Feedback Log
# ===============================
def log_feedback(rating: str, comment: str, answer: str, query: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "rating": rating,
        "comment": comment
    }
    st.session_state.feedback_log.append(entry)
    st.success("Feedback submitted!")

# ===============================
# Sidebar: Settings and File Upload
# ===============================
st.sidebar.header("Settings & Document Upload")

# Parameter sliders
top_k = st.sidebar.slider("Top-K Retrieval", min_value=1, max_value=10, value=DEFAULT_TOP_K)
chunk_size = st.sidebar.slider("Chunk Size (characters)", min_value=200, max_value=1000, value=DEFAULT_CHUNK_SIZE, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", min_value=0, max_value=300, value=DEFAULT_CHUNK_OVERLAP, step=10)

# File uploader (allow multiple files)
uploaded_files = st.sidebar.file_uploader("Choose document(s)", type=["pdf", "doc", "docx", "txt"], accept_multiple_files=True)

# Clear session button
if st.sidebar.button("Clear Document & Reset"):
    st.session_state.uploaded_files = []
    st.session_state.chunk_texts = []
    st.session_state.faiss_index = None
    st.session_state.embedding_model = None
    st.experimental_rerun()

# Process uploaded files
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    all_text = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        file_text = load_document(tmp_file_path, uploaded_file.name)
        all_text += file_text + "\n"
    if all_text:
        st.sidebar.success("Documents loaded successfully!")
        # Split text into chunks using slider parameters
        st.session_state.chunk_texts = split_text_into_chunks(all_text, chunk_size, chunk_overlap)
        # Load embedding model if not already loaded
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = load_embedding_model()
        # Build FAISS index
        st.session_state.faiss_index = build_faiss_index(st.session_state.chunk_texts, st.session_state.embedding_model)
        st.sidebar.success(f"Document indexed into {len(st.session_state.chunk_texts)} chunks.")

# ===============================
# Main App: Ask Question & Get Answer
# ===============================
st.title("Multi‑Source Document QA System with Advanced Features")
st.markdown(
    """
    **Overview:**  
    Upload one or more documents (PDF, DOCX, or TXT), adjust parameters, and then ask a question.  
    The system retrieves relevant document chunks and generates an answer using a local generative model.
    """
)

query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    if st.session_state.faiss_index is None or not st.session_state.chunk_texts:
        st.error("Please upload document(s) first!")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            retrieved = retrieve_chunks(query, st.session_state.embedding_model, st.session_state.faiss_index,
                                        st.session_state.chunk_texts, top_k)
            context = "\n\n".join(retrieved)
            if st.session_state.gen_model is None:
                st.session_state.gen_model = load_gen_model()
            answer = generate_answer(query, context, st.session_state.gen_model)
            st.subheader("Answer")
            st.write(answer)
            
            # Optionally show retrieved context
            if st.checkbox("Show retrieved context used for answer"):
                st.markdown("### Retrieved Context")
                st.text_area("Context", value=context, height=200)
            
            # Optional BLEU evaluation if reference answer provided
            st.markdown("---")
            st.subheader("Optional Evaluation")
            reference = st.text_area("Enter reference answer (if available):")
            if st.button("Compute BLEU Score"):
                if reference:
                    bleu = compute_bleu(reference, answer)
                    st.write(f"BLEU Score: {bleu:.4f}")
                else:
                    st.info("Please provide a reference answer for evaluation.")
            
            # Feedback section
            st.markdown("---")
            st.subheader("Provide Feedback")
            rating = st.radio("Rate the answer:", options=["Excellent", "Good", "Average", "Poor"])
            comment = st.text_area("Additional comments (optional):")
            if st.button("Submit Feedback"):
                log_feedback(rating, comment, answer, query)
            
            # Download retrieved context
            if st.button("Download Retrieved Context"):
                buffered = StringIO()
                buffered.write(context)
                buffered.seek(0)
                st.download_button(label="Download Context as Text File", data=buffered,
                                   file_name="retrieved_context.txt", mime="text/plain")

# ===============================
# Display Feedback Log (if any)
# ===============================
if st.session_state.feedback_log:
    st.markdown("---")
    st.subheader("Feedback Log")
    for entry in st.session_state.feedback_log:
        st.markdown(f"**Time:** {entry['timestamp']}")
        st.markdown(f"**Query:** {entry['query']}")
        st.markdown(f"**Answer:** {entry['answer']}")
        st.markdown(f"**Rating:** {entry['rating']}")
        st.markdown(f"**Comment:** {entry['comment']}")
        st.markdown("---")
