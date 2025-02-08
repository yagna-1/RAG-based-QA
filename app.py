import os
import re
import tempfile
import faiss
import numpy as np
import streamlit as st
from io import StringIO
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pdfplumber
from PyPDF2 import PdfReader
from docx import Document

# ========================
# Configuration Defaults
# ========================
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_GENERATOR_MODEL = "google/flan-t5-large"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_ANSWER_LENGTH = 512
DEFAULT_TEMPERATURE = 0.9

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
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_GENERATOR_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_GENERATOR_MODEL)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if st._is_running_with_streamlit else -1
    )

# ===============================
# Document Processing Functions
# ===============================
def extract_text(file_path: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    
    try:
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages])
        elif ext in [".doc", ".docx"]:
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
    
    return re.sub(r"\s+", " ", text).strip()

def semantic_chunking(text: str, max_chars: int = 1000) -> list:
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_chars:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# ===============================
# FAISS Index & Retrieval Functions
# ===============================
def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def retrieve_context(query, embedder, index, chunks, top_k: int):
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ===============================
# Generation & Evaluation Functions
# ===============================
def generate_answer(query: str, context: str, gen_pipeline) -> str:
    prompt = f"""
    Based on the following context, answer the question in detail.
    If you don't know the answer, say "I don't have enough information".
    
    Context: {context}
    
    Question: {query}
    
    Detailed Answer:
    """
    
    return gen_pipeline(
        prompt,
        max_length=DEFAULT_MAX_ANSWER_LENGTH,
        temperature=DEFAULT_TEMPERATURE,
        repetition_penalty=1.2,
        num_beams=4,
        do_sample=True,
        early_stopping=True
    )[0]["generated_text"]

# ===============================
# UI Components
# ===============================
def sidebar_controls():
    st.sidebar.header("Settings & Documents")
    
    return {
        "top_k": st.sidebar.slider("Context Sources", 1, 10, DEFAULT_TOP_K),
        "max_length": st.sidebar.slider("Max Answer Length", 128, 1024, DEFAULT_MAX_ANSWER_LENGTH),
        "temperature": st.sidebar.slider("Creativity", 0.1, 1.0, DEFAULT_TEMPERATURE),
        "files": st.sidebar.file_uploader(
            "Upload Documents", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
    }

def main_interface():
    st.title("Document Intelligence Assistant")
    st.markdown("""
    **Upload documents** (PDF, DOCX, TXT) and ask questions. 
    The AI will analyze content and provide detailed answers.
    """)
    
    query = st.text_input("Ask your question:", key="query_input")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Generate Answer"):
            handle_query(query)
    
    with col2:
        if st.button("Clear Session"):
            reset_session()

# ===============================
# Core Logic
# ===============================
def handle_query(query: str):
    if not validate_session():
        return
    
    with st.spinner("Analyzing documents and generating response..."):
        try:
            context_chunks = retrieve_context(
                query,
                st.session_state.embedding_model,
                st.session_state.faiss_index,
                st.session_state.chunk_texts,
                st.session_state.top_k
            )
            context = "\n\n".join(context_chunks)
            
            answer = generate_answer(
                query,
                context,
                st.session_state.gen_model
            )
            
            display_results(query, answer, context)
            handle_feedback(answer, query)
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

def validate_session():
    if not st.session_state.get("faiss_index"):
        st.error("Please upload documents first!")
        return False
    if not st.session_state.get("gen_model"):
        st.session_state.gen_model = load_gen_model()
    return True

# ===============================
# Core Logic (Fixed Version)
# ===============================
def display_results(query: str, answer: str, context: str):
    st.subheader("Answer")
    st.markdown(f"**{answer}**")
    
    with st.expander("See context sources"):
        st.markdown(context)
    
    with st.expander("Technical details"):
        # Define newline character separately
        nl = '\n'
        st.markdown(f"""
        - Retrieved context passages: {len(context.split(nl + nl))}
        - Answer length: {len(answer.split())} words
        - Context relevance score: {calculate_relevance(query, context):.2f}/1.0
        """)

def calculate_relevance(query: str, context: str) -> float:
    query_embed = st.session_state.embedding_model.encode([query])
    context_embed = st.session_state.embedding_model.encode([context])
    return np.dot(query_embed, context_embed.T)[0][0]

def handle_feedback(answer: str, query: str):
    with st.form("feedback_form"):
        rating = st.radio("Rate this answer:", 
            ["Excellent", "Good", "Fair", "Poor"], index=1)
        comment = st.text_area("Additional comments:")
        
        if st.form_submit_button("Submit Feedback"):
            log_feedback(rating, comment, answer, query)
            st.success("Feedback recorded!")

def log_feedback(rating: str, comment: str, answer: str, query: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "rating": rating,
        "comment": comment
    }
    st.session_state.feedback_log.append(entry)

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# ===============================
# Document Processing
# ===============================
def process_uploaded_files(files):
    if not files:
        return False
    
    all_text = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            text = extract_text(tmp_file.name, file.name)
            if text:
                all_text.append(text)
    
    if all_text:
        combined_text = "\n\n".join(all_text)
        st.session_state.chunk_texts = semantic_chunking(combined_text)
        
        if not st.session_state.embedding_model:
            st.session_state.embedding_model = load_embedding_model()
            
        st.session_state.faiss_index = build_faiss_index(
            st.session_state.chunk_texts,
            st.session_state.embedding_model
        )
        
        st.sidebar.success(f"Processed {len(files)} files with {len(st.session_state.chunk_texts)} chunks")
        return True
    return False

# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    controls = sidebar_controls()
    st.session_state.top_k = controls["top_k"]
    
    if process_uploaded_files(controls["files"]):
        main_interface()
