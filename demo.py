#  Research Navigator App 
import streamlit as st
import pdfplumber, docx, os, tempfile, re, requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#TEXT EXTRACTION
# Extract text from PDF file
def extract_text_from_pdf(file_obj):
    text_pages = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)  # Join all pages

# Extract text from DOCX file
def extract_text_from_docx(file_obj):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    tmp.write(file_obj.read())  # Write uploaded file to temp
    tmp.close()
    doc = docx.Document(tmp.name)
    texts = [p.text for p in doc.paragraphs if p.text.strip()]  # Get non-empty paragraphs
    os.unlink(tmp.name)  # Delete temp file
    return "\n".join(texts)

# Clean extra spaces from text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Split text into smaller chunks for processing
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

#  MODEL LOADING
# Load all ML models 
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return summarizer, embedder, qa_pipeline

# Create embeddings for text chunks
def embed_sentences(chunks, embedder):
    embs = embedder.encode(chunks, convert_to_numpy=True)
    embs = embs.astype('float32')
    faiss.normalize_L2(embs)
    return embs

# Build FAISS index for semantic search
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# Search relevant chunks using semantic similarity
def semantic_search(query, index, chunks, embedder, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# WEB SCRAPING
# Scrape text from webpage
def scrape_webpage(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return clean_text(" ".join(paragraphs))
    except Exception as e:
        return f"Error fetching page: {e}"

#STREAMLIT UI 
st.set_page_config(page_title="Research Navigator", layout="wide", page_icon="📄")

#styles for UI
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #e3f2fd, #fce4ec); padding: 30px; border-radius: 15px; }
    h1,h2,h3,h4 { color: #2e3b4e !important; font-family: 'Poppins', sans-serif; }
    .stButton>button { background-color: #4CAF50; color: white; font-weight: 600; border-radius: 8px; padding: 8px 18px; border: none; }
    .stButton>button:hover { background-color: #43a047; }
    .stTextInput>div>input { background-color: #e3f2fd; border-radius: 5px; padding: 8px; }
    .stTabs [data-baseweb="tab-list"] { background-color: #e8f5e9; border-radius: 10px; padding: 10px; }
    .stTabs [data-baseweb="tab"] { color: #1b5e20; font-weight: 600; }
    .stTabs [data-baseweb="tab-highlight"] { background: #66bb6a; }
    .stExpander { background-color: #f1f8e9; border-left: 5px solid #66bb6a; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# App title and subtitle
st.title("Research Navigator: Semantic Paper Analyzer")
st.markdown("<p style='color:#1b5e20; font-size:20px;'>Summarize, Chat, or Scrape research content instantly</p>", unsafe_allow_html=True)

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Document Summarizer", "Chatbot", "Web Scraper"])

#TAB 1: DOCUMENT SUMMARIZER
with tab1:
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf","docx"])
    if uploaded_file:
        if "pdf" in uploaded_file.type:
            text = extract_text_from_pdf(uploaded_file)
        else:
            uploaded_file.seek(0)
            text = extract_text_from_docx(uploaded_file)
        text = clean_text(text)
        
        summarizer, _, _ = load_models()
        
        if st.button("->Generate Summary"):
            chunks = chunk_text(text, chunk_size=500, overlap=50)  # Split text
            summaries = [summarizer(c, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] for c in chunks]
            final_summary = summarizer(" ".join(summaries), max_length=200, min_length=50, do_sample=False)[0]['summary_text']
            st.success("Summary Generated")
            st.write(final_summary)
    else:
        st.info("Please upload a PDF or DOCX file to continue.")

#TAB 2: CHATBOT
with tab2:
    uploaded_file = st.file_uploader("Upload the same document for chat", type=["pdf","docx"], key="chat_upload")
    if uploaded_file:
        if "pdf" in uploaded_file.type:
            text = extract_text_from_pdf(uploaded_file)
        else:
            uploaded_file.seek(0)
            text = extract_text_from_docx(uploaded_file)
        text = clean_text(text)
        
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        summarizer, embedder, qa_pipeline = load_models()
        
        embeddings = embed_sentences(chunks, embedder)  # Get embeddings
        index = build_faiss_index(embeddings)  # Build FAISS index
        
        question = st.text_input("=>Ask your question here...")
        if question:
            top_passages = semantic_search(question, index, chunks, embedder, top_k=3)
            if top_passages:
                context = " ".join(top_passages)
                answer = qa_pipeline(question=question, context=context)  # Get answer
                st.markdown(f"**Answer:** <span style='color:#43a047;'>{answer['answer']}</span>", unsafe_allow_html=True)
                st.caption(f"Confidence Score: {answer['score']:.3f}")
                with st.expander("View Supporting Context"):
                    for p in top_passages:
                        st.write("- " + p)
            else:
                st.warning("No relevant context found.")
    else:
        st.info("Upload a document to chat with it.")

#TAB 3: WEB SCRAPER
with tab3:
    st.subheader("=>Scrape & Summarize Web Content")
    url = st.text_input("Enter a webpage URL to summarize:")
    if st.button("=>Scrape & Summarize"):
        if url:
            with st.spinner("Scraping and summarizing webpage..."):
                page_text = scrape_webpage(url)
                if "Error" not in page_text:
                    summarizer, _, _ = load_models()
                    chunks = chunk_text(page_text, chunk_size=500, overlap=50)
                    summaries = [summarizer(c, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] for c in chunks]
                    final_summary = summarizer(" ".join(summaries), max_length=200, min_length=50, do_sample=False)[0]['summary_text']
                    st.success("=>Webpage Summary Generated")
                    st.write(final_summary)
                else:
                    st.error(page_text)
        else:
            st.warning("Please enter a valid URL.")
