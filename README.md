# AI-Research-Navigator

🧭 Research Navigator: Semantic Paper Analyzer

An AI-powered web app built with Streamlit that helps researchers and students analyze, summarize, and interact with research documents and online content.
The app can:

📄 Summarize PDF/DOCX research papers

💬 Chat with uploaded documents using semantic search + QA models

🌐 Scrape and summarize academic web pages

🚀 Features
🧩 1. Document Summarizer

Upload a PDF or Word file to get a concise AI-generated summary of the content.

Supports .pdf and .docx

Automatically extracts, cleans, and chunks the text

Uses a pretrained BART summarization model

🤖 2. Document Chatbot

Ask questions about your uploaded research paper.

Embeds document chunks using SentenceTransformer (MiniLM)

Builds a FAISS similarity index for semantic search

Uses DistilBERT Question Answering to generate context-aware answers

🌍 3. Web Scraper

Scrape research content from webpages and summarize it instantly.

Extracts text from <p> tags using BeautifulSoup

Summarizes the cleaned text using the same summarization pipeline

🛠️ Tech Stack
Category	Technology Used
Framework	Streamlit
NLP Models	🤗 Hugging Face Transformers (distilbart-cnn, distilbert-base-cased, all-MiniLM-L6-v2)
Semantic Search	FAISS
Text Extraction	pdfplumber, python-docx
Web Scraping	BeautifulSoup, requests
Language	Python 3.8+
📦 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/research-navigator.git
cd research-navigator

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the app
streamlit run app.py


Then open http://localhost:8501
 in your browser.

📁 Project Structure
ResearchNavigator/
│
├── app.py                     # Main Streamlit application
├── requirements.txt            # Dependencies list
├── README.md                   # Documentation (this file)
└── assets/                     # (Optional) Store icons, images, etc.

⚙️ Key Functions Overview
Function	Purpose
extract_text_from_pdf()	Extracts all text from uploaded PDF files
extract_text_from_docx()	Extracts text from Word documents
chunk_text()	Splits long text into overlapping chunks for model input
load_models()	Loads summarizer, embedder, and QA models
embed_sentences()	Generates sentence embeddings for chunks
build_faiss_index()	Builds FAISS index for semantic similarity
semantic_search()	Finds top relevant chunks for a given question
scrape_webpage()	Fetches and cleans webpage text
🧠 Models Used
Model	Task	Source
sshleifer/distilbart-cnn-12-6	Text Summarization	🤗 Transformers
distilbert-base-cased-distilled-squad	Question Answering	🤗 Transformers
all-MiniLM-L6-v2	Sentence Embeddings	SentenceTransformers
🎨 UI Styling

Custom CSS styling is used to enhance the Streamlit interface:

Gradient backgrounds

Styled buttons and tabs

Highlighted context expanders

⚡ Example Use Cases

📚 Researchers – Quickly summarize long academic papers.

🧑‍🎓 Students – Chat with uploaded notes or research articles.

📰 Writers – Summarize content from blogs or websites.

🔍 Analysts – Extract key information from online reports.

🧾 License

This project is licensed under the MIT License — you’re free to use and modify it.

🤝 Contributing

Pull requests are welcome!
If you’d like to contribute:

Fork the repo

Create a feature branch

Submit a PR

👩‍💻 Author

Shaik Anisah Firdaws
