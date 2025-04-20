# build_vectorstore.py
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 0: Remove existing vector store if it exists
if os.path.exists(DB_FAISS_PATH):
    print("🧹 Removing old vector store...")
    shutil.rmtree(DB_FAISS_PATH)

# Step 1: Load all PDFs
def load_pdf_files(data_path):
    print("📄 Loading PDF documents...")
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

documents = load_pdf_files(DATA_PATH)

# Step 2: Split documents into chunks
def create_chunks(docs):
    print("🔪 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_documents(docs)

text_chunks = create_chunks(documents)

# Step 3: Load embedding model
def get_embedding_model():
    print("🤖 Loading embedding model...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Step 4: Create and save new FAISS vector store
print("📦 Creating FAISS vector store...")
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print(f"✅ FAISS vector store built and saved at: {DB_FAISS_PATH}")
