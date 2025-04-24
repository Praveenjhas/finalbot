import os
import shutil
import io
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# IMPORT SOLUTION - Try both possible import locations
try:
    # New recommended import location
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        # Fallback to community version
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        # Final fallback - install required package
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-community"])
        from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure Tesseract path (update if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_pdf_files(data_path):
    print("📄 Loading PDF documents...")
    documents = []
    
    for filename in os.listdir(data_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            print(f"🔍 Processing: {filename}")
            
            try:
                with fitz.open(file_path) as pdf:
                    if pdf.is_encrypted:
                        print(f"   ⚠️ Encrypted PDF: {filename}")
                        continue
                        
                    text = ""
                    for page_num, page in enumerate(pdf, 1):
                        try:
                            page_text = page.get_text()
                            if not page_text.strip():
                                pix = page.get_pixmap()
                                img_bytes = pix.tobytes("png")
                                img = Image.open(io.BytesIO(img_bytes))
                                page_text = pytesseract.image_to_string(img, lang='eng+hin')
                                
                            text += f"\n[Page {page_num}]\n{page_text}\n"
                        except Exception as page_error:
                            print(f"   ❌ Page {page_num} error: {str(page_error)}")
                            continue
                    
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": filename}
                        ))
                    else:
                        print(f"⚠️ Skipped empty PDF: {filename}")
                        
            except Exception as e:
                print(f"❌ Failed to process {filename}: {str(e)}")
                
    return documents

def main():
    DATA_PATH = "data"
    DB_FAISS_PATH = "vectorstore/db_faiss"
    
    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH)
    
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        raise ValueError("No valid documents found")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    # Initialize embeddings - this will now work
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Vector store created with {len(chunks)} chunks")

if __name__ == "__main__":
    main()
