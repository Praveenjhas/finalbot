import os
import shutil
import io
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure Tesseract path
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
                        plant_name = os.path.splitext(filename)[0].lower()
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": filename, 
                                "plant": plant_name,
                                "title": plant_name.capitalize()
                            }
                        ))
                    else:
                        print(f"⚠️ Skipped empty PDF: {filename}")
                        
            except Exception as e:
                print(f"❌ Failed to process {filename}: {str(e)}")
                
    return documents
def create_vector_store(documents, db_path):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "•", ":", "(?<=\d\))", "(?<=\.)", " "]
    )
    chunks = splitter.split_documents(documents)
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(db_path)
    print(f"✅ Vector store created with {len(chunks)} chunks")
    return db

def main():
    DATA_PATH = "data"
    DB_FAISS_PATH = "vectorstore/db_faiss"
    
    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH)
    
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        raise ValueError("No valid documents found")
    
    create_vector_store(documents, DB_FAISS_PATH)

if __name__ == "__main__":
    main()
