st.cache_resource.clear() 
import asyncio
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress coroutine warning
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), "torch_cache")
import io
import re
import json
import time
import uuid
from pathlib import Path
import shutil
import streamlit as st
from gtts import gTTS
from datetime import datetime
from langdetect import detect
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ---- Set Your API Key Here ----

DB_FAISS_PATH = "vectorstore/db_faiss"

# Then modify your get_vectorstore() function:
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    # 1. Check if vectorstore exists
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Vectorstore path {DB_FAISS_PATH} not found")
    
    # 2. Get latest modified times
    data_mtime = max(os.path.getmtime(f) for f in Path("data").glob("*.pdf"))
    vs_mtime = os.path.getmtime(DB_FAISS_PATH)
    
    # 3. Rebuild if PDFs are newer than vectorstore
    if data_mtime > vs_mtime:
        st.warning("New PDFs detected - rebuilding vectorstore...")
        from langchain.document_loaders import DirectoryLoader
        loader = DirectoryLoader("data", glob="*.pdf")
        docs = loader.load()
        
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Create fresh vectorstore
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(DB_FAISS_PATH)
        st.success("Vectorstore updated!")
    
    # Load existing store
    return FAISS.load_local(
        DB_FAISS_PATH,
        HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
        allow_dangerous_deserialization=True
    )

def load_llm_openrouter():
    return ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=st.secrets["openrouter"]["api_key"],
    default_headers={
        "HTTP-Referer": "http://localhost",  # Replace with your real domain if deployed
        "User-Agent": "LangChain-App"
    },
    model="mistralai/mistral-7b-instruct"
)


def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

def extract_sources(source_documents):
    sources = set()
    for doc in source_documents:
        if 'source' in doc.metadata:
            sources.add(doc.metadata['source'])
    return list(sources)

def format_sources(sources):
    if not sources:
        return ""
    lines = ["\n*Sources:*"]
    for idx, src in enumerate(sources, 1):
        lines.append(f"{idx}. {src}")
    return "\n".join(lines)

def highlight_keywords(text):
    keywords = ["strawberry", "disease", "soil", "climate", "water", "fertilizer"]
    for kw in keywords:
        text = re.sub(fr"\b({kw})\b", r"\1**", text, flags=re.IGNORECASE)
    return text

# ----------------- MAIN APP -----------------
def main():
    st.set_page_config(page_title="Strawberry Chatbot 🍓", layout="centered")
    st.markdown(custom_css, unsafe_allow_html=True)
    st.title("🍓 Strawberry Chatbot – Ask Your Questions!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "🧑"):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                audio_key = f"audio_{idx}"
                btn_key = f"btn_{idx}"
                if st.button("🔊 Generate Audio", key=btn_key):
                    with st.spinner("Generating audio..."):
                        audio_data = text_to_speech(message["content"], message.get("lang", "en"))
                        st.session_state.audio_cache[audio_key] = audio_data
                        st.rerun()
                if audio_key in st.session_state.audio_cache:
                    st.audio(st.session_state.audio_cache[audio_key], format="audio/mp3")

    prompt = st.chat_input("Ask something about strawberries...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("✍ Thinking...")

            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm_openrouter(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PromptTemplate(
                    template="""
Answer the user's question using the context provided below and the knowledge that you have about the plant.
If the context contains relevant information, use it directly. If the context is insufficient, include your own knowledge but make sure it's relevant.
Cite the sources used.

Context:
{context}

Question:
{question}

Answer:
""", input_variables=["context", "question"]
                )}
            )

            response = qa_chain.invoke({"query": prompt})
            answer = highlight_keywords(response["result"])
            sources = format_sources(extract_sources(response["source_documents"]))
            full_answer = answer + "\n\n" + sources

            displayed_text = ""
            for char in full_answer:
                displayed_text += char
                placeholder.markdown(displayed_text + "▌")
                time.sleep(0.008)
            placeholder.markdown(displayed_text)

            lang_code = detect_language(full_answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "lang": lang_code
            })

            new_audio_key = f"audio_{len(st.session_state.messages)-1}"
            new_btn_key = f"btn_{len(st.session_state.messages)-1}"

            if st.button("🔊 Generate Audio", key=new_btn_key):
                with st.spinner("Generating audio..."):
                    audio_data = text_to_speech(full_answer, lang_code)
                    st.session_state.audio_cache[new_audio_key] = audio_data
                    st.rerun()

            if new_audio_key in st.session_state.audio_cache:
                st.audio(st.session_state.audio_cache[new_audio_key], format="audio/mp3")

    # Sidebar tools
    with st.sidebar:
        st.header("🛠 Tools")
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.session_state.audio_cache = {}
            st.rerun()
        st.download_button(
            label="📄 Download .txt",
            data="\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        st.download_button(
            label="🧾 Download .json",
            data=json.dumps(st.session_state.messages, indent=2),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ----------------- CSS -----------------
custom_css = """
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    .stMarkdown {
        font-size: 16px;
        line-height: 1.7;
    }
    .stChatInput input {
        padding: 12px;
        font-size: 16px;
        border-radius: 12px !important;
        border: 1px solid #ccc;
    }
    .stButton > button {
        border-radius: 10px;
        padding: 6px 12px;
        font-weight: bold;
    }
    audio {
        width: 100%;
        margin-top: 10px;
    }
</style>
"""
if __name__ == "__main__":
    main()
