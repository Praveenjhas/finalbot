import asyncio
import warnings
from googletrans import Translator
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress coroutine warning
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), "torch_cache")
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
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
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
# ---- Set Your API Key Here ----
class HinglishTranslator:
    def __init__(self):
        self.trans = Translator()
        self.technical_terms = {
            'soil': 'मिट्टी',
            'cultivation': 'खेती',
            'fertilizer': 'उर्वरक',
            'strawberry': 'स्ट्रॉबेरी',
            'tomato': 'टमाटर',
            'wheat': 'गेहूँ'
        }
    
    async def detect_lang(self, text: str) -> str:
        try:
            detected = await self.trans.detect(text)
            return 'hi' if detected.lang == 'hi' else 'en'
        except:
            return 'en'
    
    async def to_english(self, text: str) -> str:
        try:
            for hin, eng in {v:k for k,v in self.technical_terms.items()}.items():
                text = text.replace(hin, eng)
            return (await self.trans.translate(text, src='hi', dest='en')).text
        except:
            return text
    
    async def to_hindi(self, text: str) -> str:
        try:
            result = (await self.trans.translate(text, src='en', dest='hi')).text
            for eng, hin in self.technical_terms.items():
                result = result.replace(eng, hin)
            return result
        except:
            return text

translator = HinglishTranslator()
# Then modify your get_vectorstore() function:
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={'device':'cpu'}
)
vectorstore = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# --- Plant Management ---
class PlantContextManager:
    def __init__(self):
        self.current_plant = None
        self.awaiting_confirmation = False
        self.last_recommendation = None
    
    def extract_plant(self, query: str) -> str:
        plants = [f.split('.')[0].lower() for f in os.listdir("data") if f.endswith('.pdf')]
        for plant in plants:
            if re.search(rf'\b{plant}\b', query, re.IGNORECASE):
                return plant
        return None
    
    def should_use_general(self, query: str) -> bool:
        general_keywords = ['general']
        return any(re.search(rf'\b{k}\b', query, re.IGNORECASE) for k in general_keywords)
    
    def is_affirmative(self, query: str) -> bool:
        affirmatives = ['yes', 'ok', 'sure', 'agree', 'correct', 'right']
        return any(re.search(rf'\b{k}\b', query, re.IGNORECASE) for k in affirmatives)

plant_manager = PlantContextManager()

# --- LLM & Prompt Setup ---
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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",
    model="google/gemma-3-27b-it",
    temperature=0.7,
    max_tokens=1024,
    streaming=False,
    max_retries=5,
    request_timeout=60
)


# --- LLM & Prompt Setup ---

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
# --- build  universal chain per plant ---
qa_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history", "current_plant"],
    template="""
**Agricultural Expert System** 🌱

**Crop**: {current_plant}
**Query**: {question}
**Conversation History**: {chat_history}

**Relevant Technical Documentation**:
{context}

**Response Guidelines**:

1. **Knowledge Base Protocol**:
   - Base responses exclusively on the provided technical documentation
   - If documentation is insufficient, state: "Available data suggests: [partial answer]. For comprehensive guidance, consult regional agricultural authorities."
   - Never reference undocumented crops or practices

2. **Response Structure**:
   give answer in  bullet points
   - **Technical Specifications**: Present numerical values (pH, NPK ratios, temperatures) in table format


3. **Tone & Style**:
   - Maintain FAO report-level professionalism
   - Use subheadings and bold key terms for scanability
   - Employ metric units with imperial conversions in parentheses

4. **Treatment Prioritization**:
   - Lead with organic/IPM solutions
   - Chemical interventions:
     • List only when explicitly requested
     • Format as: "Chemical Option (Last Resort): [Product] - [Application Rate]"

5. **Continuation Protocol**:
   - Conclude with some other one  followup question and also remember that which can be asked as a followup for that based on the context.
---------------------------------
"""
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# --- Conversational Chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": qa_prompt,
        "document_variable_name": "context"
    },
    return_source_documents=True,
    output_key="answer"
)

async def get_response(user_query: str):
    global plant_manager
    
    # Detect language and translate if needed
    lang = await translator.detect_lang(user_query)
    translated_query = await translator.to_english(user_query) if lang == 'hi' else user_query
    
    # Handle plant context
    plant = plant_manager.extract_plant(translated_query)
    if(plant_manager.should_use_general(user_query) and (not plant)):
        plant="general"
    current_plant = plant or plant_manager.current_plant or "general"
    
    # Update plant context
    if plant:
        plant_manager.current_plant = plant
    
    # Prepare the input dictionary with ALL required keys
    inputs = {
        "question": translated_query,
        "current_plant": current_plant,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    }
    
    # Configure retriever with plant filter
    qa_chain.retriever.search_kwargs = {
        'filter': {'plant': current_plant},
        'k': 3,
        'score_threshold': 0.75
    }
    
    try:
        # Get QA response with proper input
        qa_response = await qa_chain.ainvoke(inputs)
        
        # Process sources and response
        sources = list(set(
            os.path.basename(d.metadata["source"]) 
            for d in qa_response["source_documents"]
        ))
        
        response_text = qa_response["answer"]
        
        # Store last recommendation context
        plant_manager.last_recommendation = response_text
        
        # Translate if needed
        if lang == 'hi':
            response_text = await translator.to_hindi(response_text)
        
        return response_text, sources
    
    except Exception as e:
        print(f"Error: {str(e)}")
        error_msg = "Please try again with a different question."
        if lang == 'hi':
            error_msg = await translator.to_hindi(error_msg)
        return error_msg, []


# ----------------- MAIN APP -----------------
def main():
    
    st.cache_resource.clear()
    st.set_page_config(page_title="Plant Chatbot ", layout="centered")
    st.markdown(custom_css, unsafe_allow_html=True)
    st.title("Chatbot – Ask Your Questions!")

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

    prompt = st.chat_input("Hey,Ask something about your plant")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("✍ Thinking...")
            answer, sources = loop.run_until_complete(get_response(prompt))
            displayed_text = ""
            full_answer=answer
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
