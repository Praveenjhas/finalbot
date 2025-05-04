import warnings
import asyncio
import re
import os
from googletrans import Translator
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# --- Setup asyncio loop ---
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# --- Translator ---
class HinglishTranslator:
    def __init__(self):
        self.trans = Translator()
        self.technical_terms = {
            'soil': 'मिट्टी',
            'cultivation': 'खेती',
            'fertilizer': 'उर्वरक',
            'strawberry': 'स्ट्रॉबेरी',
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

# --- Load FAISS vectorstore ---
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
    # Normalize plant names and query
      plants = [f.split('.')[0].lower().replace("_", " ") for f in os.listdir("data") if f.endswith('.pdf')]
      query = query.lower().strip(",.!?")
    
    # Match all forms (singular/plural/hyphenated)
      for plant in plants:
        pattern = rf'\b{re.escape(plant)}s?\b'  # Matches "lettuce" and "lettuces"
        if re.search(pattern, query):
            return plant.replace(" ", "_")  # Maintain original metadata format
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

# --- Enhanced Response Handler ---
# --- Enhanced Response Handler ---
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
        'score_threshold': 0.5
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
# --- Main Chat Interface ---
def main():
    print("\n🌱 Agri-Assistant (English/Hinglish) - Type 'exit' to quit\n")
    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break
        
        answer, sources = loop.run_until_complete(get_response(user_input))
        
        print(f"\n🤖 Assistant:\n{answer}\n")
        if sources:
            print("📚 Sources:")
            for src in sources:
                print(f"- {src}")
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()
