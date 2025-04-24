import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory

# ✅ Paths
DB_FAISS_PATH = "vectorstore/db_faiss"

# ✅ Enhanced memory to remember plant type
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# ✅ LLM setup (OpenRouter API with Mistral-7B)
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=st.secrets["openrouter"]["api_key"],
    model="mistralai/mistral-7b-instruct",
    temperature=0.7,
    max_tokens=1024,
    streaming=True,
)

# ✅ Improved prompt template for HyDE
prompt_template = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
You are a friendly, knowledgeable assistant helping users with plant-related questions.

Your task is to analyze the user's current question along with the previous conversation to determine whether the query is about a specific plant, and then generate a thoughtful, context-aware response.

---

🔄 **Chat History**:
{chat_history}

❓ **Current Question**:
{question}

---

🧠 **Instructions**:

1. **Plant Detection**: 
   - If the user asks about pests, diseases, or weed control and **mentions a specific plant** (e.g., "tomatoes", "roses", "wheat"), assume the response should focus on that plant.
   - If **no plant is mentioned** in the current question, check the **chat history** for any previously mentioned plant and use that context.

2. **Clarification Strategy**:
   - If no specific plant is found in the question or chat history, respond with:
     _"Could you please let me know which plant you're growing? I’ll provide you with advice tailored specifically for that crop."_

3. **Content Style**:
   - Prioritize **organic methods** when discussing weed or pest control.
   - Include **chemical treatments** only if asked or if organic options are not effective.
   - Keep tone **professional, empathetic, and informative**, like a helpful advisor or agronomist.

4. **Goal**:
   - Provide a clear, specific, and actionable answer that reflects the user's context and improves their growing experience.

---

✅ **Now generate a high-quality, human-like response that follows these principles.**
"""
)

# ✅ LLMChain for HyDE
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# ✅ Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ HyDE wrapper around base embeddings
hyde_embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embedding_model
)

# ✅ Load FAISS vector store
vectorstore = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ✅ Retriever with HyDE
retriever = vectorstore.as_retriever(embedding=hyde_embeddings)

# ✅ Custom QA Chain with plant identification
def get_response(qa_chain, user_query, chat_history):
    # Check if this is a follow-up about plant type
    if "Which plant are you growing?" in chat_history and len(chat_history.split("\n")) == 1:
        # User is responding with plant type
        memory.save_context({"question": chat_history}, {"answer": f"User is growing: {user_query}"})
        return "Thank you! Now I can give you specific advice for your " + user_query + ". Could you please repeat your original question?"
    
    # Normal question processing
    response = qa_chain.invoke({"question": user_query})
    answer = response["answer"]
    
    # Check if we need to ask about plant type
    if any(keyword in user_query.lower() for keyword in ["weed", "pest", "disease", "control"]) and not any(plant in chat_history.lower() for plant in ["tomato", "rose", "plant", "growing"]):
        if "which plant" not in answer.lower():
            answer = "Which plant are you growing? This will help me give tailored advice for weed control."
    
    return answer

# ✅ QA Chain with memory and source documents
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# ✅ Start chat loop
print("\n🌱 Plant Care Chatbot is ready. Ask about weed control or other plant care topics!\n(Type 'exit' to stop)\n")

try:
    while True:
        user_query = input("🧑 You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Happy gardening!")
            break

        # Get chat history from memory
        chat_history = memory.load_memory_variables({})["chat_history"]
        chat_history_str = "\n".join([msg.content for msg in chat_history])
        
        # Get response
        response = get_response(qa_chain, user_query, chat_history_str)
        
        print("\n🤖 Bot:")
        print(response)

        # Show sources only if it's not a plant identification question
        if "which plant" not in response.lower():
            result = qa_chain.invoke({"question": user_query})
            print("\n📚 Sources:")
            for doc in result["source_documents"]:
                print("-", doc.metadata.get("source", "Unknown"))
        
        print("\n" + "-"*60 + "\n")

except KeyboardInterrupt:
    print("\n👋 Exiting due to keyboard interrupt.")
