import os
import json
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="DeepSEEK API")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Define allowed origins (add your frontend URL here)
origins = [
    "http://localhost:3000",  # Frontend running locally
    "https://se-project-rag.onrender.com",  # Backend URL
    "*"  # Add production frontend URL here
]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only specified origins
    allow_credentials=True,  # Allow cookies and credentials
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Initialize the language model (ChatGroq is used here)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define Alfred's prompt template
prompt = ChatPromptTemplate.from_template(
    """
You are 'Alfred', a friendly and knowledgeable assistant.
Answer the following question using the provided context.
Keep the answer brief, but ensure you cover all the essential aspects.
If it is Machine Learning related, aim for 100-150 words;
if it is a Python question, aim for 150-200 words.
Mention important points in bullets or highlight them.
Provide:
- Lecture video number (if applicable)
- Slide number (if applicable)
- Wikipedia link
- Google top link (non-Wikipedia)
Give all of these things after your RAG response.
Context:
{context}
Question:
{input}
"""
)

# -------------------------------
# Persistent FAISS Vector Store Setup
# -------------------------------
def build_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./pdf_files")  # Load PDFs from this folder
    docs = loader.load()  # Document loading

    # Attach metadata: PDF name and page/slide number
    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", "Unknown PDF")
        doc.metadata["page"] = doc.metadata.get("page", "Unknown Page")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs[:20])
    vectors = FAISS.from_documents(final_docs, embeddings)
    return vectors, embeddings

# Directory where the FAISS vector store is saved
FAISS_INDEX_DIR = "./faiss_index"
if os.path.exists(FAISS_INDEX_DIR):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Set allow_dangerous_deserialization=True only if you trust the persisted file
    vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    vector_store, embeddings = build_vector_store()
    vector_store.save_local(FAISS_INDEX_DIR)

# -------------------------------
# Persistent Chat History Setup
# -------------------------------
CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    return history

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f)

def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)

# -------------------------------
# Data Model for API Requests
# -------------------------------
class QueryRequest(BaseModel):
    query: str

# -------------------------------
# API Endpoints
# -------------------------------
@app.post("/ask")
def ask(query_request: QueryRequest):
    user_query = query_request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    history = load_chat_history()

    # Check if the query is for summarization
    if "summarize" in user_query.lower():
        if not history:
            return {"response": "No chat history available."}
        conversation = ""
        # For summarization, consider all available interactions
        for chat in history:
            conversation += f"User: {chat['query']}\nAlfred: {chat['answer']}\n"
        summarization_prompt = (
            "You are Alfred, a summarization assistant. "
            "Please summarize the following conversation history in concise bullet points:\n\n"
            + conversation +
            "\nSummary:"
        )
        summary_response = llm.invoke(summarization_prompt)
        summary_text = summary_response.content  # Use .content to extract text
        return {"response": summary_text}

    # Otherwise, process a normal query or a follow-up conversation
    if len(history) > 0:
        # Include prior conversation context if present
        context = "\n".join([f"User: {chat['query']}\nAlfred: {chat['answer']}" for chat in history])
        context += f"\nUser: {user_query}"
        retrieval_prompt = f"Based on this conversation:\n\n{context}\n\nProvide a response:"
        response_obj = llm.invoke(retrieval_prompt)
        response_text = response_obj.content
    else:
        # No prior context: use document-based retrieval
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": user_query})
        elapsed_time = time.process_time() - start_time
        response_text = response.get("answer", "")
        # Optionally, you can include the response time in the response
        # response_text = f"(Response Time: {elapsed_time:.2f} seconds) " + response_text

    # Append this conversation to chat history and save it persistently
    history.append({"query": user_query, "answer": response_text})
    save_chat_history(history)
    
    return {"response": response_text}

@app.post("/clear")
def clear():
    clear_chat_history()
    return {"message": "Chat history cleared. You can start a new conversation."}
