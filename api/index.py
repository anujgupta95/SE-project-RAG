import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import LangChain and related modules
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")  # Your Google Custom Search API Key
google_cx = os.getenv("GOOGLE_CX")  # Your Google Custom Search Engine CX

app = FastAPI()

# Initialize the language model (ChatGroq in this example)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Set up the prompt for Alfred
prompt = ChatPromptTemplate.from_template(
    """
You are 'Alfred', a friendly and knowledgeable assistant.
Answer the following question using the provided context.
Keep the answer short but also ensure that you cover all the essential aspects.
Mention the important points in bullets or highlight them.
Just give 2 lines answer if the question is not relevant to the content.
Context:
{context}
Question:
{input}
"""
)

# Global vector store will hold our document embeddings
vector_store = None

def vector_embedding():
    """
    Create the FAISS vector store from PDF documents.
    Each document chunk has metadata (like PDF name and page/slide number) added.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("api/pdf")  # Data Ingestion
    docs = loader.load()  # Document Loading

    # Add metadata (e.g., PDF name, page number)
    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", "Unknown PDF")
        doc.metadata["page"] = doc.metadata.get("page", "Unknown Page")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

@app.on_event("startup")
async def startup_event():
    """
    Create the vector_store on startup. This ensures that
    the FAISS index is built once and is available for all API calls.
    """
    global vector_store
    vector_store = vector_embedding()
    print("Vector Store DB is ready.")

class QueryRequest(BaseModel):
    query: str
    option: str  # Accepts "Search Documents" or "Search the Internet"

def format_similarity_results(results):
    """
    Format similarity search results with metadata, returning a list of dictionaries.
    """
    formatted_results = []
    for result in results:
        source = result.metadata.get("source", "Unknown PDF")
        page = result.metadata.get("page", "Unknown Page")
        content = result.page_content.strip()
        formatted_results.append({
            "pdf_name": source,
            "page": page,
            "relevant_content": content
        })
    return formatted_results

def perform_web_search(query: str):
    """
    Use the Google Custom Search JSON API to perform web searches.
    """
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_search_api_key,
        "cx": google_cx,
        "q": query,
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        return None

@app.post("/ask")
async def ask(query_request: QueryRequest):
    """
    This endpoint accepts a JSON payload with 'query' and 'option'.
    Based on the option, it either returns document-based answers (with metadata)
    or performs a web search.
    """
    query_text = query_request.query
    option = query_request.option

    if option == "Search Documents":
        # Create retrieval chain for document-based search
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': query_text})
        elapsed_time = time.process_time() - start_time
        
        answer = response.get("answer", "No answer found.")
        similarity_results = []
        if "context" in response and response["context"]:
            similarity_results = format_similarity_results(response["context"])
        
        return {
            "mode": "documents",
            "response_time": elapsed_time,
            "answer": answer,
            "similarity_results": similarity_results
        }

    elif option == "Search the Internet":
        # Perform a web search using the helper function
        results = perform_web_search(query_text)
        if results:
            search_results = []
            for item in results:
                search_results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "link": item.get("link")
                })
            return {
                "mode": "internet",
                "answer": "Here are some resources I found on the internet:",
                "search_results": search_results
            }
        else:
            return {
                "mode": "internet",
                "answer": "Sorry, I couldn't find any relevant resources on the internet."
            }
    else:
        raise HTTPException(status_code=400, detail="Invalid option provided. Use 'Search Documents' or 'Search the Internet'.")