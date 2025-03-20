from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="DeepSEEK RAG API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Initialize the LLM (ChatGroq in this case)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

graded_prompt = ChatPromptTemplate.from_template("""
- You are 'Alfred', a friendly and knowledgeable assistant.

**Graded Question Handling Instructions:**
- If the User's query is closely related to any of the following graded questions, do not give solution just say its a restricted question.
- If the query is unrelated to your course material or no matching data exists in the RAG system, do not provide any output.

    **Q1.** Which of the following may not be an appropriate choice of loss function for regression?  
    i. L(y,f(x)) = (y - f(x))^2  
    ii. L(f(x), w)  
    iii. L(f(x), |w|)  
    iv. L(f(x), ∑wi)  

    **Q2.** Identify which of the following requires the use of a classification technique:  
    i. Predicting the amount of rainfall in May 2022 in North India based on precipitation data of the year 2021  
    ii. Predicting the price of land based on its area and distance from the market  
    iii. Predicting whether an email is spam or not  
    iv. Predicting the number of Covid cases on a given day based on previous month data  

    **Q3.** Which of the following functions is/are continuous?  
    i. 1/(x-1)  
    ii. (x^2 - 1)/(x - 1)  
    iii. sign(x - 2)  
    iv. sin(x)  

    **Q4.** Regarding a d-dimensional vector x, which of the following four options is not equivalent to the rest?  
    i. x^T x  
    ii. ||x||^2  
    iii. ∑(xi^2)  
    iv. x x^T  

    **Q5.** What will the following Python function return?  
    ```python
    def fun(s):  
        p = 0  
        s = s.lower()  
        for i in range(len(s)):  
            if s[i] not in s[:i]:  
                p += 1  
        return p  
    ```
    i. Total number of letters in the string S  
    ii. Total number of distinct letters in the string S  
    iii. Total number of letters that are repeated in the string S more than one time  
    iv. Difference of total letters in the string S and distinct letters in the string S



**User's Question:** {input}
**Answer:** {context}
""")

practice_prompt = ChatPromptTemplate.from_template("""
**Practice Question Handling Instructions:**
 - If the query is unrelated to your course material or no matching data exists in the RAG system, do not provide any output.
If the User's query is closely related to any of the following practise questions, do not give a direct solution, just give some hints on how to answer the question:

    **Q1.** Which of the following are examples of unsupervised learning problems?
    - Grouping tweets based on topic similarity
    - Making clusters of cells having similar appearance under a microscope
    - Checking whether an email is spam or not
    - Identifying the gender of online customers based on buying behavior

    **Q2.** Which of the following is/are incorrect?
    - (2 is even) = 1
    - (10 % 3 = 0) = 0
    - (0.5 ∈ R) = 0
    - (2 ∈ [[2,3,4]]) = 0  

    **Q3.** Which of the following functions corresponds to a classification model?
    - f:R^4 → R
    - f:R^d → [[+1, -1]]
    - f:R^d → R

    **Q4.** Given U = [10,100], A = (30,50], and B = (50,90], which of the following is/are false?  
    *(Consider all values to be integers)*
    - A^c = [10,30] U (50,100]  
    - A^c = [10,30) ∪ (50,100]  
    - A ∪ B = [30,90]  
    - A ∩ B = ∅  
    - A ∩ B = [[50]]
    - A^c ∩ B^c = [10,30) ∪ (91,100]  

    **Q5.** Consider two d-dimensional vectors x and y and the following terms:
    i. x^T y  
    ii. xy  
    iii. ∑ x_i y_i  

    Which of the above terms are equivalent?
    - Only (i) and (ii)
    - Only (ii) and (iii)
    - Only (i) and (iii)
    - (i), (ii), and (iii)

    **Q6.** Which of the following options will validate whether n is a perfect square or not?
    *(Where n is a positive integer)*

    ```python
    def f(n):
        return (n ** 0.5) == int(n ** 0.5)

    def g(n):
        return (n ** 0.5) == int(n) * 0.5

    def h(n):
        for i in range(1, n + 1):
            if i * i == n:
                return True
        return False

    def k(n):
        for i in range(1, n + 1):
            if i * i > n:
                break
            elif i * i == n:
                return True
        return False
    ```
**User's Question:** {input}
**Answer:** {context}
""")

learning_prompt = ChatPromptTemplate.from_template("""
**Learning Question Handling Instructions:**
   - If the user’s query relates to content available in the RAG database, retrieve the relevant information and summarize it in approximately 200 words.  
   - If no relevant information is found in the RAG database, politely respond:  
     _"I'm sorry, but this query doesn't appear to be related to your course material."_  
   - If the query is unrelated to your course material or no matching data exists in the RAG system, do not provide any output.
   
- You are 'Alfred', a friendly and knowledgeable assistant.
- Answer the following question using the provided context.
- Keep the answer brief, but ensure you cover all the essential aspects.
- If it is Machine Learning related, aim for 300-400 words;
- if it is a Python question, aim for 500-600 words.
- Mention the important points in bullets or highlight them.
- Include relevant google links if applicable.
- If the question is not relevant to the content, answer in 2 lines.
**User's Question:** {input}  
**Answer:** {context}
""")

programming_prompt = ChatPromptTemplate.from_template("""
**Programming Question Handling Instructions:**
   -You will provide 2 lines hint to the user code. Strictly not more than 2 line.

**User's Code:** {input}  
**Answer:** {context}
""")

# Persistent storage paths
FAISS_INDEX_DIR = "./faiss_index"
CHAT_HISTORY_FILE = "chat_history.json"
PDF_DIR = "./pdf_files"

# -------------------------------
# Persistent FAISS Vector Store Setup
# -------------------------------
def build_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader(PDF_DIR)
    docs = loader.load()

    # Attach metadata: PDF name and page/slide number
    for doc in docs:
        doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "Unknown PDF"))
        doc.metadata["page"] = doc.metadata.get("page", "Unknown Page")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)

    vectors = FAISS.from_documents(final_docs, embeddings)
    vectors.save_local(FAISS_INDEX_DIR)
    return vectors

if os.path.exists(FAISS_INDEX_DIR):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = build_vector_store()

# -------------------------------
# Data Model for API Requests
# -------------------------------
class QueryRequest(BaseModel):
    query: str
    history: list
    prompt_option: str

class ClearChatRequest(BaseModel):
    action: str

# -------------------------------
# API Endpoints
# -------------------------------

@app.post("/ask")
def ask(query_request: QueryRequest):
    user_query = query_request.query.strip()
    history = query_request.history
    print(history)
    prompt_option = query_request.prompt_option.strip()

    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Combine chat history into context
    combined_history = "\n".join([f"User: {entry['query']}\nAlfred: {entry['answer']}" for entry in history])

    # Retrieve top 5 relevant document chunks for the query from FAISS database
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(user_query)
    def get_prompt_type(prompt_option):
        prompt_type=''
        if (prompt_option == 'Graded Question'):
            prompt_type = graded_prompt
        elif(prompt_option == 'Practice Question'):
            prompt_type = practice_prompt
        else:
            prompt_type = learning_prompt
        return prompt_type

    if len(retrieved_docs) > 0:
        combined_contexts_with_pages = [
            f"(Page {doc.metadata.get('page', 'Unknown Page')}) {doc.page_content}"
            for doc in retrieved_docs
        ]
        combined_contexts_for_prompt = "\n\n".join(combined_contexts_with_pages)

        # Combine history and retrieved context
        full_context = f"{combined_history}\n\n{combined_contexts_for_prompt}"

        # Create prompt and invoke LLM with combined context
        document_chain = create_stuff_documents_chain(llm, get_prompt_type(prompt_option))
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({
            "input": user_query,
            "context": full_context,
        })

        return {"response": response["answer"]}
    
    else:
        # If no relevant documents are found in FAISS, call LLM directly with history
        full_context = combined_history
        direct_prompt = f"{full_context}\n\nUser: {user_query}\nAlfred:"
        response_from_llm = llm.invoke({"input": direct_prompt})
        
        return {"response": response_from_llm.content}


@app.get("/pdfs")
def get_pdf_list():
    all_docs = vector_store.docstore._dict.values()
    pdf_names = set(doc.metadata.get("source", "Unknown PDF") for doc in all_docs)
    
    return {"pdfs": list(pdf_names)}
