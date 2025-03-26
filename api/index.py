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
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

graded_prompt = ChatPromptTemplate.from_template("""
- You are 'Alfred', a friendly and knowledgeable assistant.
- Mention the important points in bullets or highlight them.

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
 - You are 'Alfred', a friendly and knowledgeable assistant.
 - Answer the following question using the provided context.
 - Mention the important points in bullets or highlight them.
 - Include relevant google links if applicable(Please provide clickable links and highlight it).
 - If the query is not related to our course material or no matching data exists in the RAG database, do not provide any output.
 - If the User's query is closely related to any of the following practise questions, do not give a direct solution, just give some hints on how to answer the question:
 
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
You are 'Alfred', a friendly and knowledgeable assistant. Please provide a conversational response to the user's query, keeping the following guidelines in mind:

1. Ensure the response is relevant to the course material.
2. If the query is not related to the course, politely ask the user to ask a course-related question.
3. Use a natural, conversational tone in your responses.
4. Provide context from previous interactions when appropriate.
5. If you're unsure about something, it's okay to say so.

User's question: {input}
Previous context: {context}
Current conversation topic: {current_topic}

Please provide a helpful and engaging response:
""")

# Define prompt templates for debugging code
debug_prompt = ChatPromptTemplate.from_template("""
You are 'Alfred', an expert Python programmer and debugging assistant.
Analyze the provided Python code and identify any errors or issues.
Respond with a concise explanation in **two lines only**.

**Code:** {code}

**Question:** {question}
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
    history: list[dict[str, str]]
    prompt_option: str

class DebugCodeRequest(BaseModel):
    question: str
    code: str

class ClearChatRequest(BaseModel):
    action: str

class QuestionsRequest(BaseModel):
    questions: list[str]

# -------------------------------
# API Endpoints
# -------------------------------

# Helper Functions for Follow-up/Conservational AI feature

def summarize_conversation(history: list[dict[str, str]], max_length: int = 5) -> str:
    if len(history) <= max_length:
        return "\n".join([f"User: {entry['query']}\nAlfred: {entry['answer']}" for entry in history])
    else:
        summary = f"Summary of previous {len(history) - max_length} messages:\n"
        summary_prompt = f"Summarize this conversation:\n{json.dumps(history[:-max_length])}"
        summary_response = llm.invoke(summary_prompt)
        summary += summary_response.content if isinstance(summary_response, AIMessage) else str(summary_response)
        summary += "\n\nRecent messages:\n"
        summary += "\n".join([f"User: {entry['query']}\nAlfred: {entry['answer']}" for entry in history[-max_length:]])
        return summary


def get_topic(question: str) -> str:
    # Simple topic extraction (can be improved with NLP techniques)
    keywords = question.lower().split()
    topics = ['machine learning', 'programming', 'mathematics', 'data science']
    for topic in topics:
        if any(keyword in topic for keyword in keywords):
            return topic
    return "general"

def is_follow_up_question(question: str) -> bool:
    follow_up_phrases = ['you mentioned', 'earlier you said', 'previously', 'in relation to']
    return any(phrase in question.lower() for phrase in follow_up_phrases)



@app.post("/top-questions")
def get_top_questions(request: QuestionsRequest):
 
    questions = request.questions
    
    if not questions:
        raise HTTPException(status_code=400, detail="The 'questions' list cannot be empty.")

    prompt_input = (
    "Analyze these programming questions and return the top 5 most frequent topics. "
    "Respond with ONLY a Python list literal containing exactly 5 unquoted items separated by commas, "
    "formatted exactly like this example: "
    "[Python functions, Lists, Error handling, OOP, File handling] "
    "STRICT REQUIREMENTS: "
    "1. No quotation marks of any kind "
    "2. No backslashes or escape characters "
    "3. No counts or numbering "
    "4. No additional text or commentary "
    "5. Exactly 5 comma-separated items between square brackets "
    "6. Each topic must be 2-4 words in lowercase/uppercase "
    "If you include any quotes, backslashes, or other formatting, the response is wrong.\n\n"
    "QUESTIONS:\n" +
    "\n".join(f"- {question}" for question in questions)
)
    response_from_llm = llm.invoke(prompt_input)
    return {"response": response_from_llm.content}


@app.post("/debug/code")
def debug_code(request: DebugCodeRequest):
    user_question = request.question.strip()
    user_code = request.code.strip()

    if not user_question or not user_code:
        raise HTTPException(status_code=400, detail="Both 'question' and 'code' fields are required.")

    # Create prompt for debugging code and invoke LLM
    prompt_input = debug_prompt.format(question=user_question, code=user_code)
    
    # Pass the prompt directly as a string to llm.invoke()
    response_from_llm = llm.invoke(prompt_input)

    return {"response": response_from_llm.content}

@app.post("/ask")
def ask(query_request: QueryRequest):
    try:
        user_query = query_request.query.strip()
        history = query_request.history[-10:]  # Keep last 10 interactions
        prompt_option = query_request.prompt_option.strip()

        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # Combine chat history into context
        combined_history = summarize_conversation(history)

        # Handle "What was my last question?" and similar queries
        if "last question" in user_query.lower():
            if len(history) > 0:
                last_question = history[-1]["query"]
                last_topic = get_topic(last_question)
                return {
                    "response": f"Your last question was about {last_topic}. You asked: '{last_question}'",
                    "updated_history": history
                }
            else:
                return {
                    "response": "I don't have any record of your previous questions yet. Feel free to ask me anything about the course!",
                    "updated_history": history
                }

        # Retrieve relevant document chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(user_query)

        current_topic = get_topic(user_query)

        if len(retrieved_docs) > 0:
            combined_contexts = "\n\n".join([doc.page_content for doc in retrieved_docs])
            full_context = f"{combined_history}\n\n{combined_contexts}"

            # Create prompt and invoke LLM with combined context
            document_chain = create_stuff_documents_chain(llm, learning_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({
                "input": user_query,
                "context": full_context,
                "current_topic": current_topic
            })

            # Handle follow-up questions
            if is_follow_up_question(user_query):
                response["answer"] = f"Regarding your previous question, {response['answer']}"

            # Append the current query and response to the history
            history.append({"query": user_query, "answer": response["answer"]})

            return {"response": response["answer"], "updated_history": history}
        
        else:
            # If no relevant documents are found, use a more general response
            direct_prompt = f"{combined_history}\n\nUser: {user_query}\nAlfred:"
            response_from_llm = llm.invoke(direct_prompt)

            # Append the current query and response to the history
            history.append({"query": user_query, "answer": response_from_llm.content})

            return {"response": response_from_llm.content, "updated_history": history}
    except Exception as e:
        # Log the error for debugging
        print(f"Error in /ask endpoint: {str(e)}")
        # Return a user-friendly error message
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

@app.get("/pdfs")
def get_pdf_list():
    all_docs = vector_store.docstore._dict.values()
    pdf_names = set(doc.metadata.get("source", "Unknown PDF") for doc in all_docs)
    
    return {"pdfs": list(pdf_names)}
