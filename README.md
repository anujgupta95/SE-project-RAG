# DeepSEEK RAG Portal

A Retrieval-Augmented Generation (RAG) system built with FastAPI and LangChain that provides intelligent responses based on course content from PDF documents.

## Features

- **RAG System**: Context-aware responses using FAISS vector store
- **Multiple Prompt Types**:
  - Graded Questions (hints only)
  - Practice Questions (guided hints)
  - Learning Mode (detailed explanations)
- **Code Debugging**: Python code analysis endpoint
- **PDF Management**: Persistent vector storage of course materials
- **Conversation History**: Context-aware multi-turn conversations

## Prerequisites

- Python 3.7+
- [Groq API Key](https://console.groq.com/)
- [Google API Key](https://cloud.google.com/)
- PDF course materials in `./pdf_files` directory

## Installation

1. **Clone Repository**:
git clone https://github.com/anujgupta95/SE-project-RAG.git

2. **Set Up Virtual Environment**:
python -m venv venv source venv/bin/activate # Linux/MacOS
venv\Scripts\activate # Windows

3. **Install Dependencies**:
pip install -r requirements.txt

4. **Environment Variables**:
Create `.env` file:
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

## Usage

1. **Start Server**:
uvicorn app:app –reload –host 0.0.0.0 –port 8000

2. **API Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Main query endpoint (supports 'graded', 'practice', 'learning' modes) |
| `/debug/code` | POST | Python code debugging assistance |
| `/top-questions` | POST | Analyze question patterns |
| `/pdfs` | GET | List indexed PDF documents |

## API Reference

### POST /ask
**Request**:
{ “query”: “What is merge sort?”, “history”: [], “prompt_option”: “learning” }

**Response**:
{ “response”: “Formatted answer with resources…”, “updated_history”: [] }

### POST /debug/code
**Request**:
{ “question”: “What’s wrong with this code?”, “code”: “def example(): pass” }

### GET /pdfs
**Response**:
{ “pdfs”: “machine-learning.pdf”, “algorithms.pdf” }

## Configuration

### Environment Variables
- `GROQ_API_KEY`: Groq Cloud API key
- `GOOGLE_API_KEY`: Google Generative AI credentials

### PDF Storage
- Place all PDFs in `./pdf_files` directory
- FAISS vector store auto-builds on first run

## Deployment

1. **Production Server**:
uvicorn app:app –host 0.0.0.0 –port 8000

2. **Docker** (optional):
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD “uvicorn”, “app:app”, “–host”, “0.0.0.0”, “–port”, “8000”

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Note**: For production deployment, ensure proper CORS configuration and input sanitization.

