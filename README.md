# 🤖 Intelligent Research Assistant (Agentic RAG)
**A Production-Ready Multi-Tool AI Agent for Scientific Research Analysis.**

## 🌟 Overview
This project is a high-performance, Agentic Retrieval-Augmented Generation (RAG) system designed to analyze dense research PDFs. It moves beyond "basic chat" by using a multi-tool agent that can search local vector stores, browse the web for real-time verification, and evaluate its own faithfulness.

## 🚀 Key Engineering Milestones (30-Day Build)
- **Agentic Orchestration**: Built with LangGraph/LangChain to support multi-tool reasoning (PDF Search + Web Search).
- **Asynchronous Streaming**: Implemented Server-Sent Events (SSE) via FastAPI to deliver real-time "Thought Traces" and token streaming.
- **Multi-User Isolation**: Engineered metadata-level multi-tenancy in ChromaDB, ensuring data privacy across user sessions.
- **Automated Evaluation**: Integrated **RAGAS** metrics (Faithfulness & Relevancy) directly into the response pipeline.
- **Production Infrastructure**: Fully containerized using Docker & Docker-Compose with layout-analysis support (Poppler/Tesseract).

## 🏗️ The Tech Stack
- **Backend**: FastAPI, Python 3.11, Uvicorn
- **Frontend**: Streamlit
- **LLM**: GPT-4o-mini (OpenAI)
- **Vector DB**: ChromaDB
- **Tools**: Tavily (Web Search), Unstructured (PDF Partitioning)

## 🔧 Installation & Setup
```bash
# Clone the repo
git clone [https://github.com/darshil1995/IntelligentResearchAssistant.git](https://github.com/darshil1995/IntelligentResearchAssistant.git)

# Run with Docker (Recommended)
docker-compose up --build
```

## 🕵️ Transparency Features
- Source Inspector: Real-time access to raw retrieved chunks for manual auditing.
- Metrics Dashboard: Instant Faithfulness and Relevancy scores for every answer.
