import os
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from src.agents.research_agent import get_agent_executor
from src.evaluation.evaluator import run_eval_experiment
from src.utils.logger import get_logger

# Initialize API and Logger
logger = get_logger(__name__)
app = FastAPI(
    title="Intelligent Research Assistant API",
    description="A multi-tool Agentic RAG backend with integrated evaluation and memory.",
    version="1.0.0"
)

# Initialize the Agent once for the application lifecycle
agent_executor = get_agent_executor()


# --- SCHEMAS ---

class QueryRequest(BaseModel):
    input: str
    session_id: str = "default_session"


class AgentResponse(BaseModel):
    answer: str
    faithfulness: float
    relevancy: float
    session_id: str


# --- ENDPOINTS ---

@app.post("/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to ingest new PDF documents into the local knowledge vault.
    """
    upload_dir = "data/raw"
    os.makedirs(upload_dir, exist_ok=True)

    try:
        # 1. Save file to disk
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"API: Received file '{file.filename}'. Starting ingestion...")

        # 2. Trigger the Day 7 Ingestion Pipeline
        # Note: We import here to avoid circular dependencies
        from main import process_new_uploads
        process_new_uploads([file_path])

        return {"message": f"Successfully indexed {file.filename}", "file_path": file_path}

    except Exception as e:
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat", response_model=AgentResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Primary chat endpoint for interacting with the Agent.
    """
    try:
        # 1. Prepare Session Config
        config = {"configurable": {"session_id": request.session_id}}

        logger.info(f"API: Processing query for session '{request.session_id}'")

        # 2. Invoke Agent (Supports PDF Search, Web Search, and Memory)
        # This is a blocking call, but FastAPI runs it in a threadpool
        response_dict = agent_executor.invoke({"input": request.input}, config=config)
        answer = response_dict["output"]

        # 3. Context Extraction for RAGAS
        retrieved_contexts = []
        if "intermediate_steps" in response_dict:
            for action, tool_output in response_dict["intermediate_steps"]:
                if action.tool == "PDF_Research_Search":
                    retrieved_contexts.extend(tool_output.split("\n\n"))

        # 4. Evaluate the response asynchronously
        eval_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}
        if retrieved_contexts:
            logger.info("API: Triggering RAGAS evaluation...")
            eval_scores = await run_eval_experiment(
                query=request.input,
                response=answer,
                contexts=retrieved_contexts
            )

        return AgentResponse(
            answer=answer,
            faithfulness=eval_scores.get("faithfulness", 0.0),
            relevancy=eval_scores.get("answer_relevancy", 0.0),
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"Chat API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Agent Error")


@app.get("/health")
async def health_check():
    """Returns the status of the API service."""
    return {"status": "online", "model": "gpt-4o-mini", "agent_ready": True}


if __name__ == "__main__":
    import uvicorn

    # Use uvicorn to serve the API
    uvicorn.run(app, host="0.0.0.0", port=8000)