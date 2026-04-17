import os
import shutil
import json
import asyncio

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from src.agents.research_agent import get_agent_executor
from src.evaluation.evaluator import run_eval_experiment
from src.utils.logger import get_logger
from fastapi.responses import StreamingResponse

# Initialize API and Logger
logger = get_logger(__name__)
app = FastAPI(
    title="Intelligent Research Assistant API",
    description="A multi-tool Agentic RAG backend with multi-user isolation.",
    version="1.2.0"
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
async def upload_document(
        session_id: str = Query(..., description="Unique session ID for multi-user isolation"),
        file: UploadFile = File(...)
):
    """
    Endpoint to ingest new PDF documents into the local knowledge vault,
    tagged with a session_id for isolation.
    """
    # Create session-specific upload directory to prevent file overwrite collisions
    upload_dir = os.path.join("data/raw", session_id)
    os.makedirs(upload_dir, exist_ok=True)

    try:
        # 1. Save file to disk
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"API: Received file '{file.filename}' for session '{session_id}'.")

        # 2. Trigger the Ingestion Pipeline with the session_id
        # We pass the session_id so chroma_manager can tag the chunks
        from main import process_new_uploads
        process_new_uploads([file_path], session_id=session_id)

        return {"message": f"Successfully indexed {file.filename} for session {session_id}"}

    except Exception as e:
        logger.error(f"Upload Error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat", response_model=AgentResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Primary chat endpoint for interacting with the Agent.
    """
    try:
        # 1. Prepare Session Config (Memory relies on this)
        config = {"configurable": {"session_id": request.session_id}}

        logger.info(f"API: Processing query for session '{request.session_id}'")

        # 2. Invoke Agent (Tool inputs will include session_id for filtering)
        # Note: Your agent tools need to access this session_id to filter ChromaDB
        response_dict = agent_executor.invoke(
            {"input": request.input, "session_id": request.session_id},
            config=config
        )
        answer = response_dict["output"]

        # 3. Context Extraction for RAGAS
        retrieved_contexts = []
        if "intermediate_steps" in response_dict:
            for action, tool_output in response_dict["intermediate_steps"]:
                if action.tool == "PDF_Research_Search":
                    retrieved_contexts.extend(tool_output.split("\n\n"))

        # 4. Evaluate the response
        eval_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}
        if retrieved_contexts:
            try:
                eval_scores = await run_eval_experiment(
                    query=request.input,
                    response=answer,
                    contexts=retrieved_contexts
                )
            except Exception as e:
                logger.error(f"RAGAS evaluation failed: {e}")

        return AgentResponse(
            answer=answer,
            faithfulness=eval_scores.get("faithfulness", 0.0),
            relevancy=eval_scores.get("answer_relevancy", 0.0),
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"Chat API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Agent Error")


@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    Streaming endpoint with session isolation support.
    """

    async def event_generator():
        # Inject session_id into the input so tools can use it for filtering
        input_data = {"input": request.input, "session_id": request.session_id}
        config = {"configurable": {"session_id": request.session_id}}

        retrieved_contexts = []
        final_answer = ""

        try:
            async for event in agent_executor.astream_events(
                    input_data,
                    config=config,
                    version="v1"
            ):
                kind = event["event"]

                if kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'thought', 'content': f'Searching {event['name']}...'})}\n\n"

                elif kind == "on_tool_end":
                    if event["name"] == "PDF_Research_Search":
                        retrieved_contexts.append(event["data"].get("output", ""))

                elif kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        final_answer += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            # Final Evaluation Step
            if retrieved_contexts and final_answer:
                try:
                    eval_scores = await asyncio.wait_for(
                        run_eval_experiment(request.input, final_answer, retrieved_contexts),
                        timeout=30.0
                    )
                    yield f"data: {json.dumps({
                        'type': 'eval',
                        'faithfulness': eval_scores.get('faithfulness', 0.0),
                        'relevancy': eval_scores.get('answer_relevancy', 0.0)
                    })}\n\n"
                except Exception as eval_error:
                    logger.error(f"RAGAS failed: {eval_error}")
                    yield f"data: {json.dumps({'type': 'eval', 'faithfulness': 0.0, 'relevancy': 0.0})}\n\n"

        except Exception as main_error:
            logger.error(f"Streaming Error: {main_error}")
            yield f"data: {json.dumps({'type': 'token', 'content': ' [Error: Connection Interrupted]'})}\n\n"
        finally:
            logger.info(f"API: Closing Stream for {request.session_id}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    return {"status": "online", "model": "gpt-4o-mini", "agent_ready": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)