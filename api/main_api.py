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
    description="A multi-tool Agentic RAG backend with multi-user isolation and source transparency.",
    version="1.3.0"
)

# Initialize the Agent
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
    upload_dir = os.path.join("data/raw", session_id)
    os.makedirs(upload_dir, exist_ok=True)
    try:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from main import process_new_uploads
        process_new_uploads([file_path], session_id=session_id)
        return {"message": f"Successfully indexed {file.filename}"}
    except Exception as e:
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    Streaming endpoint that now yields raw source chunks for UI inspection.
    """

    async def event_generator():
        input_data = {"input": request.input, "session_id": request.session_id}
        config = {"configurable": {"session_id": request.session_id}}
        retrieved_contexts = []
        final_answer = ""

        try:
            async for event in agent_executor.astream_events(
                    input_data, config=config, version="v1"
            ):
                kind = event["event"]

                if kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'thought', 'content': f'Searching {event['name']}...'})}\n\n"

                elif kind == "on_tool_end":
                    if event["name"] == "PDF_Research_Search":
                        # We capture the output to return it as a 'source_chunks' packet later
                        output = event["data"].get("output", "")
                        if output:
                            retrieved_contexts.append(output)

                elif kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        final_answer += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            # 1. Yield Source Chunks (The new Day 28 feature)
            # This is sent after the answer but before evaluation
            if retrieved_contexts:
                yield f"data: {json.dumps({'type': 'source_chunks', 'content': retrieved_contexts})}\n\n"

            # 2. Final Evaluation Step
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