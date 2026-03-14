import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.deepagent import deepagents_agent
from agents.claudeagent import claude_agent_stream
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/agents", tags=["agents"])

class StreamRequest(BaseModel):
    messages: str

@router.post("/deepagents-stream")
async def deepagents_stream(payload: StreamRequest):

    """Run content builder agent with streaming response."""

    def generate():
        config = {"configurable": {"thread_id": "a12345"}}
        message = HumanMessage(content=payload.messages)
        for chunk in deepagents_agent.stream(
            {"messages": [message]},
            config=config,
        ):
            yield json.dumps(chunk, default=str) + "\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/claude-stream")
async def claude_stream(payload: StreamRequest):
    async def generate():
        yield ": connected\n\n"
        async for event in claude_agent_stream(payload.messages):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")