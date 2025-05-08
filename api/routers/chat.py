from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from agent_state import AgentState
from graph import create_supervisor_graph
from dotenv import load_dotenv
from ..schemas.chat_dto import QueryRequest, QueryResponse, Message, ReportSource, map_to_message

router = APIRouter()

@router.post("/chat", response_model=QueryResponse)
async def execute_query(request: QueryRequest, fastapi_request: Request): 
    try:
        graph = fastapi_request.app.state.supervisor_graph

        # Create initial state
        state = AgentState(
            input_query=request.input_query,
            thread_id=request.thread_id,
            project_name=request.project_name,
            project_explain=request.project_explain,
            messages=[]
        )
        
        state = graph.invoke(state)
        
        messages = []
        for msg in state["messages"]:
            message=map_to_message(msg)
            messages.append(message)
        
        return QueryResponse(messages=messages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
