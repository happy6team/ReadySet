from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from agent_state import AgentState
from graph import create_supervisor_graph
from dotenv import load_dotenv
from ..schemas.chat_dto import QueryRequest, QueryResponse, Message

router = APIRouter()

@router.post("/chat", response_model=QueryResponse)
async def execute_query(request: QueryRequest): 
    try:
        # Load environment variables
        load_dotenv()
        
        # Create graph
        graph = create_supervisor_graph()
        
        # Create initial state
        state = AgentState(
            input_query=request.input_query,
            thread_id=request.thread_id,
            project_name=request.project_name,
            project_explain=request.project_explain,
            messages=[]
        )
        
        # Execute graph
        state = graph.invoke(state)
        
        # Convert messages to response format
        messages = []
        for msg in state["messages"]:
            messages.append(Message(
                content=str(msg),
                type=msg.__class__.__name__
            ))
        
        return QueryResponse(messages=messages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
