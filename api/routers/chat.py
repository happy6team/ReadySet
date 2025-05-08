from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from agent_state import AgentState
from ..schemas.chat_dto import QueryRequest, QueryResponse, Message, ReportSource, map_to_message
import copy

router = APIRouter()

@router.post("/chat", response_model=QueryResponse)
async def execute_query(request: QueryRequest, fastapi_request: Request): 
    try:
        graph = fastapi_request.app.state.supervisor_graph
        # 기본 AgentState의 복사본 생성 (깊은 복사)
        state = copy.deepcopy(fastapi_request.app.state.base_agent_state)
        # 요청 데이터로 상태 업데이트
        state["input_query"] = request.input_query
        state = graph.invoke(state)

        # 쿼리 히스토리 업데이트
        fastapi_request.app.state.add_thread_query(fastapi_request.app, state["thread_id"], request.input_query)
        fastapi_request.app.state.add_thread_messages(fastapi_request.app, state["thread_id"], state["messages"])
        
        messages = []
        for msg in state["messages"]:
            message = map_to_message(msg)
            messages.append(message)
        
        return QueryResponse(messages=messages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
