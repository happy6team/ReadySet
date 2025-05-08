from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from agent_state import AgentState
from graph import create_supervisor_graph
from dotenv import load_dotenv
from ..schemas.chat_dto import QueryRequest, QueryResponse, Message, ReportSource

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
            # 메시지 유형 확인
            msg_type = msg.__class__.__name__
            content_str = str(msg)
            sources = None
            
            # 딕셔너리 형태의 메시지인 경우 sources 추출 시도
            if msg_type == "dict" or (isinstance(content_str, str) and content_str.startswith("{")):
                try:
                    # 문자열이면 JSON으로 파싱
                    if isinstance(msg, str):
                        content_dict = json.loads(content_str)
                    else:
                        # 이미 딕셔너리인 경우
                        content_dict = msg
                    
                    # answer와 sources 추출
                    if isinstance(content_dict, dict):
                        # 응답 내용은 answer 필드에서 가져옴
                        if "answer" in content_dict:
                            content_str = content_dict["answer"]
                        
                        # sources 정보 추출
                        if "sources" in content_dict and isinstance(content_dict["sources"], list):
                            sources = []
                            for src in content_dict["sources"]:
                                if isinstance(src, dict):
                                    sources.append(ReportSource(
                                        content=src.get("content"),
                                        section=src.get("section"),
                                        source=src.get("source"),
                                        filename=src.get("filename"),
                                        rank=src.get("rank")
                                    ))
                except Exception as e:
                    # 파싱 실패 시 원본 내용 사용
                    print(f"메시지 파싱 오류: {e}")
            
            # 메시지 객체 생성
            message = Message(
                content=content_str,
                type=msg_type,
                sources=sources
            )
            
            messages.append(message)
        
        return QueryResponse(messages=messages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
