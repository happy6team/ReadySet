from fastapi import APIRouter, Depends, HTTPException, Request
from ..schemas.chat_dto import QueryRequest, QueryResponse, Message, ReportSource, map_to_message, ChatHistoryListResponse, ChatHistory
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
        thread_id=state["thread_id"]
        state = graph.invoke(state)  # ✅ 여기에서 thread_id가 default로 바뀜!!!

        # 히스토리에 대화 내용 추가
        fastapi_request.app.state.add_thread_query(fastapi_request.app, thread_id, request.input_query)
        fastapi_request.app.state.add_thread_messages(fastapi_request.app, thread_id, state["messages"])
        
        messages = []
        for msg in state["messages"]:
            message = map_to_message(msg)
            messages.append(message)
        
        return QueryResponse(messages=messages)
        
    except Exception as e:
        print("Error in execute_query:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/histories", response_model=ChatHistoryListResponse)
async def get_chat_histories(fastapi_request: Request): 
    try:
        # 기본 AgentState의 thread_id 가져오기
        thread_id = fastapi_request.app.state.base_agent_state["thread_id"]

        # 히스토리 조회
        messages_history = fastapi_request.app.state.get_thread_messages(fastapi_request.app, thread_id)
        queries_history = fastapi_request.app.state.get_thread_queries(fastapi_request.app, thread_id)
        
        # 히스토리 데이터 정제
        histories = []
        for query, messages in zip(queries_history, messages_history):
            processed_messages = []
            for msg in messages:
                message = map_to_message(msg)
                processed_messages.append(message)
            
            history = ChatHistory(
                query=query,
                messages=processed_messages
            )
            histories.append(history)
        
        return ChatHistoryListResponse(histories=histories)
        
    except Exception as e:
        print("Error in get_chat_histories:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

