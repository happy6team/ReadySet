from fastapi import APIRouter, Depends, HTTPException,Form, Request, Response
from ..schemas.chat_dto import QueryRequest, QueryResponse, Message, ReportSource, map_to_message, ChatHistoryListResponse, ChatHistory
import copy
from fastapi.responses import FileResponse
from urllib.parse import unquote
import pathlib
import os

router = APIRouter()

@router.post("/chat", response_model=QueryResponse)
async def execute_query( fastapi_request: Request, input_query: str = Form(...)): 
    try:
        graph = fastapi_request.app.state.supervisor_graph
        # 기본 AgentState의 복사본 생성 (깊은 복사)
        state = copy.deepcopy(fastapi_request.app.state.base_agent_state)

        # 요청 데이터로 상태 업데이트
        state["input_query"] = input_query
        thread_id=state["thread_id"]
        state = graph.invoke(state)  # ✅ 여기에서 thread_id가 default로 바뀜!!!

        # 히스토리에 대화 내용 추가
        fastapi_request.app.state.add_thread_query(fastapi_request.app, thread_id, input_query)
        fastapi_request.app.state.add_thread_messages(fastapi_request.app, thread_id, state["messages"])
        
        messages = []
        for msg in state["messages"]:
            message = map_to_message(msg)
            messages.append(message)
        
        return QueryResponse(messages=messages)
        
    except Exception as e:
        print("Error in execute_query:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/reports/download")
async def download_report_file(fastapi_request: Request, source: str): 
    try:
        # URL 디코딩 (URL에서 한글이나 특수문자가 인코딩되었을 경우)
        decoded_source = unquote(source)
        
        # 상대 경로를 절대 경로로 변환
        # 클라이언트가 "./vector_store/docs/..." 형식으로 보냈을 때 처리
        if decoded_source.startswith("./"):
            decoded_source = decoded_source[2:]  # './' 제거
        # 프로젝트 루트 디렉토리 (필요에 따라 수정)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # 실제 파일 경로 생성
        file_path = os.path.join(root_dir, decoded_source)
        # 경로 정규화 및 보안 검사 (경로 주입 공격 방지)
        file_path = os.path.normpath(file_path)
        root_path = os.path.normpath(root_dir)
        # 파일이 root_dir 외부에 있는지 확인 (경로 주입 공격 방지)
        if not file_path.startswith(root_path):
            raise HTTPException(status_code=403, detail="접근이 허용되지 않은 파일입니다.")
        
        # 파일 존재 여부 확인
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        # 파일 이름 추출
        file_name = os.path.basename(file_path)
        
        # 파일 다운로드 응답 반환
        return FileResponse(
            path=file_path, 
            filename=file_name,
            media_type='application/pdf'  # PDF 파일의 MIME 타입
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"파일 다운로드 오류: {str(e)}")


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

