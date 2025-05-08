from fastapi import APIRouter, Depends, HTTPException,Form, Request, Response
from ..schemas.report_dto import ReportListResponse, ReportSource, process_history_for_documents
import copy
from fastapi.responses import FileResponse
from urllib.parse import unquote
import pathlib
import os

router = APIRouter()
ROOT_DIR = os.environ.get('PROJECT_ROOT_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 그동안 받았던 문서 목록 확인 api
@router.get("/reports", response_model=ReportListResponse)
async def get_report_list( fastapi_request: Request): 
    try:
        # 기본 AgentState의 thread_id 가져오기
        thread_id = fastapi_request.app.state.base_agent_state["thread_id"]

        # 히스토리 조회
        messages_history = fastapi_request.app.state.get_thread_messages(fastapi_request.app, thread_id)
        queries_history = fastapi_request.app.state.get_thread_queries(fastapi_request.app, thread_id)
        
        # 히스토리 데이터에서 문서 데이터 가져오기
        reports = process_history_for_documents(queries_history, messages_history)
        
        return ReportListResponse(sources=reports)
        
    except Exception as e:
        print("Error in execute_query:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# 문서 목록 중에서 선택한 문서 다운로드 api
@router.post("/reports/download")
async def download_report_file(fastapi_request: Request, source: str): 
    try:
        # URL 디코딩
        decoded_source = unquote(source)
        
        # 상대 경로 처리
        if decoded_source.startswith("./"):
            decoded_source = decoded_source[2:]
        
        # 경로 구분자 표준화 (OS에 맞게 변환)
        decoded_source = decoded_source.replace('\\', os.path.sep)
        
        # 경로 생성
        file_path = os.path.join(ROOT_DIR, decoded_source)
        file_path = os.path.normpath(file_path)
        
        # 경로 검증 및 파일 존재 확인
        if not os.path.isfile(file_path):
            # 디버깅을 위한 추가 정보
            dir_path = os.path.dirname(file_path)
            if os.path.isdir(dir_path):
                files_in_dir = os.listdir(dir_path)
            else:
                print(f"Directory does not exist: {dir_path}")
            
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")
        
        # 파일 반환
        file_name = os.path.basename(file_path)
        return FileResponse(
            path=file_path, 
            filename=file_name,
            media_type='application/pdf'
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"파일 다운로드 오류: {str(e)}")