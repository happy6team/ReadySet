from fastapi import APIRouter, Depends, HTTPException,Form, Request, Response
from ..schemas.report_dto import ReportListResponse, ReportSource, process_history_for_documents
import copy
from fastapi.responses import FileResponse
from urllib.parse import unquote
import pathlib
import os

router = APIRouter()

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