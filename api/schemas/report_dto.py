from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class ReportSource(BaseModel):
    query: Optional[str] = None # 문서를 받았던 당시 사용자 질의
    section: Optional[str] = None
    source: Optional[str] = None
    filename: Optional[str] = None

class ReportListResponse(BaseModel):
    sources: List[ReportSource]

# 히스토리 처리 함수
def process_history_for_documents(queries_history, messages_history):
    all_reports = []
    seen_sources = set()  # 중복 제거를 위한 세트
    
    for query, messages in zip(queries_history, messages_history):
        if not query or not messages:
            continue
            
        for msg in messages:
            sources = extract_sources_from_message(msg, query)
            
            for source in sources:
                # 중복 제거 (source와 filename으로 식별)
                source_key = f"{source.source}_{source.filename}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    all_reports.append(source)
    
    return all_reports

# 메시지에서 문서 정보 추출하는 함수
def extract_sources_from_message(message: Dict[Any, Any], query: Optional[str] = None) -> List[ReportSource]:
    sources = []
    
    # 메시지 내 'sources' 키가 있는 경우
    if 'sources' in message and isinstance(message['sources'], list):
        for source in message['sources']:
            if 'source' in source and 'filename' in source:  # 필수 필드 확인
                report_source = ReportSource(
                    query=query,
                    section=source.get('section'),
                    source=source.get('source'),
                    filename=source.get('filename')
                )
                sources.append(report_source)
    
    return sources