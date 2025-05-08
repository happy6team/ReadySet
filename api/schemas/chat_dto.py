from pydantic import BaseModel
from typing import List, Optional, Any

class QueryRequest(BaseModel):
    input_query: str
    thread_id: str
    project_name: str
    project_explain: str

class ReportSource(BaseModel):
    content: Optional[str] = None
    section: Optional[str] = None
    source: Optional[str] = None
    filename: Optional[str] = None
    rank: Optional[int] = None

class Message(BaseModel):
    content: str
    sources: Optional[List[ReportSource]] = None
    type: str

class QueryResponse(BaseModel):
    messages: List[Message]

def map_to_message(msg: Any) -> Message:
    """
    에이전트 메시지를 DTO Message 객체로 변환하는 매퍼 함수
    
    Args:
        msg: 변환할 메시지 객체
        
    Returns:
        Message: 변환된 DTO Message 객체
    """

    msg_type = msg.__class__.__name__
    content_str = str(msg)
    sources = None

    content_dict = msg
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

    return Message(
        content=content_str,
        type=msg_type,
        sources=sources
    )
    
