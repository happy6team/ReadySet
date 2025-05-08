from pydantic import BaseModel
from typing import List, Optional

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