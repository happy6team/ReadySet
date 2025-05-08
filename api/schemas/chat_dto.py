from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    input_query: str
    thread_id: str
    project_name: str
    project_explain: str

class Message(BaseModel):
    content: str
    type: str

class QueryResponse(BaseModel):
    messages: List[Message]