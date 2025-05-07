from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    input_query: str
    thread_id: str
    project_name: Optional[str]
    project_explain: Optional[str]
    messages: List[str] 