from fastapi import FastAPI
from api.routers import chat, meeting, reports, human_resource
from dotenv import load_dotenv
from graph import create_supervisor_graph
from agent_state import AgentState
from typing import Dict, List, Any
from api.utils.chat_history_utils import get_thread_messages, add_thread_messages, add_thread_query, get_thread_queries
from fastapi.middleware.cors import CORSMiddleware


# 애플리케이션 시작 시 환경 변수 로드 및 그래프 초기화
load_dotenv()
supervisor_graph = create_supervisor_graph()

# 기본 AgentState 인스턴스
base_agent_state = AgentState(
    input_query="",  
    thread_id="thread-001",    
    project_name="차세대 한국형 스마트팜 개발", 
    project_explain="스마트팜 기술개발 프로젝트", 
    messages=[]
)

# 쓰레드별 히스토리 저장소 초기화
thread_message_history: Dict[str, List[List[Any]]] = {} # thread_id, message list
thread_query_history: Dict[str, List[str]] = {} # thread_id, query list

app = FastAPI(
    title="ReadySet API", 
    description="신입 개발자의 빠른 실무 투입을 돕는 AI 기반 멘토링 플랫폼 ReadySet🍀의 API 명세서입니다.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vue dev 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 그래프 객체와 history를 app.state에 저장하여 전역적으로 접근 가능하게 함
app.state.supervisor_graph = supervisor_graph
app.state.base_agent_state = base_agent_state
app.state.thread_message_history = thread_message_history
app.state.thread_query_history = thread_query_history

# 히스토리 관리 함수들을 앱 상태에 등록
app.state.get_thread_messages = get_thread_messages
app.state.add_thread_messages = add_thread_messages
app.state.add_thread_query = add_thread_query
app.state.get_thread_queries = get_thread_queries

@app.get("/")
async def welcome():
    return {"message": "Welcome to the ReadySet FastAPI server!"}

app.include_router(chat.router)
app.include_router(meeting.router)
app.include_router(reports.router)
app.include_router(human_resource.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 