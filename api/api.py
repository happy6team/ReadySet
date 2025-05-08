from fastapi import FastAPI
from api.routers import chat
from dotenv import load_dotenv
from graph import create_supervisor_graph

# 애플리케이션 시작 시 환경 변수 로드 및 그래프 초기화
load_dotenv()
supervisor_graph = create_supervisor_graph()

app = FastAPI(
    title="TeamFit API", 
    description="TeamFit Graph Execution API"
)

# 그래프 객체를 app.state에 저장하여 전역적으로 접근 가능하게 함
app.state.supervisor_graph = supervisor_graph

@app.get("/")
async def root():
    return {"message": "Welcome to the TeamFit FastAPI server!"}

app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 