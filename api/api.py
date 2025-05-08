from fastapi import FastAPI
from api.routers import chat

app = FastAPI(title="TeamFit API", description="TeamFit Graph Execution API")

@app.get("/")
async def root():
    return {"message": "Welcome to the TeamFit FastAPI server!"}

app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 