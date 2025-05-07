from agents.exception_agent import invoke as exception_agent
from langchain_core.runnables.config import RunnableConfig

if __name__ == "__main__":
    # 테스트용 질문 (다른 agent들이 처리하지 못하는 일반 질문)
    input_state = {"message": "회사 점심시간이 언제에요?"}
    config = RunnableConfig(configurable={"thread_id": "thread-999"})

    result = exception_agent(input_state, config)
    print("🧩 예외 처리 Agent 응답:\n")
    print(result.get("fallback_answer", "(결과 없음)"))
