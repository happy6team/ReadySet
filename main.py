from langchain_core.runnables.config import RunnableConfig
from agents.word_explain_agent import invoke as word_agent
from agents.code_check_agent import invoke as code_agent

if __name__ == "__main__":
    thread_id = "thread-001"
    config = RunnableConfig(configurable={"thread_id": thread_id})

    # 용어 설명 Agent 실행
    print("📘 용어 설명 Agent 테스트")
    word_input = {"term": "스마트팜"}
    word_result = word_agent(word_input, config)
    print("👉 설명 결과:\n", word_result.get("explanation", "(결과 없음)"))

    print("\n" + "="*60 + "\n")

    # 코드 검수 Agent 실행
    print("🧠 코드 검수 Agent 테스트")
    user_code = """
class user_profile:
    def __init__(self):
        self.Name = "홍길동"

def GetUserName():
    return self.Name
"""
    code_input = {"code": user_code}
    code_result = code_agent(code_input, config)
    print("👉 검수 결과:\n", code_result.get("feedback", "(결과 없음)"))
