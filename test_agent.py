from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import RunnableConfig
from main import AgentState
from langgraph.graph import StateGraph

from vector_store.retrieval import test_vector_retrieval

def test_vector_db():
    # 벡터 DB 검색 테스트
    results = test_vector_retrieval(
        query="스마트팜 프로젝트의 단계별 추진 체계와 책임자는 누구인가요?",
        k=3,  # 상위 3개 결과 검색
        db_path="./vector_store/db/reports_chroma"
    )
    print(results)

def test_ung_agent(graph: StateGraph, state: AgentState):
    # 1차: 용어 설명 테스트
    state = graph.invoke(
        state,
        config=RunnableConfig(configurable={"thread_id": "thread-001"})
    )

    # 2차: 코드 검수 테스트용으로 input 변경
    state["input_query"] = """
class user_profile:
    def __init__(self):
        self.Name = "홍길동"

# def GetUserName():
#     return self.Name
# """

    state = graph.invoke(
        state,
        config=RunnableConfig(configurable={"thread_id": "thread-001"})
    )

    # 3차: 코드 검수 테스트용으로 input 변경
    state["input_query"] = """오늘 점심이 뭐야?
"""
    state = graph.invoke(
        state,
        config=RunnableConfig(configurable={"thread_id": "thread-001"})
    )

