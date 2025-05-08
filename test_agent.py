from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import RunnableConfig
from agent_state import AgentState
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

def pretty_print_result(state: AgentState):
    # 결과 예쁘게 출력
    print("\n=== 응답 결과 ===")
    for msg in state["messages"]:
        if "answer" in msg:
            print("\n답변:")
            print(msg["answer"])
            
            if "sources" in msg:
                print("\n참고 문서:")
                for i, source in enumerate(msg["sources"], 1):
                    print(f"\n[{i}] {source.get('section', 'Unknown')}")
                    print(f"파일이름: {source.get('filename', 'Unknown')}")
                    print(f"출처: {source.get('source', 'Unknown')}")
                    print(f"내용: {source.get('content', '')[:200]}...")
        elif "error" in msg:
            print("\n오류:")
            print(msg["error"])

def test_find_report_agent(graph: StateGraph, state: AgentState):
    # 문서 검색 agent 테스트용
    state["input_query"] = """스마트팜 프로젝트의 단계별 추진 체계와 책임자를 문서에서 찾아주세요."""
    state = graph.invoke(
        state,
        config=RunnableConfig(configurable={"thread_id": "thread-001"})
    )

    pretty_print_result(state)

    return state

def test_report_writing_guide_agent(graph: StateGraph, state: AgentState):
    # 보고서 생성 가이드라인 제공 agent 테스트용
    state["input_query"] = """회의록 작성 시 의결사항에 대해서는 어떻게 작성하는게 좋을까요?"""
    state = graph.invoke(
        state,
        config=RunnableConfig(configurable={"thread_id": "thread-001"})
    )

    pretty_print_result(state)

    return state

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

    return state

def test_email_agent(graph: StateGraph, state: AgentState):
    # 보고서 생성 가이드라인 제공 agent 테스트용
    state["input_query"] = """변경, 이재웅, 정중하게, 회의실 8층으로 변경"""
    state = graph.invoke(
        state,
        config=RunnableConfig(configurable={"thread_id": "thread-001"})
    )

    pretty_print_result(state)
