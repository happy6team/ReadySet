from typing import Literal, Optional, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from agents.word_explain_agent import invoke as word_agent
from agents.code_check_agent import invoke as code_agent
from agents.exception_agent import invoke as exception_agent

from langchain_core.runnables.config import RunnableConfig

from dotenv import load_dotenv
from vector_store.builder import ensure_vector_db_exists
from vector_store.retrieval import test_vector_retrieval

class AgentState(TypedDict):
    input_query: str
    thread_id: str
    project_name: Optional[str]
    project_explain: Optional[str]
    messages: List[str]

# 라우팅 프롬프트 체인 정의
router_prompt = PromptTemplate.from_template("""
당신은 사용자의 질문 또는 코드 입력을 보고 아래 중 어떤 기능이 필요한지 판단하는 AI 라우터입니다.
각 기능의 목적은 다음과 같습니다:

1. word_explain: 프로젝트 관련 용어나 개념 설명
2. code_check: 사용자가 작성한 코드에 대해 규칙 검토
3. exception_agent: 어떤 기능에도 해당하지 않음

사용자가 입력한 내용이 코드처럼 보이면 'code_check'로 판단하세요.

다음 사용자 입력에 가장 적합한 기능 이름만 한 단어로 출력해주세요. (예: code_check)

입력:
{input_query}
""")

router_chain = router_prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

# 라우팅 함수

def route_agent(state: AgentState) -> Literal["word_explain", "code_check", "exception_agent"]:
    result = router_chain.invoke({"input_query": state["input_query"]}).strip().lower()

    print(f"🧭 라우팅 결과: {result}")  # 🔍 Debug 출력

    if result in {"word_explain", "code_check"}:
        return result
    return "exception_agent"


# Supervisor Graph 생성 함수
from copy import deepcopy 

def create_supervisor_graph():
    builder = StateGraph(AgentState)

    def wrap_agent(agent_func):
        def wrapper(state: AgentState, config: RunnableConfig) -> AgentState:
            result = agent_func(state, config)
            new_state = deepcopy(state)

             # 기존 메시지 유지하고 병합만 수행
            new_state = state.copy()
            new_state["messages"] = result.get("messages", state.get("messages", []))

            return new_state
        return wrapper

    builder.add_node("word_explain", wrap_agent(word_agent))
    builder.add_node("code_check", wrap_agent(code_agent))
    builder.add_node("exception_agent", wrap_agent(exception_agent))

    builder.set_conditional_entry_point(route_agent)

    builder.add_edge("word_explain", END)
    builder.add_edge("code_check", END)
    builder.add_edge("exception_agent", END)

    return builder.compile()

def create_reports_vector_db():
    ensure_vector_db_exists("./vector_store/db/reports_chroma", "./vector_store/docs")

    # 벡터 DB 검색 테스트
    results = test_vector_retrieval(
        query="스마트팜 프로젝트의 단계별 추진 체계와 책임자는 누구인가요?",
        k=3,  # 상위 3개 결과 검색
        db_path="./vector_store/db/reports_chroma"
    )
    print(results)
    

def main():
    load_dotenv()
    create_reports_vector_db()

    graph = create_supervisor_graph()

    # 초기 상태 정의
    state = AgentState(
        input_query="스마트팜이 뭐야?",
        thread_id="thread-001",
        project_name="차세대 한국형 스마트팜 개발",
        project_explain="스마트팜 기술개발 프로젝트",
        messages=[]
    )

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

def GetUserName():
    return self.Name
"""

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



    # 결과 출력
    print("💬 저장된 메시지:")
    for i, msg in enumerate(state["messages"], 1):
        print(f"{i}. {msg}\n")

if __name__ == "__main__":
    main()