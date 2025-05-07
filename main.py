from agent_state import AgentState
from graph import create_supervisor_graph

from dotenv import load_dotenv
from vector_store.builder import ensure_vector_db_exists
from test_agent import test_ung_agent, test_vector_db, test_find_report_agent, test_report_writing_guide_agent

def create_reports_vector_db():
    ensure_vector_db_exists("./vector_store/db/reports_chroma", "./vector_store/docs")
    # test_vector_db()
    
def main():
    load_dotenv()
    # create_reports_vector_db()

    graph = create_supervisor_graph()

    # 초기 상태 정의
    state = AgentState(
        input_query="스마트팜이 뭐야?",
        thread_id="thread-001",
        project_name="차세대 한국형 스마트팜 개발",
        project_explain="스마트팜 기술개발 프로젝트",
        messages=[]
    )

    # state = test_ung_agent(graph, state)
    state = test_find_report_agent(graph, state)
    # state = test_report_writing_guide_agent(graph, state)

    print("💬 저장된 메시지:")
    for i, msg in enumerate(state["messages"], 1):
        print(f"{i}. {msg}\n")

if __name__ == "__main__":
    main()