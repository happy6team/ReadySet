from agent_state import AgentState
from graph import create_supervisor_graph

from dotenv import load_dotenv
from vector_store.builder import ensure_vector_db_exists
from vector_store.builder import ensure_code_rule_vector_db_exists

from test_agent import test_ung_agent, test_vector_db, test_find_report_agent, test_report_writing_guide_agent

from pprint import pprint

def create_vector_db():
    ensure_vector_db_exists("./vector_store/db/reports_chroma", "./vector_store/docs/report_docs")
    print("rule_vector_db 실행")
    ensure_code_rule_vector_db_exists()
    # test_vector_db()
    
def main():
    load_dotenv()
    create_vector_db()

    graph = create_supervisor_graph()

    # 초기 상태 정의
    state = AgentState(
        input_query="",
        thread_id="thread-001",
        project_name="차세대 한국형 스마트팜 개발",
        project_explain="스마트팜 기술개발 프로젝트",
        messages=[]
    )

    # state = test_ung_agent(graph, state)
    # state = test_find_report_agent(graph, state)
    # state = test_report_writing_guide_agent(graph, state)


    print("\n===== ✅ 최종 Agent 응답 요약 ✅ =====")

    for i, msg in enumerate(state["messages"], 1):
        print(f"\n🔹 [Agent {i}]")

        if isinstance(msg, str):
            print("📄 문자열 응답:")
            print(msg.strip())

        elif isinstance(msg, dict):
            answer = msg.get("answer", "[❌ No answer]")
            print("🧠 Dict 응답 (answer):")
            print(answer.strip())

            if "sources" in msg:
                print("\n📂 참조 문서:")
                for s in msg["sources"]:
                    print(f"  - 📄 {s.get('filename', '')} (Rank {s.get('rank', '?')})")

        else:
            print("⚠️ Unknown message format:", type(msg))

    import json
    with open("debug_all_messages.json", "w", encoding="utf-8") as f:
        json.dump(state["messages"], f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    main()