from langchain_core.graphs import StateGraph
from typing import Dict, Any
import time

# 각 에이전트 모듈에서 invoke 함수 가져오기
from new_employee_agent import invoke as employee_invoke
from project_requirement_agent import invoke as project_invoke
from project_match_agent import invoke as project_matching_invoke

def create_workflow_graph():
    """워크플로우 그래프 생성"""
    # 워크플로우 초기화
    workflow = StateGraph(name="Project-Matching-Workflow")
    
    # 노드 추가 - 각 에이전트의 invoke 함수 사용
    workflow.add_node("extract_applicants", employee_invoke)
    workflow.add_node("extract_projects", project_invoke)
    workflow.add_node("match_applicants_projects", project_matching_invoke)
    
    # 시작 노드 설정 (병렬 실행을 위해 두 개 설정)
    workflow.set_entry_point("extract_applicants")
    workflow.set_entry_point("extract_projects")
    
    # 엣지 추가 - 두 추출 노드가 모두 매칭 노드로 연결
    workflow.add_edge("extract_applicants", "match_applicants_projects")
    workflow.add_edge("extract_projects", "match_applicants_projects")
    
    # 조건부 엣지 추가 (워크플로우 종료 조건)
    workflow.add_conditional_edges(
        "match_applicants_projects",
        lambda x: True,  # 항상 종료
        {
            True: "END"
        }
    )
    
    # 워크플로우 컴파일
    return workflow.compile()

def run_workflow(
    applicant_path="../vector_store/docs/applicant_profiles/applicant_profiles.txt", 
    project_path="../vector_store/docs/project_requirement/project_requirements_kor.pdf", 
    top_n=3
):
    """워크플로우 실행 함수"""
    # 초기 상태 설정
    initial_state = {
        "messages": [],
        "applicant_path": applicant_path,
        "project_path": project_path,
        "pdf_path": project_path,  # project_invoke에서 사용하는 키 이름
        "text_path": applicant_path,  # employee_invoke에서 사용할 수 있는 키 이름
        "top_n": top_n
    }
    
    # 워크플로우 생성 및 실행
    print("워크플로우 생성 및 실행 중...")
    workflow = create_workflow_graph()
    final_state = workflow.invoke(initial_state)
    
    return final_state

if __name__ == "__main__":
    # 워크플로우 실행
    print("=== 프로젝트-신입사원 매칭 워크플로우 시작 ===")
    start_time = time.time()
    
    result = run_workflow()
    
    # 결과 출력
    print("\n=== 메시지 목록 ===")
    for msg in result.get("messages", []):
        print(msg)
    
    # 매칭 결과 확인
    if "matching_results" in result:
        print("\n=== 최종 매칭 결과 ===")
        print(result["matching_results"])
    
    # 총 실행 시간 계산 - 한 번만 출력
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== 총 실행 시간: {total_time:.2f}초 ===")