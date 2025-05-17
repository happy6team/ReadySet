from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from new_employee_agent import extract_applicant_profile
from project_requirement_agent import extract_all_requirements
import time

load_dotenv()

def project_matching(new_employee, projects, top_n):
    llm = ChatOpenAI(model="gpt-4o-mini") 

    prompt = PromptTemplate(
        input_variables=['new_employee', 'projects', 'top_n'],  # top_n 변수 추가
        template="""
다음은 프로젝트 요구사항입니다.
{projects}

다음은 신입사원의 프로필 목록입니다.
{new_employee}

각 프로젝트별로 가장 적합한 신입사원 {top_n}명을 선정하고 그 이유를 설명해주세요.
선정 기준:
1. 기술 스택 일치도 (40%)
2. 역할 적합성 (30%)
3. 성격적 요구사항 부합도 (20%)
4. 희망부서 일치도 (10%)

다음 형식으로 반환하세요:
-프로젝트ID:
-프로젝트명:
-추천 인재: [
        신입사원 이름:
        종합 점수: 0.0    # 0.0~10.0 사이 점수
        평가 항목별 점수: 
            기술 스택 일치도: 0.0,    # 0.0~4.0
            역할 적합성: 0.0,         # 0.0~3.0
            성격 부합도: 0.0,         # 0.0~2.0
            희망부서 일치도: 0.0       # 0.0~1.0
        
        선정 이유: 
            상세한 이유 1,
            상세한 이유 2,
            상세한 이유 3
        
        ]
매칭 과정에서 각 기준별 점수를 명확하게 계산하고, 그 근거를 상세히 설명해주세요. 각 신입사원이 해당 프로젝트에 왜 적합한지 구체적인 이유를 제공하고, 각 선정 기준에 따른 점수를 투명하게 보여주세요.
"""
    )
    matching_chain = prompt | llm
    result = matching_chain.invoke({
        'new_employee': new_employee,
        'projects': projects, 
        'top_n': top_n
    })
    return result.content

def process_project_matching():
    # 신입사원 프로필 추출
    new_employee = extract_applicant_profile()
    # 프로젝트 요구사항 추출
    projects = extract_all_requirements()
    # 매칭 실행
    matching_result = project_matching(new_employee, projects, top_n=3)
    return matching_result

def invoke(state:dict, config) -> dict:
    project_matching_result = process_project_matching()

    new_messages = list(state.get("message",[]))
    new_messages.append(f"🧠 프로젝트별 추천 인재 결과:\n{project_matching_result}")

    # thread_id 설정
    thread_id = (
        getattr(config, "configurable", {}).get("thread_id")
        if hasattr(config, "configurable")
        else config.get("thread_id", "default")
    )

    return {
        **state,
        "messages": new_messages,
        "thread_id": thread_id,
    }

if __name__ == "__main__":
    # 직접 실행 시 추출 함수를 호출하고 결과를 출력
    result = process_project_matching()
    # LLM의 응답(content) 출력
    print("=== 프로젝트 매칭결과 ===")
    print(result)