from dotenv import load_dotenv
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()

# 벡터 DB 초기화
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="../vector_store/db/new_employee_chroma"
)

# 프롬프트 템플릿 정의 (수정된 평가 기준)
matching_prompt = PromptTemplate(
    input_variables=['new_employees', 'project_info', 'top_n'],
    template="""
다음은 프로젝트 요구사항입니다.
{project_info}

다음은 신입사원의 프로필 목록입니다.
{new_employees}

위 프로젝트에 가장 적합한 신입사원 {top_n}명을 선정하고 그 이유를 설명해주세요.
선정 기준:
1. 핵심 기술 일치도 (40%) - 프로젝트에 필요한 핵심 기술이 신입사원의 스킬 목록에 얼마나 포함되어 있는지
2. 실무 프로젝트 경험 연관성 (25%) - 신입사원의 기존 프로젝트 경험이 새 프로젝트와 얼마나 유사한지, 구체적인 성과를 고려
3. 자격증 및 전문 역량 (20%) - 프로젝트 관련 전문 자격증 보유 여부, 공식적으로 인증된 기술 역량
4. 업무 연속성 및 경력 적합성 (15%) - 현재 직책/부서가 프로젝트와 얼마나 일치하는지, 입사 기간과 업무 프로필 요약을 고려

다음 형식으로 반환하세요:
-프로젝트명:
-추천 인재: [
        신입사원 이름:
        종합 점수: 0.0    # 0.0~10.0 사이 점수
        평가 항목별 점수: 
            핵심 기술 일치도: 0.0,    # 0.0~4.0
            실무 프로젝트 경험 연관성: 0.0,    # 0.0~2.5
            자격증 및 전문 역량: 0.0,    # 0.0~2.0
            업무 연속성 및 경력 적합성: 0.0    # 0.0~1.5
        
        선정 이유: 
            상세한 이유 1,
            상세한 이유 2,
            상세한 이유 3
        
        ]
매칭 과정에서 각 기준별 점수를 명확하게 계산하고, 그 근거를 상세히 설명해주세요. 각 신입사원이 해당 프로젝트에 왜 적합한지 구체적인 이유를 제공하고, 각 선정 기준에 따른 점수를 투명하게 보여주세요.
"""
)

# 프로젝트와 적합한 신입사원 매칭 함수 (메타데이터 활용)
def match_project_with_employees(project_info, top_n=3):
    """
    프로젝트 정보를 기반으로 적합한 신입사원을 찾아 매칭합니다.
    메타데이터를 활용하여 결과 품질을 향상시킵니다.
    """
    try:
        # 디버깅 정보 출력
        print(f"프로젝트 정보를 기반으로 검색을 시작합니다...")
        
        # 1. 벡터 검색으로 적합한 후보 10명 찾기
        results = vectorstore.similarity_search_with_score(
            project_info, 
            k=10  # 후보 풀로 10명 검색
        )
        
        # 검색 결과 확인 (결과가 있는지)
        if not results:
            print("검색 결과가 없습니다.")
            return "적합한 신입사원을 찾을 수 없습니다."
        
        print(f"{len(results)}명의 후보를 찾았습니다.")
        
        # 2. 신입사원 정보 텍스트로 변환 (메타데이터 활용)
        employees_text = ""
        for i, (doc, score) in enumerate(results, 1):
            # 메타데이터 확인
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            # 메타데이터 정보 출력 (디버깅용)
            print(f"후보 {i}: {metadata.get('name', '이름 없음')} - 유사도: {(1-score):.3f}")
            
            # 메타데이터를 포함한 구조화된 텍스트 생성
            employee_info = f"""신입사원 #{i}:
ID: {metadata.get('id', 'ID 정보 없음')}
이름: {metadata.get('name', '이름 정보 없음')}
직책: {metadata.get('position', '직책 정보 없음')}
부서: {metadata.get('department', '부서 정보 없음')}
기술 스택: {metadata.get('skills', '기술 정보 없음')}

{doc.page_content}
유사도 점수: {(1-score):.2f}

"""
            employees_text += employee_info
        
        # 3. LLM을 사용한 상세 매칭 수행
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        matching_chain = matching_prompt | llm
        
        # 매칭 결과 생성
        print("LLM을 사용하여 최종 매칭을 수행 중...")
        result = matching_chain.invoke({
            'new_employees': employees_text,
            'project_info': project_info,
            'top_n': top_n
        })
        
        print("매칭 완료!")
        return result.content
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"매칭 과정에서 오류가 발생했습니다: {str(e)}"

# 메인 함수
def main():
    print("🤖 신입사원 매칭 시스템에 오신 것을 환영합니다! 🤖\n")
    print("프로젝트 정보를 입력하면 적합한 신입사원을 추천해 드립니다.\n")
    
    print("=" * 50)
    project_name = input("프로젝트 이름: ")
    project_description = input("프로젝트 설명: ")
    role = input("필요한 역할: ")
    skills = input("필요한 기술 스택(쉼표로 구분): ")
    
    if not project_name or not project_description:
        print("프로젝트 이름과 설명은 필수 입력 항목입니다.")
        return
    
    # 기본 프로젝트 정보 생성
    project_info = f"""
    프로젝트 이름: {project_name}
    프로젝트 설명: {project_description}
    필요한 역할: {role}
    필요한 기술 스택: {skills}
    """
    
    # 입력한 프로젝트 정보 출력
    print("\n입력하신 프로젝트 정보:")
    print(project_info)
    
    # 추가 정보 확인
    add_more = input("\n더 추가하실 항목은 없으신가요? (y/n): ")
    
    # 추가 정보 입력
    if add_more.lower() == 'y':
        additional_info = input("\n추가 정보를 입력해주세요: ")
        # 추가 정보 병합
        project_info += f"""
    추가 정보: {additional_info}
    """
        # 업데이트된 정보 출력
        print("\n업데이트된 프로젝트 정보:")
        print(project_info)
    
    # 매칭 수행
    print("\n🔍 적합한 신입사원을 찾는 중...\n")
    result = match_project_with_employees(project_info, top_n=3)
    print(result)
    
    print("\n프로그램을 종료합니다. 감사합니다!")

# # 벡터 DB 테스트 함수
# def test_vector_db():
#     """
#     벡터 DB가 제대로 설정되었는지 테스트합니다.
#     """
#     print("벡터 DB 테스트 중...")
    
#     try:
#         # 간단한 쿼리로 테스트
#         results = vectorstore.similarity_search("UI 디자이너", k=1)
        
#         if results:
#             print("✅ 벡터 DB 테스트 성공!")
#             print(f"샘플 결과: {results[0].page_content[:100]}...")
            
#             # 메타데이터 확인
#             if hasattr(results[0], 'metadata') and results[0].metadata:
#                 print("✅ 메타데이터 존재!")
#                 print(f"메타데이터 샘플: {results[0].metadata}")
#             else:
#                 print("⚠️ 메타데이터가 없습니다. 결과 품질이 저하될 수 있습니다.")
#         else:
#             print("❌ 벡터 DB에 데이터가 없습니다.")
        
#         return bool(results)
    
#     except Exception as e:
#         print(f"❌ 벡터 DB 테스트 실패: {str(e)}")
#         return False

# if __name__ == "__main__":
#     # 시작 전 벡터 DB 테스트
#     if test_vector_db():
#         main()
#     else:
#         print("\n⚠️ 벡터 DB 설정에 문제가 있습니다.")
#         print("store_new_employees.py를 먼저 실행하여 데이터를 저장해주세요.")

if __name__ == "__main__":
    main()