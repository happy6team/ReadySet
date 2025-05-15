from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 텍스트 파일 로더로 신입사원 프로필 문서 로드
def load_text_document(text_path):
    """텍스트 문서를 로드하고 텍스트 추출"""
    try:
        with open(text_path, 'r', encoding='utf-8') as file:
            full_text = file.read()
        return full_text
    except Exception as e:
        print(f"문서 로딩 중 오류 발생: {e}")
        return None

# LLM을 사용하여 신입사원 정보 추출
def extract_applicant_info_llm(text, model_name="gpt-4o-mini"):
    """LLM을 사용하여 신입사원 정보 추출"""
    # LLM 설정
    llm = ChatOpenAI(model=model_name)
    
    # 신입사원 정보 추출을 위한 프롬프트
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
아래는 신입사원 지원자들의 프로필 정보가 담긴 텍스트 문서입니다.

{text}

각 지원자별로 다음 JSON형식으로 정보를 추출하세요:
- 이름: 신입사원 이름
- 전공: 학교 전공
- 기술 스택: 신입사원이 가진 기술 목록
- 역할: 희망하는 직무/역할
- 희망부서: 1지망, 2지망, 3지망
- 인적성검사 결과: 의사소통, 논리력, 창의력, 리더십, 책임감 점수

모든 지원자 정보를 빠짐없이 추출하고, 정확한 형식으로 반환하세요.
반드시 {text}에 있는 사람 정보만 추출하세요.
"""
    )
    
    # 체인 결합
    extraction_chain = prompt | llm
    
    # 정보 추출
    result = extraction_chain.invoke({"text": text})
    return result

def extract_applicant_profile(text_path=None):
    if text_path is None:
        text_path = "../vector_store/docs/applicant_profiles/applicant_profiles.txt"

    full_text = load_text_document(text_path)
    if not full_text:
        return "텍스트 파일 로딩에 실패했습니다."
    
    return extract_applicant_info_llm(full_text)

def invoke(state:dict, config) -> dict:
    #.txt 경로 설정
    text_path = state.get("../vector_store/docs/applicant_profiles/applicant_profiles.txt")

    full_text = load_text_document(text_path)
    if full_text:
        new_employee = extract_applicant_profile(full_text)
    else:
        new_employee = "텍스트 파일 로딩에 실패했습니다."
    
    new_messages = list(state.get("message", []))
    new_messages.append(f"🧠 프로젝트 요구사항 추출 결과:\n{new_employee}")
        
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

if __name__ == '__main__':
    result = extract_applicant_profile()
    print ("===신입사원 파싱 결과===")
    print(result)

