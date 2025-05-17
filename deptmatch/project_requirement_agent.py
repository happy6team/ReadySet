from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# PDF 로더로 프로젝트 요구사항 문서 로드
def load_pdf_document(pdf_path):
    """PDF 문서를 로드하고 텍스트 추출"""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        # 모든 페이지 텍스트를 하나의 문자열로 결합
        full_text = "\n".join([doc.page_content for doc in docs])
        return full_text
    except Exception as e:
        print(f"문서 로딩 중 오류 발생: {e}")
        return None

# 프로젝트 요구사항 추출 함수
def extract_project_requirements(text, model_name="gpt-4o-mini"):
    """텍스트에서 프로젝트 요구사항 추출"""
    # LLM 설정
    llm = ChatOpenAI(model=model_name)
    
    # 프로젝트 요구사항 추출을 위한 프롬프트
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
아래는 여러 프로젝트의 요구사항이 담긴 PDF 문서 원문입니다.

{text}

각 프로젝트별로 다음 JSON 형식으로 추출하세요:
-project_id:          # 예: prj-001
-project_name:        # 프로젝트 이름
-description:           # 프로젝트 설명
-roles:                 # 역할 목록
-skills: [],                # 필요 기술 스택
-personality:           # 성격적 요구사항


모든 정보를 빠짐없이 추출하고, 정확한 JSON 형식으로 반환하세요.
"""
    )
    
    # 체인 결합
    requirement_chain = prompt | llm
    
    # 요구사항 추출
    result = requirement_chain.invoke({"text": text})
    return result.content

# 전체 요구사항 추출 함수
def extract_all_requirements(pdf_path=None):
    """PDF에서 모든 프로젝트 요구사항 추출"""
    if pdf_path is None:
        pdf_path="../vector_store/docs/project_requirement/project_requirements_kor.pdf"
        
    full_text = load_pdf_document(pdf_path)
    if not full_text:
        return "PDF 로딩에 실패했습니다."
    
    # 요구사항 추출
    return extract_project_requirements(full_text)

# LangGraph Supervisor용 invoke 함수
def invoke(state: dict, config) -> dict:
    """LangGraph에서 사용할 invoke 함수"""
    # PDF 경로 설정
    pdf_path = state.get("pdf_path", "../vector_store/docs/project_requirement/project_requirements_kor.pdf")
    
    # 이미 로드된 전체 텍스트에서 요구사항 추출
    full_text = load_pdf_document(pdf_path)
    if full_text:
        requirements = extract_project_requirements(full_text)
    else:
        requirements = "PDF 로딩에 실패했습니다."
    
    # 이전 메시지 유지
    new_messages = list(state.get("messages", []))
    new_messages.append(f"🧠 프로젝트 요구사항 추출 결과:\n{requirements}")

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
    result = extract_all_requirements()
    # LLM의 응답(content) 출력
    print("=== 프로젝트 요구사항 추출 결과 ===")
    print(result)