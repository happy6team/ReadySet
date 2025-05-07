import os
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def initialize_employee_vectorstore():
    """직원 정보 텍스트 파일을 구조화된 방식으로 벡터 스토어에 저장합니다."""
    
    # 1. TXT 파일 로드
    txt_file_path = "employee_info/smartfarm-employee-data-revised.txt"
    try:
        with open(txt_file_path, mode='r', encoding='utf-8') as file:
            text_content = file.read()
        print(f"✓ 텍스트 파일 로드 완료: {len(text_content)} 바이트")
    except Exception as e:
        print(f"❌ 파일 로드 오류: {str(e)}")
        return None
    
    # 2. 텍스트를 행으로
    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
    
    # 3. 각 행을 직원 정보로 파싱
    documents = []

    for line in lines:
        # 쉼표로 분리된 값 처리
        parts = line.split(',')
        if len(parts) >= 5:  # 최소한 5개 필드가 있는지 확인
            name = parts[0].strip()
            email = parts[1].strip()
            department = parts[2].strip()
            position = parts[3].strip()
            job_description = parts[4].strip()
            
            # 직원 정보 텍스트 생성
            employee_info = f"""
            이름: {name}
            이메일: {email}
            부서: {department}
            직책: {position}
            담당업무: {job_description}
            """
            # Document 객체 생성
            doc = Document(page_content=employee_info)
            documents.append(doc)

    print(f"✓ {len(documents)}명의 직원 정보 파싱 완료")

     # 4. 임베딩 모델 초기화
    embedding_model = OpenAIEmbeddings()
    
    # 5. 벡터 스토어 저장 경로 설정
    persist_path = f"vector_store/employee_info_chroma"

    # 6. 디렉토리가 없으면 생성
    os.makedirs(persist_path, exist_ok=True)

    # 7. Chroma 벡터 스토어 생성
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_path,
    )

    # 8. 벡터 스토어 저장
    vectorstore.persist()
    print(f"직원 정보 벡터 DB 저장 완료: {persist_path}")
    
    return vectorstore

# VectorDB 로딩
def load_vectorstore():
    embedding = OpenAIEmbeddings()
    return Chroma(
        embedding_function=embedding,
        persist_directory="vector_store/employee_info_chroma"
    )

def match_person_for_query(query: str, project_name: str):
    vs = load_vectorstore()
    related_employees = vs.similarity_search(query, k=3)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 직원 정보 포맷팅
    employee_info = ""
    for i, doc in enumerate(related_employees, 1):
        employee_info += f"직원 {i}:\n{doc.page_content}\n\n"
    
    # 직접 추천 생성
    prompt = f"""
당신은 {project_name}의 담당자 매칭 에이전트입니다. 사용자의 질문을 분석하여 적합한 담당자를 연결해주는 역할을 합니다.

1. 사용자 질문에서 {project_name} 관련 업무 키워드를 추출하세요.
2. 추출된 키워드와 관련성이 높은 담당자를 찾아 추천하세요.
3. 질문이 {project_name}과 관련이 없거나 키워드를 추출할 수 없는 경우 "적합한 담당자가 없습니다. {project_name} 관련 질문을 해주세요."라고 답변하세요.

답변 형식:
- {project_name} 관련 질문인 경우: 담당자 정보와 추천 이유 제공
- {project_name} 무관 질문인 경우: "적합한 담당자가 없습니다." 메시지 제공

직원 정보는 다음과 같습니다:
{employee_info}

사용자 질문: {query}

프로젝트: {project_name}

먼저 이 질문이 {project_name}과 관련이 있는지 판단하고, 관련이 있다면 반드시 위 직원 정보 중에서 가장 적합한 담당자 한 명을 선택해 추천해주세요.
"""
    
    response = llm.invoke(prompt)
    return response.content



