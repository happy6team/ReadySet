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
    txt_file_path = "vector_store/docs/employee_info"
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
    persist_path = f"vector_store/db/employee_info_chroma"

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
        persist_directory="vector_store/db/employee_info_chroma"
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
You are the person-in-charge matching agent for {project_name}. Your role is to analyze the user's question and connect them with the appropriate person-in-charge.

1. Extract {project_name}-related task keywords from the user's question.  
2. Find and recommend the most relevant person-in-charge based on the extracted keywords.  
3. If the question is unrelated to {project_name} or no keywords can be extracted, respond with:  
   "No suitable person-in-charge found. Please ask a question related to {project_name}."

Response format:  
- If the question is related to {project_name}: Provide the person-in-charge information and the reason for the recommendation.  
- If the question is unrelated to {project_name}: Provide the message "No suitable person-in-charge found."

Employee information is as follows:  
{employee_info}

User question: {query}

Project: {project_name}

First, determine whether the question is related to {project_name}. If it is, select and recommend the single most appropriate person-in-charge from the above employee information.

IMPORTANT: Please make sure to generate in Korean language.
"""
    
    response = llm.invoke(prompt)
    return response.content

# LangGraph Supervisor용 invoke 함수 
def invoke(state: dict, config) -> dict:
    """LangGraph Supervisor용 invoke 함수"""
    # 입력 쿼리 가져오기
    query = state.get("input_query", "")
    project_name = state.get("project_name", "스마트팜 프로젝트")  # 기본값 설정

    # 설정에서 thread_id 가져오기
    thread_id = (
        getattr(config, "configurable", {}).get("thread_id")
        if hasattr(config, "configurable")
        else config.get("thread_id", "default")
    )

    # 담당자 매칭 함수 호출
    result = match_person_for_query(query, project_name)
    # print(f"담당자 매칭 결과:\n{result}")

    # messages 누적
    new_messages = list(state.get("messages", []))  # 기존 메시지 유지
    new_messages.append(f"👨‍💼 담당자 매칭 결과:\n{result}")

    return {
        **state,
        "messages": new_messages,
        "thread_id": thread_id,
        "matching_result": result  # 매칭 결과 추가
    }