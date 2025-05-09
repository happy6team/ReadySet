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
    # 현재 스크립트의 절대 경로를 기준으로 파일 경로 설정
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file_path = os.path.join(current_dir, "vector_store", "docs", "employee_info", "smartfarm-employee-data-revised.txt")

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
            # print(doc)
            documents.append(doc)

    print(f"✓ {len(documents)}명의 직원 정보 파싱 완료")

     # 4. 임베딩 모델 초기화
    embedding_model = OpenAIEmbeddings()
    
    # 5. 벡터 스토어 저장 경로 설정
    persist_path = os.path.join(current_dir, "vector_store", "db", "employee_info_chroma")

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
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vector_db_path = os.path.join(current_dir, "vector_store", "db", "employee_info_chroma")
    
    return Chroma(
        embedding_function=embedding,
        persist_directory=vector_db_path
    )

def match_person_for_query(query: str, project_name: str):
    vs = load_vectorstore()
    related_employees = vs.similarity_search(query, k=3)
    # print(related_employees)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 직원 정보 포맷팅
    employee_info = ""
    for i, doc in enumerate(related_employees, 1):
        employee_info += f"직원 {i}:\n{doc.page_content}\n\n"
    
    # print("✅employee_info", employee_info)
    # 직접 추천 생성
    prompt = f"""
You are the person-in-charge matching agent for {project_name}. Your role is to analyze the user's question and connect them with the appropriate person-in-charge.

Instructions:
1. Extract keywords from the user's question.
2. ALWAYS match with one of the provided employees, regardless of question relevance to {project_name}.
3. Select the employee whose responsibilities best align with the question's keywords.
4. If no clear keyword match exists, select the employee whose role seems most general or administrative.

Employee information is as follows:  
{employee_info}

User question: {query}

Project: {project_name}

Response requirements:
- ALWAYS provide a matched employee from the list. NEVER say "적합한 담당자를 찾을 수 없습니다."
- Start immediately with the matched employee information without any introductory text
- Format your response as follows:
  이름: [이름]
  이메일: [이메일]
  부서: [부서]
  직책: [직책]
  담당업무: [담당업무]
  
  [매칭 이유 설명]
- Keep the explanation concise and direct
- Do not include any closing statements

IMPORTANT: Generate response in Korean language. Be direct and concise.
"""
    
    response = llm.invoke(prompt)
    # print(response)
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

if __name__ == "__main__":
    # initialize_employee_vectorstore()
    match_person_for_query("농업 규제에 대응하는 담당자 누구야", "스마트팜 프로젝트")