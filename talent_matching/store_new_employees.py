from dotenv import load_dotenv
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# 환경 변수 로드
load_dotenv()

# JSON 파일에서 신입사원 데이터 로드
json_file_path = 'data/hr_employees_data.json'

# 파일 읽기
with open(json_file_path, 'r', encoding='utf-8') as file:
    employees_data = json.load(file)

# 직원 정보를 텍스트로 변환
texts = []
metadatas = []

for employee in employees_data["employees"]:
    # 직원 정보를 텍스트로 변환
    text = f"""
    이름: {employee['name']}
    직책: {employee['position']}
    부서: {employee['department']}
    입사일: {employee['join_date']}
    기술: {', '.join(employee['skills'])}
    프로젝트: {', '.join(employee['projects'])}
    학력: {employee['education']['degree']} ({employee['education']['school']})
    자격증: {', '.join(employee['certifications'])}
    언어: {', '.join(employee['languages'])}
    프로필: {employee['profile_summary']}
    """
    
    # 메타데이터 추가
    metadata = {
        "id": employee["id"],
        "name": employee["name"],
        "position": employee["position"],
        "department": employee["department"],
        "skills": ", ".join(employee["skills"])
    }
    
    texts.append(text)
    metadatas.append(metadata)

# 벡터 DB 초기화 및 데이터 저장
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory="vector_store/db/new_employee_chroma"
)

# 저장
vectorstore.persist()

print("신입사원 데이터가 벡터 DB에 성공적으로 저장되었습니다.")