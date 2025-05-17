from dotenv import load_dotenv
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# 환경 변수 로드
load_dotenv()

print("=============== 신입사원 데이터 로드 시작 ===============")

# JSON 파일 로드
json_file_path = '../data/hr_employees_data.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    employees_data = json.load(file)

# 데이터 개수 확인
total_employees = len(employees_data["employees"])
print(f"총 신입사원 수: {total_employees}")

# 데이터 크기 제한 (100명만 사용)
sample_size = 100  # 매우 작게 설정
reduced_employees = employees_data["employees"][:sample_size]
# print(f"샘플링한 신입사원 수: {len(reduced_employees)}")
# print(f"첫 번째 신입사원: {reduced_employees[0]['name']}")
# print(f"마지막 신입사원: {reduced_employees[-1]['name']}")

# print("=============== 텍스트 변환 시작 ===============")

# 직원 정보를 텍스트로 변환 - reduced_employees 사용!
texts = []
metadatas = []

for employee in reduced_employees:  # reduced_employees 사용
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

# print(f"변환된 텍스트 수: {len(texts)}")
# print(f"첫 번째 텍스트 샘플: {texts[0][:100]}...")

print("=============== 배치 처리 시작 ===============")

# 배치 크기 설정
batch_size = 10  # 매우 작게 설정
print(f"배치 크기: {batch_size}")

# 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings()

# 벡터 DB 초기화
vectorstore = None

# 배치 처리
for i in range(0, len(texts), batch_size):
    batch_end = min(i + batch_size, len(texts))
    # print(f"배치 처리 중: {i} ~ {batch_end-1} / {len(texts)}")
    
    batch_texts = texts[i:batch_end]
    batch_metadatas = metadatas[i:batch_end]
    
    if vectorstore is None:
        # 첫 번째 배치
        # print(f"첫 번째 배치 처리 - 신규 벡터 DB 생성")
        vectorstore = Chroma.from_texts(
            texts=batch_texts,
            embedding=embedding_model,
            metadatas=batch_metadatas,
            persist_directory="../vector_store/db/new_employee_chroma"
        )
    else:
        # 이후 배치
        # print(f"후속 배치 처리 - 기존 벡터 DB에 추가")
        vectorstore.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas
        )
    
    # 각 배치 후 저장
    vectorstore.persist()
    # print(f"배치 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} 완료")

print("=============== 처리 완료 ===============")
print(f"총 {len(texts)}개의 신입사원 데이터가 벡터 DB에 성공적으로 저장되었습니다.")