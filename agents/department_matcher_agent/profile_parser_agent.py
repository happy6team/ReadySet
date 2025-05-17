from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import re

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

# 정규식을 사용하여 텍스트에서 직접 신입사원 정보 추출
import re
import json

def extract_applicant_info_regex(text):
    """정규식을 사용하여 신입사원 정보 추출 (이름 기준)"""
    applicants = []
    
    # 각 지원자 정보 블록 분리 (‘이름: …’ 부터 다음 ‘이름:’ 이전 또는 문서 끝까지)
    block_pattern = r'이름: .+?(?=(?:\n이름: |\Z))'
    blocks = re.findall(block_pattern, text, re.DOTALL)
    
    for block in blocks:
        applicant = {}
        
        # 이름 추출
        name_match = re.search(r'이름: (.+?)\n', block)
        if name_match:
            applicant['name'] = name_match.group(1).strip()
        
        # 전공 추출
        major_match = re.search(r'전공: (.+?)\n', block)
        if major_match:
            applicant['major'] = major_match.group(1).strip()
        
        # 기술 스택 추출
        skills_match = re.search(r'프로젝트/기술 스택: (.+?)\n', block)
        if skills_match:
            applicant['skills'] = [s.strip() for s in skills_match.group(1).split(',')]
        
        # 역할 추출
        role_match = re.search(r'역할: (.+?)\n', block)
        if role_match:
            applicant['role'] = role_match.group(1).strip()
        
        # 희망부서(1~3지망) 추출
        dept_matches = re.findall(r'(\d)지망: (.+?)\n', block)
        # {'first_choice': '...', 'second_choice': '...', 'third_choice': '...'}
        mapping = {'1': 'first_choice', '2': 'second_choice', '3': 'third_choice'}
        applicant['preferred_departments'] = {
            mapping[num]: dept.strip() for num, dept in dept_matches if num in mapping
        }
        
        # 인적성검사 결과 추출
        personality_match = re.search(r'인적성검사 결과: (\{.+?\})', block)
        if personality_match:
            json_str = personality_match.group(1).replace("'", '"')
            try:
                applicant['personality_results'] = json.loads(json_str)
            except json.JSONDecodeError:
                applicant['personality_results'] = json_str
        
        applicants.append(applicant)
    
    return applicants


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

각 지원자별로 다음 JSON 형식으로 추출하세요:
[
  {
    "name": "",                    # 예: "김민수"
    "major": "",                   # 전공
    "skills": [],                  # 기술 스택 (배열)
    "role": "",                    # 역할
    "preferred_departments": {     # 희망부서 (1, 2, 3지망)
      "first_choice": "",
      "second_choice": "",
      "third_choice": ""
    },
    "personality_results": {       # 인적성검사 결과
      "의사소통": "",
      "논리력": "",
      "창의력": "",
      "리더십": "",
      "책임감": ""
    }
  }
]

모든 지원자 정보를 빠짐없이 추출하고, 정확한 JSON 형식으로 반환하세요.
"""
    )
    
    # 체인 결합
    extraction_chain = prompt | llm
    
    # 정보 추출
    result = extraction_chain.invoke({"text": text})