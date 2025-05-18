from pydantic import BaseModel, Field
from typing import List, Optional
import re
import logging

# 요청 모델
class MatchingRequest(BaseModel):
    project_name: str = Field(..., description="프로젝트 이름")
    top_n: int = Field(3, description="프로젝트당 추천할 신입사원 수")

# 프로젝트 구인 상세 정보
class ProjectInfoResponse(BaseModel):
    project_name: str = Field(..., description="프로젝트 이름")
    project_description: str = Field(..., description="프로젝트 설명")
    project_role: str = Field(..., description="신입사원이 이 프로젝트에서 해야 할 역할")
    tech_stack: str = Field(..., description="기술 스택")
    additional_info: str = Field(..., description="추가 정보")
    
# 평가 점수 모델
class EvaluationScore(BaseModel):
    """평가 항목별 점수"""
    tech_match: float = Field(..., description="핵심 기술 일치도 (0.0-4.0)")
    project_experience: float = Field(..., description="실무 프로젝트 경험 연관성 (0.0-2.5)")
    certifications: float = Field(..., description="자격증 및 전문 역량 (0.0-2.0)")
    career_fit: float = Field(..., description="업무 연속성 및 경력 적합성 (0.0-1.5)")

# 매칭된 후보자 모델
class CandidateMatch(BaseModel):
    """매칭된 신입사원 정보"""
    name: str = Field(..., description="신입사원 이름")
    id: str = Field(..., description="신입사원 ID") 
    department: Optional[str] = Field(None, description="소속 부서")
    tech_skills: Optional[str] = Field(None, description="기술 스택")
    total_score: float = Field(..., description="종합 점수 (0.0-10.0)")
    scores: EvaluationScore = Field(..., description="평가 항목별 점수")
    reasons: List[str] = Field(..., description="선정 이유 목록")

# 매칭 결과 모델
class MatchingResponse(BaseModel):
    """매칭 응답"""
    project_info: ProjectInfoResponse = Field(..., description="프로젝트 정보")
    candidates: List[CandidateMatch] = Field(..., description="매칭된 신입사원 목록")

def parse_matching_result(content):
    """
    LLM 응답을 파싱하여 구조화된 데이터로 변환
    """
    try:
        candidates = []
        
        # 추천 인재 블록 찾기
        candidate_blocks = content.split('-추천 인재: [')[1:]
        
        for block in candidate_blocks:
            try:
                # 이름 추출
                name_match = re.search(r'이름:\s*(.*?)(?=\n|$)', block)
                name = name_match.group(1).strip() if name_match else "이름 없음"
                
                # ID 추출
                id_match = re.search(r'ID:\s*(.*?)(?=\n|$)', block)
                emp_id = id_match.group(1).strip() if id_match else None
                
                # 부서 추출
                department_match = re.search(r'부서:\s*(.*?)(?=\n|$)', block)
                department = department_match.group(1).strip() if department_match else None
                
                # 기술 스택 추출
                tech_match = re.search(r'기술 스택:\s*(.*?)(?=\n|종합 점수:)', block, re.DOTALL)
                tech_skills = tech_match.group(1).strip() if tech_match else None
                
                # 종합 점수 추출
                total_score_match = re.search(r'종합 점수:\s*(\d+\.\d+)', block)
                total_score = float(total_score_match.group(1)) if total_score_match else 0.0
                
                # 평가 항목별 점수 추출
                tech_match = re.search(r'핵심 기술 일치도:\s*(\d+\.\d+)', block)
                tech_score = float(tech_match.group(1)) if tech_match else 0.0
                
                project_exp = re.search(r'실무 프로젝트 경험 연관성:\s*(\d+\.\d+)', block)
                project_exp_score = float(project_exp.group(1)) if project_exp else 0.0
                
                cert_match = re.search(r'자격증 및 전문 역량:\s*(\d+\.\d+)', block)
                cert_score = float(cert_match.group(1)) if cert_match else 0.0
                
                career_match = re.search(r'업무 연속성 및 경력 적합성:\s*(\d+\.\d+)', block)
                career_score = float(career_match.group(1)) if career_match else 0.0
                
                # 선정 이유 추출
                reasons_text = re.search(r'선정 이유:(.*?)(?=-추천 인재:|$)', block, re.DOTALL)
                reasons = []
                
                if reasons_text:
                    reasons_raw = reasons_text.group(1).strip()
                    
                    # 숫자로 시작하는 이유를 기준으로 분리
                    numbered_reasons = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$|\])', reasons_raw, re.DOTALL)
                    
                    for reason in numbered_reasons:
                        # 쉼표나 줄바꿈으로 분리된 문장들을 하나로 합치기
                        reason_parts = [part.strip() for part in re.split(r',|\n', reason) if part.strip()]
                        complete_reason = ' '.join(reason_parts).strip()
                        
                        # 불필요한 접미사 제거
                        complete_reason = re.sub(r'\]$', '', complete_reason).strip()
                        
                        if complete_reason:
                            reasons.append(complete_reason)
                
                # 디버깅을 위한 정보 출력
                print(f"파싱된 정보: 이름={name}, ID={emp_id}, 부서={department}")
                
                candidate = CandidateMatch(
                    name=name,
                    id=emp_id,  # 필드명이 id인지 emp_id인지 확인 필요
                    department=department,
                    tech_skills=tech_skills,
                    total_score=total_score,
                    scores=EvaluationScore(
                        tech_match=tech_score,
                        project_experience=project_exp_score,
                        certifications=cert_score,
                        career_fit=career_score
                    ),
                    reasons=reasons
                )
                candidates.append(candidate)
                
            except Exception as e:
                logging.error(f"후보자 파싱 오류: {str(e)}")
                continue
        
        return candidates
    except Exception as e:
        logging.error(f"매칭 결과 파싱 오류: {str(e)}")
        return []