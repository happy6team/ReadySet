from typing import Optional
from fastapi import APIRouter, Body, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas.human_resource import HumanResourcePagination, HumanResourceBase
from ..schemas.matching import MatchingResponse, MatchingRequest, parse_matching_result, ProjectInfoResponse
from ..cruds.human_resource import HumanResourceRepository
from config.db_config import get_db
from talent_matching.smart_hr_matcher import match_project_with_employees

import traceback
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/hr",
    tags=["신입사원"])

@router.get("", response_model=HumanResourcePagination)
async def get_employees(
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1, description="페이지 번호"),
    size: int = Query(10, ge=1, le=100, description="페이지당 항목 수"),
    department: Optional[str] = Query(None, description="부서로 필터링"),
    position: Optional[str] = Query(None, description="직책으로 필터링")
):
    """
    직원 목록을 페이지네이션으로 조회합니다.
    
    - **page**: 가져올 페이지 번호 (기본값: 1)
    - **size**: 페이지당 항목 수 (기본값: 10, 최대: 100)
    - **department**: 부서로 필터링 (선택 사항)
    - **position**: 직책으로 필터링 (선택 사항)
    """
    repository = HumanResourceRepository(db)
    result = await repository.get_paginated(page, size, department, position)
    
    if not result["items"] and page > 1:
        raise HTTPException(status_code=404, detail="Page not found")
    
    return result

@router.get("/{employee_id}", response_model=HumanResourceBase)
async def get_employee_by_id(
    employee_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    직원 ID로 한 명의 직원 정보를 조회합니다.
    
    - **employee_id**: 조회할 직원의 고유 ID
    """
    repository = HumanResourceRepository(db)
    employee = await repository.get_by_id(employee_id)
    
    if not employee:
        raise HTTPException(status_code=404, detail="직원을 찾을 수 없습니다")
    
    return employee


# 더미 프로젝트 정보 생성 함수
def get_project_info(project_name):
    """
    프로젝트 이름을 받아 더미 프로젝트 정보를 생성합니다.
    """
    project_descriptions = {
        "웹 애플리케이션 개발": {
            "설명": "사용자 친화적인 인터페이스와 확장 가능한 백엔드를 갖춘 웹 애플리케이션 개발",
            "역할": "풀스택 개발자",
            "기술_스택": "React, Node.js, Express, MongoDB, GraphQL",
            "추가_정보": "반응형 디자인 필수, REST API 설계 경험 우대"
        },
        "신규 사업 전략 수립 프로젝트": {
            "설명": "회사의 중장기 성장을 위한 신규 사업 영역을 발굴하고 구체적인 전략을 수립하는 프로젝트입니다.이해관계자 분석과 SWOT 분석을 통해 실행 가능한 전략을 도출해야 합니다.",
            "역할": "전략 기획자",
            "기술_스택": "SWOT 분석, 이해관계자 분석, 데이터 수집, 목표 설정",
            "추가_정보": "조직 설계 경험이 있는 분을 우대합니다. PMP 자격증 보유자 선호합니다."
        },
        "모바일 앱 개발": {
            "설명": "iOS 및 Android 플랫폼용 하이브리드 모바일 애플리케이션 개발",
            "역할": "모바일 개발자",
            "기술_스택": "Flutter, Dart, Firebase, RESTful API",
            "추가_정보": "네이티브 앱 개발 경험자 우대, UI/UX 디자인 감각 필요"
        },
        "데이터 파이프라인 구축": {
            "설명": "대용량 데이터를 수집, 처리, 저장하는 ETL 파이프라인 구축",
            "역할": "데이터 엔지니어",
            "기술_스택": "Python, Spark, Airflow, AWS, Redshift",
            "추가_정보": "분산 시스템 경험 우대, 대용량 데이터 처리 경험 필요"
        },
        "AI 모델 개발": {
            "설명": "자연어 처리 및 이미지 인식을 위한 머신러닝 모델 개발",
            "역할": "AI/ML 엔지니어",
            "기술_스택": "Python, TensorFlow, PyTorch, NLTK, OpenCV",
            "추가_정보": "논문 구현 경험 우대, 모델 최적화 능력 필요"
        },
        "클라우드 인프라 구축": {
            "설명": "확장 가능하고 안정적인 클라우드 인프라 설계 및 구축",
            "역할": "클라우드 엔지니어",
            "기술_스택": "AWS, Terraform, Docker, Kubernetes, Ansible",
            "추가_정보": "마이크로서비스 아키텍처 경험 우대, DevOps 지식 필요"
        },
        "보안 시스템 강화": {
            "설명": "기업 내부 시스템 및 애플리케이션 보안 취약점 분석 및 개선",
            "역할": "보안 엔지니어",
            "기술_스택": "네트워크 보안, 침투 테스트, SIEM, Python, 암호화",
            "추가_정보": "보안 인증 보유자 우대, 취약점 분석 경험 필요"
        },
        "UI/UX 디자인": {
            "설명": "사용자 중심의 웹 및 모바일 애플리케이션 인터페이스 디자인",
            "역할": "UI/UX 디자이너",
            "기술_스택": "Figma, Adobe XD, HTML/CSS, JavaScript, 사용자 조사",
            "추가_정보": "포트폴리오 제출 필수, 사용성 테스트 경험 우대"
        },
        "블록체인 서비스 개발": {
            "설명": "탈중앙화 애플리케이션 및 스마트 컨트랙트 개발",
            "역할": "블록체인 개발자",
            "기술_스택": "Solidity, Ethereum, Web3.js, JavaScript, Go",
            "추가_정보": "암호학 지식 우대, 블록체인 프로젝트 참여 경험 필요"
        },
        "데이터 분석 시스템": {
            "설명": "기업 데이터 분석 및 인사이트 도출을 위한 시스템 개발",
            "역할": "데이터 분석가",
            "기술_스택": "Python, SQL, R, Tableau, Power BI, 통계 분석",
            "추가_정보": "데이터 시각화 경험 우대, 비즈니스 인사이트 도출 능력 필요"
        },
        "품질 보증 시스템": {
            "설명": "소프트웨어 품질 보증을 위한 테스트 자동화 시스템 구축",
            "역할": "QA 엔지니어",
            "기술_스택": "Selenium, Appium, Jenkins, JIRA, Python",
            "추가_정보": "테스트 케이스 설계 경험 우대, CI/CD 파이프라인 지식 필요"
        }
    }

        # 입력된 프로젝트 이름과 가장 유사한 키 찾기
    if project_name in project_descriptions:
        info = project_descriptions[project_name]
    else:
        # 기본 프로젝트 정보 (입력된 이름이 목록에 없는 경우)
        info = {
            "설명": "새로운 기술 스택을 활용한 혁신적인 프로젝트",
            "역할": "소프트웨어 엔지니어",
            "기술_스택": "Python, JavaScript, React, Node.js, AWS",
            "추가_정보": "팀 협업 능력 중요, 빠른 학습 능력 필요"
        }
    
    return ProjectInfoResponse(
        project_name=project_name,
        project_description=info["설명"],
        project_role=info["역할"],
        tech_stack=info["기술_스택"],
        additional_info=info["추가_정보"]
    )

@router.post("/project-matching", response_model=MatchingResponse)
async def project_matching_endpoint(request: MatchingRequest = Body(...)):
    try:
        project_name = request.project_name
        project_info = get_project_info(project_name)

        # 포맷팅된 프로젝트 정보 반환
        input_project_info = f"""
        프로젝트 이름: {project_name}
        프로젝트 설명: {project_info.project_description}
        필요한 역할: {project_info.project_role}
        필요한 기술 스택: {project_info.tech_stack}
        추가 정보: {project_info.additional_info}
        """
        
        # 프로젝트 매칭 실행
        logger.info(f"'{project_name}' 프로젝트에 대한 매칭 시작")
        matching_result= match_project_with_employees(input_project_info, request.top_n)
        print(matching_result)
        
        # 매칭 결과 파싱
        candidates = parse_matching_result(matching_result)
        
        return MatchingResponse(
            project_info=project_info,
            candidates=candidates
        )
    
    except Exception as e:
        logger.error(f"API 처리 중 오류: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")
    
# 프로젝트 더미 데이터 확인 엔드포인트
@router.get("/projects")
async def get_sample_projects():
    sample_projects = [
        "웹 애플리케이션 개발",
        "신규 사업 전략 수립 프로젝트",
        "모바일 앱 개발",
        "데이터 파이프라인 구축",
        "AI 모델 개발",
        "클라우드 인프라 구축",
        "보안 시스템 강화",
        "UI/UX 디자인",
        "블록체인 서비스 개발",
        "데이터 분석 시스템",
        "품질 보증 시스템"
    ]
    return {"projects": sample_projects}
