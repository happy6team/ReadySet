from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas.human_resource import HumanResourcePagination
from ..cruds.human_resource import HumanResourceRepository
from config.db_config import get_db

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