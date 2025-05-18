from typing import List, Dict, Any, Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from models.human_resource import HumanResource

class HumanResourceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_paginated(
        self, 
        page: int = 1, 
        size: int = 10, 
        department: Optional[str] = None,
        position: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        페이지네이션된 직원 목록을 조회합니다.
        필터링 옵션을 포함합니다.
        """
        # 기본 쿼리 생성
        query = select(HumanResource)
        count_query = select(func.count()).select_from(HumanResource)
        
        # 필터 적용
        if department:
            query = query.filter(HumanResource.department == department)
            count_query = count_query.filter(HumanResource.department == department)
        
        if position:
            query = query.filter(HumanResource.position == position)
            count_query = count_query.filter(HumanResource.position == position)
        
        # 페이지네이션 적용
        query = query.limit(size).offset((page - 1) * size)
        
        # 쿼리 실행
        result = await self.session.execute(query)
        total = await self.session.execute(count_query)
        
        items = result.scalars().all()
        total_count = total.scalar()
        total_pages = (total_count + size - 1) // size  # 올림 나눗셈
        
        return {
            "items": items,
            "total": total_count,
            "page": page,
            "size": size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    
    async def get_by_id(self, employee_id: str) -> Optional[HumanResource]:
        """
        직원 ID로 한 명의 직원 정보를 조회합니다.
        """
        query = select(HumanResource).where(HumanResource.id == employee_id)
        result = await self.session.execute(query)
        return result.scalars().first()
    
