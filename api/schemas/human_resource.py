from typing import List, Optional, Any
from datetime import date, datetime
from pydantic import BaseModel, Field

class HumanResourceBase(BaseModel):
    id: str
    name: str
    position: str
    department: str
    join_date: date
    skills: List[str]
    projects: List[str]
    education_degree: Optional[str] = None
    education_school: Optional[str] = None
    education_graduation_year: Optional[int] = None
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    profile_summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class HumanResourcePagination(BaseModel):
    items: List[HumanResourceBase]
    total: int
    page: int
    size: int
    total_pages: int
    has_next: bool
    has_prev: bool