from sqlalchemy import Column, String, Date, Integer, Text, JSON
from sqlalchemy import Index

from config.db_config import Base

from datetime import datetime

class HumanResource(Base):
    __tablename__ = "human_resource"
    
    id = Column(String(10), primary_key=True)
    name = Column(String(100), nullable=False)
    position = Column(String(100), nullable=False)
    department = Column(String(50), nullable=False)
    join_date = Column(Date, nullable=False)
    skills = Column(JSON)
    projects = Column(JSON)
    education_degree = Column(String(100))
    education_school = Column(String(100))
    education_graduation_year = Column(Integer)
    certifications = Column(JSON)
    languages = Column(JSON)
    profile_summary = Column(Text)
    created_at = Column(String(50), default=lambda: datetime.now().isoformat())
    updated_at = Column(String(50), default=lambda: datetime.now().isoformat(), onupdate=lambda: datetime.now().isoformat())
    
    # 인덱스 생성
    __table_args__ = (
        Index('idx_department', department),
        Index('idx_position', position),
        Index('idx_join_date', join_date),
    )